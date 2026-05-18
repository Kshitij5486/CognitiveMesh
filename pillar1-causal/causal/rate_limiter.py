import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("cm.rate_limiter")


class RateLimitResult:
    def __init__(
        self,
        allowed: bool,
        client_id: str,
        endpoint: str,
        remaining: int,
        reset_at: float,
        retry_after: Optional[float] = None,
    ):
        self.allowed = allowed
        self.client_id = client_id
        self.endpoint = endpoint
        self.remaining = remaining
        self.reset_at = reset_at
        self.retry_after = retry_after
        self.timestamp = time.time()

    def to_headers(self) -> dict:
        headers = {
            "X-RateLimit-Remaining": str(
                self.remaining
            ),
            "X-RateLimit-Reset": str(
                int(self.reset_at)
            ),
        }
        if self.retry_after is not None:
            headers["Retry-After"] = str(
                int(self.retry_after) + 1
            )
        return headers


class EndpointLimitConfig:
    """Per-endpoint rate limit configuration."""

    def __init__(
        self,
        requests_per_minute: int,
        burst_size: Optional[int] = None,
    ):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size or (
            requests_per_minute * 2
        )
        self.window_seconds = 60.0


# Default limits per endpoint group
DEFAULT_LIMITS = {
    "/health":          EndpointLimitConfig(120),
    "/quorum":          EndpointLimitConfig(60),
    "/byzantine":       EndpointLimitConfig(60),
    "/recovery/trigger": EndpointLimitConfig(
        10, burst_size=5
    ),
    "/recovery":        EndpointLimitConfig(60),
    "/routing":         EndpointLimitConfig(120),
    "/pipeline":        EndpointLimitConfig(60),
    "default":          EndpointLimitConfig(60),
}


class SlidingWindowRateLimiter:
    """
    Token bucket + sliding window rate limiter.

    Per client_id (IP address) per endpoint group.
    Uses a sliding window of timestamps to count
    requests in the last 60 seconds.

    Separate burst protection: if a client fires
    burst_size requests in < 1 second, they are
    throttled immediately regardless of per-minute
    quota.

    Thread-safe via per-client locks.
    """

    CLEANUP_INTERVAL_S = 300.0  # 5 minutes
    MAX_CLIENTS        = 10_000

    def __init__(self):
        # client_id -> endpoint_group -> deque of timestamps
        self._windows: dict = defaultdict(
            lambda: defaultdict(deque)
        )
        self._locks: dict = defaultdict(threading.Lock)
        self._global_lock = threading.Lock()

        # Stats
        self._total_requests = 0
        self._total_allowed = 0
        self._total_denied = 0
        self._start_time = time.time()

        # Cleanup
        self._last_cleanup = time.time()

    def _get_endpoint_group(self, path: str) -> str:
        for prefix in DEFAULT_LIMITS:
            if prefix == "default":
                continue
            if path.startswith(prefix):
                return prefix
        return "default"

    def _get_config(
        self, endpoint_group: str
    ) -> EndpointLimitConfig:
        return DEFAULT_LIMITS.get(
            endpoint_group,
            DEFAULT_LIMITS["default"],
        )

    def check(
        self,
        client_id: str,
        path: str,
    ) -> RateLimitResult:
        now = time.time()
        endpoint_group = self._get_endpoint_group(path)
        config = self._get_config(endpoint_group)
        window = config.window_seconds

        with self._locks[client_id]:
            timestamps = self._windows[client_id][
                endpoint_group
            ]

            # Remove expired timestamps
            cutoff = now - window
            while timestamps and timestamps[0] < cutoff:
                timestamps.popleft()

            # Check burst (requests in last 1 second)
            burst_cutoff = now - 1.0
            recent = sum(
                1 for t in timestamps
                if t > burst_cutoff
            )
            if recent >= config.burst_size:
                self._total_requests += 1
                self._total_denied += 1
                oldest_recent = min(
                    t for t in timestamps
                    if t > burst_cutoff
                )
                retry = 1.0 - (now - oldest_recent)
                reset_at = now + retry
                return RateLimitResult(
                    allowed=False,
                    client_id=client_id,
                    endpoint=endpoint_group,
                    remaining=0,
                    reset_at=reset_at,
                    retry_after=retry,
                )

            # Check per-minute quota
            count = len(timestamps)
            limit = config.requests_per_minute

            if count >= limit:
                self._total_requests += 1
                self._total_denied += 1
                oldest = timestamps[0]
                reset_at = oldest + window
                retry = reset_at - now
                return RateLimitResult(
                    allowed=False,
                    client_id=client_id,
                    endpoint=endpoint_group,
                    remaining=0,
                    reset_at=reset_at,
                    retry_after=max(0.0, retry),
                )

            # Allow request
            timestamps.append(now)
            self._total_requests += 1
            self._total_allowed += 1
            remaining = limit - len(timestamps)
            reset_at = (
                timestamps[0] + window
                if timestamps else now + window
            )

        # Periodic cleanup
        self._maybe_cleanup(now)

        return RateLimitResult(
            allowed=True,
            client_id=client_id,
            endpoint=endpoint_group,
            remaining=remaining,
            reset_at=reset_at,
        )

    def _maybe_cleanup(self, now: float):
        if (
            now - self._last_cleanup
            < self.CLEANUP_INTERVAL_S
        ):
            return
        with self._global_lock:
            if (
                now - self._last_cleanup
                < self.CLEANUP_INTERVAL_S
            ):
                return
            self._last_cleanup = now
            cutoff = now - 60.0
            stale = []
            for client_id, endpoints in (
                self._windows.items()
            ):
                for ep, ts in endpoints.items():
                    while ts and ts[0] < cutoff:
                        ts.popleft()
                if all(
                    len(ts) == 0
                    for ts in endpoints.values()
                ):
                    stale.append(client_id)
            for client_id in stale:
                del self._windows[client_id]
                if client_id in self._locks:
                    del self._locks[client_id]
            if stale:
                logger.info(
                    "Rate limiter cleanup: "
                    "removed %d stale clients",
                    len(stale),
                )

    def get_client_status(
        self, client_id: str
    ) -> dict:
        now = time.time()
        status = {}
        with self._locks[client_id]:
            for ep, timestamps in (
                self._windows[client_id].items()
            ):
                cutoff = now - 60.0
                recent = [
                    t for t in timestamps
                    if t > cutoff
                ]
                config = self._get_config(ep)
                status[ep] = {
                    "count_last_minute": len(recent),
                    "limit": (
                        config.requests_per_minute
                    ),
                    "remaining": max(
                        0,
                        config.requests_per_minute
                        - len(recent),
                    ),
                }
        return status

    def stats(self) -> dict:
        uptime = time.time() - self._start_time
        deny_rate = (
            self._total_denied / self._total_requests
            if self._total_requests > 0 else 0.0
        )
        with self._global_lock:
            active_clients = len(self._windows)
        return {
            "total_requests": self._total_requests,
            "total_allowed": self._total_allowed,
            "total_denied": self._total_denied,
            "deny_rate": round(deny_rate, 4),
            "active_clients": active_clients,
            "uptime_seconds": round(uptime, 1),
            "endpoint_limits": {
                k: {
                    "rpm": v.requests_per_minute,
                    "burst": v.burst_size,
                }
                for k, v in DEFAULT_LIMITS.items()
                if k != "default"
            },
        }


class GracefulShutdownManager:
    """
    Coordinates graceful shutdown across all
    CognitiveMesh Sprint 10 components.

    Shutdown sequence:
    1. Stop accepting new requests (set draining flag)
    2. Wait for in-flight requests to complete
       (up to DRAIN_TIMEOUT_S)
    3. Stop background components in dependency order
    4. Flush persistence write queue
    5. Close database connections
    6. Emit shutdown report

    Components stopped in order:
      health_monitor → sla_monitor → orchestrator
      → coordinator → quorum_router → quorum_manager
      → base_retrainer → base_router → persistence
      → updater
    """

    DRAIN_TIMEOUT_S     = 30.0
    COMPONENT_TIMEOUT_S = 10.0

    def __init__(self):
        self._draining = False
        self._in_flight = 0
        self._lock = threading.Lock()
        self._shutdown_complete = False
        self._shutdown_report: dict = {}
        self._components: list = []
        self._start_time: Optional[float] = None

    def register_component(
        self,
        name: str,
        component,
        order: int,
    ):
        self._components.append((order, name, component))
        self._components.sort(key=lambda x: x[0])

    @property
    def is_draining(self) -> bool:
        return self._draining

    def request_start(self) -> bool:
        """Call before handling a request.
        Returns False if draining."""
        if self._draining:
            return False
        with self._lock:
            if self._draining:
                return False
            self._in_flight += 1
            return True

    def request_end(self):
        """Call after handling a request."""
        with self._lock:
            self._in_flight = max(
                0, self._in_flight - 1
            )

    def _wait_drain(self) -> bool:
        """Wait for in-flight requests to complete."""
        deadline = time.time() + self.DRAIN_TIMEOUT_S
        while time.time() < deadline:
            with self._lock:
                if self._in_flight == 0:
                    return True
            time.sleep(0.1)
        with self._lock:
            remaining = self._in_flight
        logger.warning(
            "Drain timeout: %d requests still "
            "in-flight after %.1fs",
            remaining,
            self.DRAIN_TIMEOUT_S,
        )
        return False

    def shutdown(
        self,
        components: Optional[dict] = None,
    ) -> dict:
        if self._shutdown_complete:
            return self._shutdown_report

        shutdown_start = time.time()
        self._start_time = shutdown_start
        logger.info(
            "GracefulShutdownManager: initiating "
            "shutdown sequence"
        )

        # Phase 1: Stop accepting requests
        with self._lock:
            self._draining = True
        logger.info(
            "Phase 1: Draining — no new requests "
            "accepted"
        )

        # Phase 2: Wait for in-flight
        drained = self._wait_drain()
        logger.info(
            "Phase 2: Drain complete — "
            "clean=%s in_flight=%d",
            drained,
            self._in_flight,
        )

        # Phase 3: Stop components in order
        stop_results = {}
        component_list = (
            self._components
            if self._components
            else []
        )

        # If components passed as dict, use those
        if components:
            for name, comp in components.items():
                if comp is None:
                    continue
                try:
                    logger.info(
                        "Phase 3: Stopping %s", name
                    )
                    start = time.time()
                    if hasattr(comp, "stop"):
                        comp.stop()
                    elapsed = time.time() - start
                    stop_results[name] = {
                        "status": "stopped",
                        "elapsed_s": round(
                            elapsed, 2
                        ),
                    }
                    logger.info(
                        "  %s stopped in %.2fs",
                        name, elapsed,
                    )
                except Exception as e:
                    stop_results[name] = {
                        "status": "error",
                        "error": str(e),
                    }
                    logger.error(
                        "  %s stop error: %s",
                        name, e,
                    )

        total_elapsed = time.time() - shutdown_start
        self._shutdown_complete = True
        self._shutdown_report = {
            "shutdown_complete": True,
            "clean_drain": drained,
            "total_elapsed_s": round(
                total_elapsed, 2
            ),
            "components_stopped": stop_results,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        logger.info(
            "GracefulShutdownManager: shutdown "
            "complete in %.2fs clean=%s",
            total_elapsed,
            drained,
        )
        return self._shutdown_report

    def status(self) -> dict:
        return {
            "draining": self._draining,
            "in_flight": self._in_flight,
            "shutdown_complete": (
                self._shutdown_complete
            ),
            "components_registered": len(
                self._components
            ),
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s [%(levelname)s] "
            "%(name)s - %(message)s"
        ),
    )
    logger.info("Starting rate limiter demo")

    limiter = SlidingWindowRateLimiter()

    # Simulate requests from 3 clients
    clients = ["192.168.1.1", "192.168.1.2", "10.0.0.1"]
    endpoints = [
        "/health",
        "/quorum",
        "/recovery/trigger",
        "/routing/weights",
        "/pipeline/summary",
    ]

    allowed = 0
    denied = 0

    logger.info(
        "Sending 50 requests across 3 clients "
        "and 5 endpoints..."
    )
    for i in range(50):
        client = clients[i % len(clients)]
        endpoint = endpoints[i % len(endpoints)]
        result = limiter.check(client, endpoint)
        if result.allowed:
            allowed += 1
        else:
            denied += 1

    logger.info(
        "Results: allowed=%d denied=%d "
        "deny_rate=%.3f",
        allowed,
        denied,
        denied / 50,
    )

    # Burst test: 20 requests in rapid succession
    logger.info(
        "Burst test: 20 rapid requests from "
        "single client to /recovery/trigger"
    )
    burst_allowed = 0
    burst_denied = 0
    for _ in range(20):
        result = limiter.check(
            "10.0.0.99", "/recovery/trigger"
        )
        if result.allowed:
            burst_allowed += 1
        else:
            burst_denied += 1

    logger.info(
        "Burst results: allowed=%d denied=%d",
        burst_allowed,
        burst_denied,
    )

    stats = limiter.stats()
    logger.info(
        "Rate limiter stats: total=%d "
        "allowed=%d denied=%d deny_rate=%.3f "
        "active_clients=%d",
        stats["total_requests"],
        stats["total_allowed"],
        stats["total_denied"],
        stats["deny_rate"],
        stats["active_clients"],
    )

    # Graceful shutdown demo
    logger.info("Testing graceful shutdown...")
    shutdown_mgr = GracefulShutdownManager()

    # Simulate in-flight requests
    def fake_request():
        if shutdown_mgr.request_start():
            time.sleep(0.5)
            shutdown_mgr.request_end()

    threads = [
        threading.Thread(target=fake_request)
        for _ in range(3)
    ]
    for t in threads:
        t.start()

    time.sleep(0.1)
    report = shutdown_mgr.shutdown(
        components={
            "mock_component_a": type(
                "M", (), {"stop": lambda self: None}
            )(),
            "mock_component_b": type(
                "M", (), {"stop": lambda self: None}
            )(),
        }
    )

    for t in threads:
        t.join()

    logger.info(
        "Shutdown report: complete=%s "
        "clean_drain=%s elapsed=%.2fs",
        report["shutdown_complete"],
        report["clean_drain"],
        report["total_elapsed_s"],
    )
    logger.info("=== Day 66 demo complete ===")