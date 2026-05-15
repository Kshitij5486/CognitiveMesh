CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET pg_stat_statements.track = 'all';
ALTER SYSTEM SET pg_stat_statements.max = 10000;
ALTER SYSTEM SET log_min_duration_statement = 0;
ALTER SYSTEM SET log_lock_waits = on;
ALTER SYSTEM SET deadlock_timeout = '1s';
ALTER SYSTEM SET track_activities = on;
ALTER SYSTEM SET track_counts = on;
ALTER SYSTEM SET track_io_timing = on;
ALTER SYSTEM SET max_connections = 200;

CREATE TABLE IF NOT EXISTS cm_workload_test (
    id          SERIAL PRIMARY KEY,
    node_id     VARCHAR(20) NOT NULL,
    payload     TEXT,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_workload_node ON cm_workload_test(node_id);
CREATE INDEX IF NOT EXISTS idx_workload_created ON cm_workload_test(created_at);