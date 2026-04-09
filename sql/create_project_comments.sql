-- Migration manuelle: project_comments + project_activity_log
-- À exécuter si alembic upgrade échoue avec DuplicateObjectError

-- ── project_comments ─────────────────────────────────────────────────────────

CREATE TYPE IF NOT EXISTS comment_type_enum AS ENUM ('GENERAL', 'GENE', 'COMPARISON', 'PATHWAY');

CREATE TABLE IF NOT EXISTS project_comments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id UUID NOT NULL,
    comment_type comment_type_enum NOT NULL DEFAULT 'GENERAL',
    target_id VARCHAR(255),
    content TEXT NOT NULL,
    parent_id UUID REFERENCES project_comments(id) ON DELETE CASCADE,
    is_resolved BOOLEAN NOT NULL DEFAULT false,
    extra_metadata JSON NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_project_comments_project_id   ON project_comments(project_id);
CREATE INDEX IF NOT EXISTS ix_project_comments_user_id      ON project_comments(user_id);
CREATE INDEX IF NOT EXISTS ix_project_comments_target_id    ON project_comments(target_id);
CREATE INDEX IF NOT EXISTS ix_project_comments_parent_id    ON project_comments(parent_id);
CREATE INDEX IF NOT EXISTS ix_project_comments_project_type ON project_comments(project_id, comment_type);
CREATE INDEX IF NOT EXISTS ix_project_comments_target       ON project_comments(project_id, target_id);

-- ── project_activity_log ──────────────────────────────────────────────────────

DO $$ BEGIN
    CREATE TYPE activity_event_type_enum AS ENUM (
        'dataset_uploaded', 'dataset_deleted', 'comparison_created',
        'enrichment_run', 'clustering_run', 'gsea_run', 'go_enrichment_run',
        'bookmark_created', 'bookmark_batch_created', 'bookmark_deleted',
        'gene_list_created', 'comment_added', 'project_shared'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

CREATE TABLE IF NOT EXISTS project_activity_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id UUID NOT NULL,
    event_type activity_event_type_enum NOT NULL,
    entity_type VARCHAR(100),
    entity_id VARCHAR(255),
    entity_name VARCHAR(500),
    extra_metadata JSON NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_project_activity_log_project_id   ON project_activity_log(project_id);
CREATE INDEX IF NOT EXISTS ix_project_activity_log_user_id      ON project_activity_log(user_id);
CREATE INDEX IF NOT EXISTS ix_project_activity_log_event_type   ON project_activity_log(event_type);
CREATE INDEX IF NOT EXISTS ix_project_activity_log_project_event ON project_activity_log(project_id, event_type);
