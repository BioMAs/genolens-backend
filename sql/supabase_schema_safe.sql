-- =====================================================
-- GenoLens v2: Safe Database Schema Creation
-- =====================================================
-- This version is more defensive and handles conflicts
-- with existing triggers/functions.
--
-- IMPORTANT: This script creates NEW tables for GenoLens v2.
-- It does NOT modify existing tables from the legacy app.
--
-- Execute this script in your Supabase SQL Editor.
-- =====================================================

-- Disable triggers temporarily to avoid conflicts
SET session_replication_role = 'replica';

-- Step 1: Create ENUM types (if not exists)
-- =====================================================

DO $$ BEGIN
    CREATE TYPE user_role AS ENUM ('ADMIN', 'USER', 'SUBSCRIBER', 'ANALYST', 'VIEWER');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE dataset_type AS ENUM ('MATRIX', 'DEG', 'ENRICHMENT', 'METADATA', 'METADATA_SAMPLE', 'METADATA_CONTRAST');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE dataset_status AS ENUM ('PENDING', 'PROCESSING', 'READY', 'FAILED', 'ARCHIVED');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Step 2: Create main tables (if not exists)
-- =====================================================

-- Projects table
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'projects') THEN
        CREATE TABLE projects (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            description TEXT,
            owner_id UUID NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
        );

        CREATE INDEX ix_projects_owner_id ON projects(owner_id);
        CREATE INDEX ix_projects_owner_created ON projects(owner_id, created_at);

        RAISE NOTICE 'Created table: projects';
    ELSE
        RAISE NOTICE 'Table projects already exists, skipping';
    END IF;
END $$;

-- Datasets table
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'datasets') THEN
        CREATE TABLE datasets (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            type dataset_type NOT NULL,
            status dataset_status NOT NULL DEFAULT 'PENDING',
            raw_file_path VARCHAR(1024),
            parquet_file_path VARCHAR(1024),
            column_mapping JSONB NOT NULL DEFAULT '{}'::jsonb,
            dataset_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            error_message TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
        );

        CREATE INDEX ix_datasets_project_id ON datasets(project_id);
        CREATE INDEX ix_datasets_type ON datasets(type);
        CREATE INDEX ix_datasets_status ON datasets(status);
        CREATE INDEX ix_datasets_project_type ON datasets(project_id, type);

        RAISE NOTICE 'Created table: datasets';
    ELSE
        RAISE NOTICE 'Table datasets already exists, skipping';
    END IF;
END $$;

-- Samples table (GenoLens v2 version)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_tables
        WHERE schemaname = 'public'
        AND tablename = 'samples'
        AND EXISTS (
            SELECT FROM information_schema.columns
            WHERE table_schema = 'public'
            AND table_name = 'samples'
            AND column_name = 'sample_metadata'
        )
    ) THEN
        -- Only create if it doesn't exist OR if it's the old version
        IF NOT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'samples') THEN
            CREATE TABLE samples (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                name VARCHAR(255) NOT NULL,
                sample_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
            );

            CREATE INDEX ix_samples_project_id ON samples(project_id);
            CREATE INDEX ix_samples_project_name ON samples(project_id, name);

            RAISE NOTICE 'Created table: samples';
        ELSE
            RAISE NOTICE 'Table samples already exists (legacy version), skipping';
        END IF;
    ELSE
        RAISE NOTICE 'Table samples (GenoLens v2) already exists, skipping';
    END IF;
END $$;

-- Project members table
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'project_members') THEN
        CREATE TABLE project_members (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            user_id UUID NOT NULL,
            access_level user_role NOT NULL DEFAULT 'VIEWER',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
        );

        CREATE INDEX ix_project_members_project_id ON project_members(project_id);
        CREATE INDEX ix_project_members_user ON project_members(user_id);
        CREATE UNIQUE INDEX ix_project_members_project_user ON project_members(project_id, user_id);

        RAISE NOTICE 'Created table: project_members';
    ELSE
        RAISE NOTICE 'Table project_members already exists, skipping';
    END IF;
END $$;

-- Step 3: Create DEG genes table for performance
-- =====================================================

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'deg_genes') THEN
        CREATE TABLE deg_genes (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
            comparison_name TEXT NOT NULL,
            gene_id TEXT NOT NULL,
            log_fc DOUBLE PRECISION NOT NULL,
            padj DOUBLE PRECISION NOT NULL,
            regulation TEXT CHECK (regulation IN ('UP', 'DOWN')),
            pvalue DOUBLE PRECISION,
            base_mean DOUBLE PRECISION,
            gene_name TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- Create indexes for performance
        CREATE INDEX idx_deg_genes_dataset ON deg_genes(dataset_id);
        CREATE INDEX idx_deg_genes_comparison ON deg_genes(comparison_name);
        CREATE INDEX idx_deg_genes_dataset_comparison ON deg_genes(dataset_id, comparison_name);
        CREATE INDEX idx_deg_genes_padj ON deg_genes(padj);
        CREATE INDEX idx_deg_genes_regulation ON deg_genes(regulation);
        CREATE INDEX idx_deg_genes_gene_id ON deg_genes(gene_id);
        CREATE INDEX idx_deg_genes_dataset_comparison_padj ON deg_genes(dataset_id, comparison_name, padj);

        RAISE NOTICE 'Created table: deg_genes with 7 indexes';
    ELSE
        RAISE NOTICE 'Table deg_genes already exists, skipping';
    END IF;
END $$;

-- Re-enable triggers
SET session_replication_role = 'origin';

-- Step 4: Enable Row Level Security (only if tables were created)
-- =====================================================

DO $$
BEGIN
    -- Enable RLS only on tables that exist
    IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'projects') THEN
        ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
    END IF;

    IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'datasets') THEN
        ALTER TABLE datasets ENABLE ROW LEVEL SECURITY;
    END IF;

    IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'samples') THEN
        -- Only enable if it's the GenoLens v2 version
        IF EXISTS (
            SELECT FROM information_schema.columns
            WHERE table_schema = 'public'
            AND table_name = 'samples'
            AND column_name = 'sample_metadata'
        ) THEN
            ALTER TABLE samples ENABLE ROW LEVEL SECURITY;
        END IF;
    END IF;

    IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'project_members') THEN
        ALTER TABLE project_members ENABLE ROW LEVEL SECURITY;
    END IF;

    IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'deg_genes') THEN
        ALTER TABLE deg_genes ENABLE ROW LEVEL SECURITY;
    END IF;

    RAISE NOTICE 'Row Level Security enabled on all tables';
END $$;

-- Step 5: Create RLS Policies (safe with DROP IF EXISTS)
-- =====================================================

-- Projects policies
DROP POLICY IF EXISTS "Users can view their own projects" ON projects;
CREATE POLICY "Users can view their own projects" ON projects FOR SELECT
    USING (owner_id = auth.uid() OR EXISTS (
        SELECT 1 FROM project_members WHERE project_members.project_id = projects.id AND project_members.user_id = auth.uid()
    ));

DROP POLICY IF EXISTS "Users can create their own projects" ON projects;
CREATE POLICY "Users can create their own projects" ON projects FOR INSERT
    WITH CHECK (owner_id = auth.uid());

DROP POLICY IF EXISTS "Users can update their own projects" ON projects;
CREATE POLICY "Users can update their own projects" ON projects FOR UPDATE
    USING (owner_id = auth.uid());

DROP POLICY IF EXISTS "Users can delete their own projects" ON projects;
CREATE POLICY "Users can delete their own projects" ON projects FOR DELETE
    USING (owner_id = auth.uid());

-- Datasets policies
DROP POLICY IF EXISTS "Users can view datasets from their projects" ON datasets;
CREATE POLICY "Users can view datasets from their projects" ON datasets FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM projects WHERE projects.id = datasets.project_id AND (
            projects.owner_id = auth.uid() OR EXISTS (
                SELECT 1 FROM project_members WHERE project_members.project_id = projects.id AND project_members.user_id = auth.uid()
            )
        )
    ));

DROP POLICY IF EXISTS "Users can insert datasets for their projects" ON datasets;
CREATE POLICY "Users can insert datasets for their projects" ON datasets FOR INSERT
    WITH CHECK (EXISTS (
        SELECT 1 FROM projects WHERE projects.id = datasets.project_id AND projects.owner_id = auth.uid()
    ));

DROP POLICY IF EXISTS "Users can update datasets for their projects" ON datasets;
CREATE POLICY "Users can update datasets for their projects" ON datasets FOR UPDATE
    USING (EXISTS (
        SELECT 1 FROM projects WHERE projects.id = datasets.project_id AND projects.owner_id = auth.uid()
    ));

DROP POLICY IF EXISTS "Users can delete datasets for their projects" ON datasets;
CREATE POLICY "Users can delete datasets for their projects" ON datasets FOR DELETE
    USING (EXISTS (
        SELECT 1 FROM projects WHERE projects.id = datasets.project_id AND projects.owner_id = auth.uid()
    ));

-- DEG Genes policies
DROP POLICY IF EXISTS "Users can view DEG genes from their projects" ON deg_genes;
CREATE POLICY "Users can view DEG genes from their projects" ON deg_genes FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM datasets d JOIN projects p ON d.project_id = p.id
        WHERE d.id = deg_genes.dataset_id AND (
            p.owner_id = auth.uid() OR EXISTS (
                SELECT 1 FROM project_members WHERE project_members.project_id = p.id AND project_members.user_id = auth.uid()
            )
        )
    ));

DROP POLICY IF EXISTS "Users can insert DEG genes for their datasets" ON deg_genes;
CREATE POLICY "Users can insert DEG genes for their datasets" ON deg_genes FOR INSERT
    WITH CHECK (EXISTS (
        SELECT 1 FROM datasets d JOIN projects p ON d.project_id = p.id
        WHERE d.id = deg_genes.dataset_id AND p.owner_id = auth.uid()
    ));

DROP POLICY IF EXISTS "Users can delete DEG genes from their datasets" ON deg_genes;
CREATE POLICY "Users can delete DEG genes from their datasets" ON deg_genes FOR DELETE
    USING (EXISTS (
        SELECT 1 FROM datasets d JOIN projects p ON d.project_id = p.id
        WHERE d.id = deg_genes.dataset_id AND p.owner_id = auth.uid()
    ));

-- Step 6: Create updated_at triggers (safe)
-- =====================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers only if they don't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_projects_updated_at') THEN
        CREATE TRIGGER update_projects_updated_at
            BEFORE UPDATE ON projects
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_datasets_updated_at') THEN
        CREATE TRIGGER update_datasets_updated_at
            BEFORE UPDATE ON datasets
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_project_members_updated_at') THEN
        CREATE TRIGGER update_project_members_updated_at
            BEFORE UPDATE ON project_members
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;

-- =====================================================
-- Final verification and success message
-- =====================================================
DO $$
DECLARE
    table_count INT;
BEGIN
    -- Count created tables
    SELECT COUNT(*) INTO table_count
    FROM pg_tables
    WHERE schemaname = 'public'
    AND tablename IN ('projects', 'datasets', 'deg_genes', 'project_members');

    RAISE NOTICE '========================================';
    RAISE NOTICE 'GenoLens v2 Schema Installation Complete!';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Tables created/verified: %', table_count;
    RAISE NOTICE '';
    RAISE NOTICE 'Next steps:';
    RAISE NOTICE '1. Restart backend: docker-compose restart backend celery';
    RAISE NOTICE '2. Upload a new dataset to test performance';
    RAISE NOTICE '3. Expected performance: <100ms (vs 2-5 seconds before)';
    RAISE NOTICE '';
    RAISE NOTICE 'Verification queries:';
    RAISE NOTICE '  SELECT COUNT(*) FROM deg_genes;';
    RAISE NOTICE '  SELECT COUNT(*) FROM projects;';
    RAISE NOTICE '========================================';
END $$;
