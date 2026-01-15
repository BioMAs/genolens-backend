-- =====================================================
-- GenoLens v2: Complete Database Schema
-- =====================================================
-- This SQL script creates all necessary tables for GenoLens v2
-- including the new deg_genes table for performance optimization.
--
-- IMPORTANT: This script creates NEW tables for GenoLens v2.
-- It does NOT modify existing tables (sequencing_projects, etc.)
-- used by the legacy application. Both applications coexist
-- and share only the auth.users table from Supabase Auth.
--
-- Execute this script in your Supabase SQL Editor.
-- =====================================================

-- Step 1: Create ENUM types
-- =====================================================

-- User role enum (if not exists)
DO $$ BEGIN
    CREATE TYPE user_role AS ENUM ('ADMIN', 'USER', 'SUBSCRIBER', 'ANALYST', 'VIEWER');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Dataset type enum
DO $$ BEGIN
    CREATE TYPE dataset_type AS ENUM ('MATRIX', 'DEG', 'ENRICHMENT', 'METADATA', 'METADATA_SAMPLE', 'METADATA_CONTRAST');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Dataset status enum
DO $$ BEGIN
    CREATE TYPE dataset_status AS ENUM ('PENDING', 'PROCESSING', 'READY', 'FAILED', 'ARCHIVED');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Step 2: Create main tables
-- =====================================================

-- Projects table
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Create indexes for projects
CREATE INDEX IF NOT EXISTS ix_projects_owner_id ON projects(owner_id);
CREATE INDEX IF NOT EXISTS ix_projects_owner_created ON projects(owner_id, created_at);

-- Datasets table
CREATE TABLE IF NOT EXISTS datasets (
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

-- Create indexes for datasets
CREATE INDEX IF NOT EXISTS ix_datasets_project_id ON datasets(project_id);
CREATE INDEX IF NOT EXISTS ix_datasets_type ON datasets(type);
CREATE INDEX IF NOT EXISTS ix_datasets_status ON datasets(status);
CREATE INDEX IF NOT EXISTS ix_datasets_project_type ON datasets(project_id, type);

-- Samples table
CREATE TABLE IF NOT EXISTS samples (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    sample_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Create indexes for samples
CREATE INDEX IF NOT EXISTS ix_samples_project_id ON samples(project_id);
CREATE INDEX IF NOT EXISTS ix_samples_project_name ON samples(project_id, name);

-- Project members table (for sharing)
CREATE TABLE IF NOT EXISTS project_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id UUID NOT NULL,
    access_level user_role NOT NULL DEFAULT 'VIEWER',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Create indexes for project_members
CREATE INDEX IF NOT EXISTS ix_project_members_project_id ON project_members(project_id);
CREATE INDEX IF NOT EXISTS ix_project_members_user ON project_members(user_id);
CREATE UNIQUE INDEX IF NOT EXISTS ix_project_members_project_user ON project_members(project_id, user_id);

-- Step 3: Create DEG genes table for performance
-- =====================================================

CREATE TABLE IF NOT EXISTS deg_genes (
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

-- Create indexes for deg_genes (critical for performance)
CREATE INDEX IF NOT EXISTS idx_deg_genes_dataset ON deg_genes(dataset_id);
CREATE INDEX IF NOT EXISTS idx_deg_genes_comparison ON deg_genes(comparison_name);
CREATE INDEX IF NOT EXISTS idx_deg_genes_dataset_comparison ON deg_genes(dataset_id, comparison_name);
CREATE INDEX IF NOT EXISTS idx_deg_genes_padj ON deg_genes(padj);
CREATE INDEX IF NOT EXISTS idx_deg_genes_regulation ON deg_genes(regulation);
CREATE INDEX IF NOT EXISTS idx_deg_genes_gene_id ON deg_genes(gene_id);
CREATE INDEX IF NOT EXISTS idx_deg_genes_dataset_comparison_padj ON deg_genes(dataset_id, comparison_name, padj);

-- Step 4: Enable Row Level Security (RLS)
-- =====================================================

-- Enable RLS on all tables
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE datasets ENABLE ROW LEVEL SECURITY;
ALTER TABLE samples ENABLE ROW LEVEL SECURITY;
ALTER TABLE project_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE deg_genes ENABLE ROW LEVEL SECURITY;

-- Step 5: Create RLS Policies
-- =====================================================

-- Projects: Users can view their own projects and shared projects
DROP POLICY IF EXISTS "Users can view their own projects" ON projects;
CREATE POLICY "Users can view their own projects"
    ON projects FOR SELECT
    USING (
        owner_id = auth.uid() OR
        EXISTS (
            SELECT 1 FROM project_members
            WHERE project_members.project_id = projects.id
            AND project_members.user_id = auth.uid()
        )
    );

-- Projects: Users can insert their own projects
DROP POLICY IF EXISTS "Users can create their own projects" ON projects;
CREATE POLICY "Users can create their own projects"
    ON projects FOR INSERT
    WITH CHECK (owner_id = auth.uid());

-- Projects: Users can update their own projects
DROP POLICY IF EXISTS "Users can update their own projects" ON projects;
CREATE POLICY "Users can update their own projects"
    ON projects FOR UPDATE
    USING (owner_id = auth.uid());

-- Projects: Users can delete their own projects
DROP POLICY IF EXISTS "Users can delete their own projects" ON projects;
CREATE POLICY "Users can delete their own projects"
    ON projects FOR DELETE
    USING (owner_id = auth.uid());

-- Datasets: Users can view datasets from their projects
DROP POLICY IF EXISTS "Users can view datasets from their projects" ON datasets;
CREATE POLICY "Users can view datasets from their projects"
    ON datasets FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = datasets.project_id
            AND (
                projects.owner_id = auth.uid() OR
                EXISTS (
                    SELECT 1 FROM project_members
                    WHERE project_members.project_id = projects.id
                    AND project_members.user_id = auth.uid()
                )
            )
        )
    );

-- Datasets: Users can insert datasets for their projects
DROP POLICY IF EXISTS "Users can insert datasets for their projects" ON datasets;
CREATE POLICY "Users can insert datasets for their projects"
    ON datasets FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = datasets.project_id
            AND projects.owner_id = auth.uid()
        )
    );

-- Datasets: Users can update datasets for their projects
DROP POLICY IF EXISTS "Users can update datasets for their projects" ON datasets;
CREATE POLICY "Users can update datasets for their projects"
    ON datasets FOR UPDATE
    USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = datasets.project_id
            AND projects.owner_id = auth.uid()
        )
    );

-- Datasets: Users can delete datasets for their projects
DROP POLICY IF EXISTS "Users can delete datasets for their projects" ON datasets;
CREATE POLICY "Users can delete datasets for their projects"
    ON datasets FOR DELETE
    USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = datasets.project_id
            AND projects.owner_id = auth.uid()
        )
    );

-- Samples: Users can view samples from their projects
DROP POLICY IF EXISTS "Users can view samples from their projects" ON samples;
CREATE POLICY "Users can view samples from their projects"
    ON samples FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = samples.project_id
            AND (
                projects.owner_id = auth.uid() OR
                EXISTS (
                    SELECT 1 FROM project_members
                    WHERE project_members.project_id = projects.id
                    AND project_members.user_id = auth.uid()
                )
            )
        )
    );

-- Samples: Users can insert samples for their projects
DROP POLICY IF EXISTS "Users can insert samples for their projects" ON samples;
CREATE POLICY "Users can insert samples for their projects"
    ON samples FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = samples.project_id
            AND projects.owner_id = auth.uid()
        )
    );

-- Samples: Users can update samples for their projects
DROP POLICY IF EXISTS "Users can update samples for their projects" ON samples;
CREATE POLICY "Users can update samples for their projects"
    ON samples FOR UPDATE
    USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = samples.project_id
            AND projects.owner_id = auth.uid()
        )
    );

-- Samples: Users can delete samples for their projects
DROP POLICY IF EXISTS "Users can delete samples for their projects" ON samples;
CREATE POLICY "Users can delete samples for their projects"
    ON samples FOR DELETE
    USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = samples.project_id
            AND projects.owner_id = auth.uid()
        )
    );

-- Project Members: Users can view members of their projects
DROP POLICY IF EXISTS "Users can view project members" ON project_members;
CREATE POLICY "Users can view project members"
    ON project_members FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = project_members.project_id
            AND (
                projects.owner_id = auth.uid() OR
                project_members.user_id = auth.uid()
            )
        )
    );

-- Project Members: Project owners can add members
DROP POLICY IF EXISTS "Project owners can add members" ON project_members;
CREATE POLICY "Project owners can add members"
    ON project_members FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = project_members.project_id
            AND projects.owner_id = auth.uid()
        )
    );

-- Project Members: Project owners can update member access
DROP POLICY IF EXISTS "Project owners can update members" ON project_members;
CREATE POLICY "Project owners can update members"
    ON project_members FOR UPDATE
    USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = project_members.project_id
            AND projects.owner_id = auth.uid()
        )
    );

-- Project Members: Project owners can remove members
DROP POLICY IF EXISTS "Project owners can remove members" ON project_members;
CREATE POLICY "Project owners can remove members"
    ON project_members FOR DELETE
    USING (
        EXISTS (
            SELECT 1 FROM projects
            WHERE projects.id = project_members.project_id
            AND projects.owner_id = auth.uid()
        )
    );

-- DEG Genes: Users can view DEG genes from their projects
DROP POLICY IF EXISTS "Users can view DEG genes from their projects" ON deg_genes;
CREATE POLICY "Users can view DEG genes from their projects"
    ON deg_genes FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM datasets d
            JOIN projects p ON d.project_id = p.id
            WHERE d.id = deg_genes.dataset_id
            AND (
                p.owner_id = auth.uid() OR
                EXISTS (
                    SELECT 1 FROM project_members
                    WHERE project_members.project_id = p.id
                    AND project_members.user_id = auth.uid()
                )
            )
        )
    );

-- DEG Genes: Users can insert DEG genes for their datasets
DROP POLICY IF EXISTS "Users can insert DEG genes for their datasets" ON deg_genes;
CREATE POLICY "Users can insert DEG genes for their datasets"
    ON deg_genes FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM datasets d
            JOIN projects p ON d.project_id = p.id
            WHERE d.id = deg_genes.dataset_id
            AND p.owner_id = auth.uid()
        )
    );

-- DEG Genes: Users can delete DEG genes from their datasets
DROP POLICY IF EXISTS "Users can delete DEG genes from their datasets" ON deg_genes;
CREATE POLICY "Users can delete DEG genes from their datasets"
    ON deg_genes FOR DELETE
    USING (
        EXISTS (
            SELECT 1 FROM datasets d
            JOIN projects p ON d.project_id = p.id
            WHERE d.id = deg_genes.dataset_id
            AND p.owner_id = auth.uid()
        )
    );

-- Step 6: Create triggers for updated_at
-- =====================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for projects
DROP TRIGGER IF EXISTS update_projects_updated_at ON projects;
CREATE TRIGGER update_projects_updated_at
    BEFORE UPDATE ON projects
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Triggers for datasets
DROP TRIGGER IF EXISTS update_datasets_updated_at ON datasets;
CREATE TRIGGER update_datasets_updated_at
    BEFORE UPDATE ON datasets
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Triggers for samples
DROP TRIGGER IF EXISTS update_samples_updated_at ON samples;
CREATE TRIGGER update_samples_updated_at
    BEFORE UPDATE ON samples
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Triggers for project_members
DROP TRIGGER IF EXISTS update_project_members_updated_at ON project_members;
CREATE TRIGGER update_project_members_updated_at
    BEFORE UPDATE ON project_members
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- Verification Queries (Optional)
-- =====================================================
-- Uncomment these to verify the tables were created correctly

-- Show all tables
-- SELECT table_name
-- FROM information_schema.tables
-- WHERE table_schema = 'public'
-- AND table_name IN ('projects', 'datasets', 'samples', 'project_members', 'deg_genes')
-- ORDER BY table_name;

-- Show deg_genes structure
-- SELECT column_name, data_type, is_nullable
-- FROM information_schema.columns
-- WHERE table_name = 'deg_genes'
-- ORDER BY ordinal_position;

-- Show all indexes on deg_genes
-- SELECT indexname, indexdef
-- FROM pg_indexes
-- WHERE tablename = 'deg_genes';

-- Show RLS policies on deg_genes
-- SELECT policyname, cmd, qual
-- FROM pg_policies
-- WHERE tablename = 'deg_genes';

-- =====================================================
-- Success Message
-- =====================================================
DO $$
BEGIN
    RAISE NOTICE 'GenoLens v2 schema created successfully!';
    RAISE NOTICE 'Tables created: projects, datasets, samples, project_members, deg_genes';
    RAISE NOTICE 'Next steps:';
    RAISE NOTICE '1. Restart backend: docker-compose restart backend celery';
    RAISE NOTICE '2. Upload a new dataset to test performance';
    RAISE NOTICE '3. For existing datasets, use: POST /datasets/{id}/reprocess';
END $$;
