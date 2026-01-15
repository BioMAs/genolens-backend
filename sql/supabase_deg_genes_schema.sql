-- =====================================================
-- GenoLens v2: DEG Genes Performance Optimization
-- =====================================================
-- IMPORTANT: This file is deprecated. Use supabase_complete_schema.sql instead!
--
-- This file only creates the deg_genes table, but it requires
-- the base tables (projects, datasets) to exist first.
--
-- For a fresh Supabase instance, use: supabase_complete_schema.sql
-- =====================================================
--
-- DEPRECATED - USE supabase_complete_schema.sql INSTEAD
--
-- =====================================================

-- Step 1: Create the deg_genes table
-- This table stores individual DEG genes for each comparison
CREATE TABLE IF NOT EXISTS deg_genes (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
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

-- Step 2: Create indexes for fast querying
-- These indexes dramatically improve query performance when filtering DEG genes

-- Index for finding all genes in a specific dataset
CREATE INDEX IF NOT EXISTS idx_deg_genes_dataset
    ON deg_genes(dataset_id);

-- Index for finding genes in a specific comparison
CREATE INDEX IF NOT EXISTS idx_deg_genes_comparison
    ON deg_genes(comparison_name);

-- Composite index for dataset + comparison (most common query pattern)
CREATE INDEX IF NOT EXISTS idx_deg_genes_dataset_comparison
    ON deg_genes(dataset_id, comparison_name);

-- Index for filtering by padj (adjusted p-value)
CREATE INDEX IF NOT EXISTS idx_deg_genes_padj
    ON deg_genes(padj);

-- Index for filtering by regulation (UP/DOWN)
CREATE INDEX IF NOT EXISTS idx_deg_genes_regulation
    ON deg_genes(regulation);

-- Index for searching by gene_id
CREATE INDEX IF NOT EXISTS idx_deg_genes_gene_id
    ON deg_genes(gene_id);

-- Composite index for common filtering pattern: dataset + comparison + padj
CREATE INDEX IF NOT EXISTS idx_deg_genes_dataset_comparison_padj
    ON deg_genes(dataset_id, comparison_name, padj);

-- Step 3: Enable Row Level Security (RLS)
ALTER TABLE deg_genes ENABLE ROW LEVEL SECURITY;

-- Step 4: Create RLS policy to ensure users can only access their own data
-- Users can view DEG genes from datasets that belong to projects they own
CREATE POLICY "Users can view DEG genes from their projects"
    ON deg_genes FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM datasets d
            JOIN projects p ON d.project_id = p.id
            WHERE d.id = deg_genes.dataset_id
            AND p.owner_id = auth.uid()
        )
    );

-- Step 5: Create policy for inserting DEG genes (for backend processing)
-- Only authenticated users can insert DEG genes for their own datasets
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

-- Step 6: Create policy for deleting DEG genes (for reprocessing)
-- Users can delete DEG genes from their own datasets
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

-- =====================================================
-- Verification Queries (Optional)
-- =====================================================
-- Uncomment these to verify the table was created correctly

-- Show table structure
-- SELECT column_name, data_type, is_nullable
-- FROM information_schema.columns
-- WHERE table_name = 'deg_genes'
-- ORDER BY ordinal_position;

-- Show all indexes
-- SELECT indexname, indexdef
-- FROM pg_indexes
-- WHERE tablename = 'deg_genes';

-- Show RLS policies
-- SELECT policyname, cmd, qual, with_check
-- FROM pg_policies
-- WHERE tablename = 'deg_genes';

-- =====================================================
-- Performance Notes
-- =====================================================
-- After running this script:
--
-- 1. Existing datasets will NOT have deg_genes populated
--    - Only NEW datasets uploaded after this change will populate deg_genes
--    - To populate existing datasets, use the reprocess endpoint:
--      POST /datasets/{dataset_id}/reprocess
--
-- 2. Expected performance improvements:
--    - Before: Loading 50MB+ Parquet file into memory (2-5 seconds)
--    - After: Direct database query with indexes (<100ms)
--    - 20-50x faster page loads for comparison details
--
-- 3. Database storage:
--    - Each DEG gene: ~200 bytes
--    - Dataset with 10,000 DEGs: ~2MB
--    - 100 datasets: ~200MB additional storage
--
-- =====================================================
