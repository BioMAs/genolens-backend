-- Désactive Row Level Security pour le développement local
-- En production avec Supabase, RLS reste activé

SET client_min_messages = 'NOTICE';

DO $$
BEGIN
    RAISE NOTICE '🔓 Disabling Row Level Security for local development...';
END $$;

-- Désactiver RLS sur toutes les tables
ALTER TABLE IF EXISTS projects DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS datasets DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS deg_genes DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS enrichment_pathways DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS project_members DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS gene_sets DISABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    RAISE NOTICE '✅ Row Level Security disabled on all tables';
    RAISE NOTICE '';
    RAISE NOTICE '⚠️  WARNING: RLS is disabled for development only!';
    RAISE NOTICE '   In production (Supabase), RLS remains enabled for security.';
    RAISE NOTICE '';
    RAISE NOTICE 'Tables affected:';
    RAISE NOTICE '  - projects';
    RAISE NOTICE '  - datasets';
    RAISE NOTICE '  - deg_genes';
    RAISE NOTICE '  - enrichment_pathways';
    RAISE NOTICE '  - project_members';
    RAISE NOTICE '  - gene_sets';
END $$;
