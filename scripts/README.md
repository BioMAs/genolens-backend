# GenoLens Management Scripts

This directory contains utility scripts for managing GenoLens v2 data and configuration.

## Gene Set Loading

### `load_gene_sets.py`

Loads gene sets from GMT files into the database for GSEA analysis.

**Usage:**

```bash
# Load MSigDB Hallmark gene sets
python scripts/load_gene_sets.py \
  --file data/h.all.v2024.1.Hs.symbols.gmt \
  --database HALLMARK \
  --version 2024.1

# Load GO Biological Process (replace existing)
python scripts/load_gene_sets.py \
  --file data/c5.go.bp.v2024.1.Hs.symbols.gmt \
  --database GO_BP \
  --version 2024.1 \
  --clear

# Load KEGG pathways
python scripts/load_gene_sets.py \
  --file data/c2.cp.kegg.v2024.1.Hs.symbols.gmt \
  --database KEGG \
  --version 2024.1

# Search for gene sets
python scripts/load_gene_sets.py --search "TNFA" --database HALLMARK

# Show statistics
python scripts/load_gene_sets.py --stats
```

**Supported Databases:**

- `GO_BP` - Gene Ontology Biological Process
- `GO_MF` - Gene Ontology Molecular Function
- `GO_CC` - Gene Ontology Cellular Component
- `KEGG` - KEGG Pathways
- `REACTOME` - Reactome Pathways
- `HALLMARK` - MSigDB Hallmark gene sets
- `C2_CURATED` - MSigDB C2 curated gene sets
- `C5_ONTOLOGY` - MSigDB C5 ontology gene sets
- `C6_ONCOGENIC` - MSigDB C6 oncogenic signatures
- `C7_IMMUNOLOGIC` - MSigDB C7 immunologic signatures
- `CUSTOM` - User-defined custom gene sets

## Downloading Gene Sets

### MSigDB (Molecular Signatures Database)

1. Register at https://www.gsea-msigdb.org/gsea/register.jsp
2. Download gene sets from https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp
3. Download the **"Human Gene Symbols"** (.gmt) versions

**Recommended downloads:**

- **Hallmark gene sets** (h.all.v2024.1.Hs.symbols.gmt) - ~50 well-defined biological states
- **C2: Curated gene sets** (c2.all.v2024.1.Hs.symbols.gmt) - KEGG, Reactome, BioCarta, etc.
- **C5: Ontology gene sets** (c5.all.v2024.1.Hs.symbols.gmt) - GO terms
- **C6: Oncogenic signatures** (c6.all.v2024.1.Hs.symbols.gmt) - Cancer gene signatures

### Gene Ontology

GO gene sets are included in MSigDB C5 collection. Alternatively:

1. Download from http://geneontology.org/
2. Use tools like GOAtools to convert to GMT format

### KEGG Pathways

KEGG gene sets are included in MSigDB C2 collection.

## Database Setup

Before loading gene sets, ensure your database is migrated:

```bash
# Run migrations
cd backend
alembic upgrade head
```

## Example Workflow

```bash
# 1. Download MSigDB gene sets (after registration)
mkdir -p backend/data/genesets
cd backend/data/genesets

# 2. Download from MSigDB website or use wget (if you have an account)
# Place .gmt files in this directory

# 3. Load Hallmark gene sets
cd ../..
python scripts/load_gene_sets.py \
  --file data/genesets/h.all.v2024.1.Hs.symbols.gmt \
  --database HALLMARK \
  --version 2024.1

# 4. Load GO Biological Process
python scripts/load_gene_sets.py \
  --file data/genesets/c5.go.bp.v2024.1.Hs.symbols.gmt \
  --database GO_BP \
  --version 2024.1

# 5. Load KEGG
python scripts/load_gene_sets.py \
  --file data/genesets/c2.cp.kegg.v2024.1.Hs.symbols.gmt \
  --database KEGG \
  --version 2024.1

# 6. Verify loading
python scripts/load_gene_sets.py --stats
```

## GMT File Format

GMT (Gene Matrix Transposed) format:
```
<gene_set_name> <tab> <description> <tab> <gene1> <tab> <gene2> <tab> ...
```

Example:
```
HALLMARK_TNFA_SIGNALING_VIA_NFKB	http://www.gsea-msigdb.org/...	ABCA1	ABI1	ACKR3	...
GO_CELL_CYCLE	Cell cycle	CDK1	CCNB1	CCNA2	CDC20	...
```

## Troubleshooting

**Error: File not found**
- Check that the GMT file path is correct
- Use absolute paths or paths relative to backend directory

**Error: Invalid database**
- Use `--database` with one of the supported database names (see list above)

**Error: Database connection failed**
- Ensure your `.env` file has correct `DATABASE_URL`
- Check that PostgreSQL is running
- Verify database credentials

**Gene sets not appearing in GSEA**
- Run `python scripts/load_gene_sets.py --stats` to verify gene sets are loaded
- Check that `min_size` and `max_size` parameters in GSEA match your gene set sizes
- Verify organism matches (default: "Homo sapiens")

## Updating Gene Sets

To update gene sets to a new version:

```bash
# Use --clear flag to replace existing gene sets
python scripts/load_gene_sets.py \
  --file data/h.all.v2024.2.Hs.symbols.gmt \
  --database HALLMARK \
  --version 2024.2 \
  --clear
```

## Future Scripts

Planned management scripts:

- `migrate_data.py` - Migrate data from legacy GenoLens v1
- `cleanup_storage.py` - Clean up orphaned files in Supabase Storage
- `export_project.py` - Export project data for backup
- `import_project.py` - Import project from backup
