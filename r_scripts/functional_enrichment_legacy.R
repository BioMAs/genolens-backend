#!/usr/bin/env Rscript
# ==============================================================================
# GenoLens Functional Enrichment Analysis (anno.db approach)
# Adapted from pipe_scilicium/workflow/scripts/r/functional_enrichment_legacy.R
#
# Uses custom hypergeometric enrichment against anno.db annotation matrices.
# Supports all species covered by anno.db files (human, mouse, rat, zebrafish, pig).
#
# Usage:
#   Rscript functional_enrichment_legacy.R \
#     --deg-results   <path/to/results_filtered.tsv> \
#     --anno-db-dir   <path/to/anno.db/> \
#     --species       human \
#     --output-prefix <outdir/enrichment> \
#     --pvalue-cutoff 0.05 \
#     --r-cutoff      3 \
#     [--databases    GO_BP,GO_MF,KEGG]
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(Matrix)
  library(data.table)
  library(optparse)
  library(jsonlite)
})

# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------

option_list <- list(
  make_option("--deg-results",    type = "character", help = "DEG results TSV (filtered)"),
  make_option("--anno-db-dir",    type = "character", help = "Directory containing anno.db RDS files"),
  make_option("--species",        type = "character", default = "human",
              help = "Species key matching anno.db filename (human, mouse, rat, zebrafish, pig) [default: human]"),
  make_option("--output-prefix",  type = "character", help = "Output file prefix (directory will be created)"),
  make_option("--pvalue-cutoff",  type = "double",    default = 0.05,
              help = "P-value cutoff for enriched terms [default: 0.05]"),
  make_option("--r-cutoff",       type = "integer",   default = 3,
              help = "Minimum number of genes per enriched term [default: 3]"),
  make_option("--databases",      type = "character",  default = "all",
              help = "Comma-separated list of databases to use, or 'all' [default: all]")
)

opt <- parse_args(OptionParser(option_list = option_list))

for (arg_name in c("deg-results", "anno-db-dir", "output-prefix")) {
  if (is.null(opt[[arg_name]])) {
    stop("Missing required argument: --", arg_name)
  }
}

results_file   <- opt[["deg-results"]]
anno_db_dir    <- opt[["anno-db-dir"]]
species        <- tolower(opt$species)
output_prefix  <- opt[["output-prefix"]]
p_cutoff       <- opt[["pvalue-cutoff"]]
r_cutoff       <- opt[["r-cutoff"]]
databases_arg  <- opt$databases

# Parse database filter
selected_databases <- if (is.null(databases_arg) || nchar(databases_arg) == 0 || tolower(databases_arg) == "all") {
  NULL  # NULL = use all categories
} else {
  strsplit(databases_arg, ",")[[1]]
}

message("================================================================")
message("GenoLens Functional Enrichment Analysis (anno.db approach)")
message("================================================================")
message("Results file:   ", results_file)
message("Anno DB dir:    ", anno_db_dir)
message("Species:        ", species)
message("Output prefix:  ", output_prefix)
message("P-value cutoff: ", p_cutoff)
message("R cutoff:       ", r_cutoff)
if (!is.null(selected_databases)) {
  message("Databases:      ", paste(selected_databases, collapse = ", "))
} else {
  message("Databases:      all available")
}

# Create output directory
output_dir <- dirname(output_prefix)
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# [1/6] Core enrichment function (hypergeometric test)
# ==============================================================================
message("\n[1/6] Loading enrichment functions...")

fa.function <- function(group.ids, anno.db.m, all.ids = NULL, group.name = "gene.cluster",
                        category = "category", p.cutoff = 0.001, r.cutoff = 3) {

  if (ncol(anno.db.m) <= 0) { return(NULL) }

  # Universe of annotated genes
  N.IDs <- unique(names(which(Matrix::rowSums(anno.db.m) > 0)))
  if (!is.null(all.ids)) {
    N.IDs <- intersect(N.IDs, all.ids)
  }

  R.IDs <- intersect(unique(group.ids), N.IDs)

  if (length(R.IDs) <= 0) {
    r.IDs <- lapply(colnames(anno.db.m), function(x) return(NULL))
    names(r.IDs) <- colnames(anno.db.m)
  } else {
    r.IDs <- apply(anno.db.m[R.IDs, , drop = FALSE], 2, function(x) unique(R.IDs[which(x > 0)]))
  }

  annour.df <- data.frame(term = colnames(anno.db.m))
  annour.df$category    <- category
  annour.df$gene.cluster <- group.name
  annour.df$genes       <- unlist(lapply(r.IDs, paste, collapse = "|"))
  annour.df$r           <- unlist(lapply(r.IDs, length))
  annour.df$R           <- length(R.IDs)
  annour.df$n           <- Matrix::colSums(anno.db.m[N.IDs, , drop = FALSE], na.rm = TRUE)
  annour.df$N           <- length(N.IDs)

  annour.df$rR       <- round(annour.df$r / annour.df$R, digit = 3) * 100
  annour.df$nN       <- round(annour.df$n / annour.df$N, digit = 3) * 100
  annour.df$rExpected <- round(annour.df$n * annour.df$R / annour.df$N, digit = 0)

  sd <- (annour.df$rExpected * (1 - annour.df$R / annour.df$N) *
         (1 - (annour.df$n - 1) / (annour.df$N - 1)))^0.5
  annour.df$zscore <- (annour.df$r - annour.df$rExpected) / sd

  annour.df$p.norm.enri  <- pnorm(annour.df$zscore, lower.tail = FALSE)
  annour.df$p.norm.depl  <- pnorm(annour.df$zscore, lower.tail = TRUE)
  annour.df$p.hyper.enri <- phyper(annour.df$r - 1, annour.df$n, annour.df$N - annour.df$n,
                                   annour.df$R, lower.tail = FALSE)
  annour.df$p.hyper.depl <- phyper(annour.df$r, annour.df$n, annour.df$N - annour.df$n,
                                   annour.df$R, lower.tail = TRUE)

  annour.df$info   <- NA
  annour.df$pvalue <- NA

  enri.idx <- which(annour.df$p.hyper.enri < p.cutoff & annour.df$r >= r.cutoff)
  depl.idx <- which(annour.df$p.hyper.depl < p.cutoff & annour.df$r >= r.cutoff)

  if (length(enri.idx) > 0) {
    annour.df$info[enri.idx]   <- "enriched"
    annour.df$pvalue[enri.idx] <- annour.df$p.hyper.enri[enri.idx]
  }
  if (length(depl.idx) > 0) {
    annour.df$info[depl.idx]   <- "depleted"
    annour.df$pvalue[depl.idx] <- annour.df$p.hyper.depl[depl.idx]
  }

  annour.df$p.adjust.enri <- if (length(enri.idx) > 0) {
    p.adjust(annour.df$p.hyper.enri, method = "BH")
  } else {
    rep(1, nrow(annour.df))
  }
  annour.df$p.adjust.depl <- if (length(depl.idx) > 0) {
    p.adjust(annour.df$p.hyper.depl, method = "BH")
  } else {
    rep(1, nrow(annour.df))
  }

  annour.df <- annour.df[order(annour.df$pvalue, na.last = TRUE), ]
  return(annour.df)
}

# ==============================================================================
# [2/6] Load anno.db
# ==============================================================================
message("\n[2/6] Loading annotation database for species: ", species)

# Map species alias → canonical filename key
species_mapping <- list(
  "hg38" = "human", "hg19" = "human", "human" = "human", "homo_sapiens" = "human",
  "mm10" = "mouse", "mm39" = "mouse", "mouse" = "mouse", "mus_musculus" = "mouse",
  "rn6"  = "rat",   "rn7"  = "rat",   "rat"   = "rat",   "rattus_norvegicus" = "rat",
  "drerio" = "zebrafish", "zebrafish" = "zebrafish", "danio_rerio" = "zebrafish",
  "pig"    = "pig",       "sus_scrofa" = "pig"
)

db_species <- species_mapping[[species]]
if (is.null(db_species)) {
  message("WARNING: Unsupported species '", species, "'. Skipping enrichment.")
  # Write empty output files so the pipeline doesn't fail
  saveRDS(list(), file = paste0(output_prefix, "_all.rds"))
  saveRDS(list(), file = paste0(output_prefix, "_onlyenriched.rds"))
  write.table(data.frame(), file = paste0(output_prefix, "_onlyenriched.txt"),
              quote = FALSE, sep = "\t", col.names = TRUE, row.names = FALSE)
  quit(save = "no", status = 0)
}

# Find most recent anno.db file for this species
anno_db_pattern <- sprintf("anno\\.db\\.%s\\.gene_symbol.*\\.rds$", db_species)
anno_db_files <- list.files(anno_db_dir, pattern = anno_db_pattern, full.names = TRUE)

if (length(anno_db_files) == 0) {
  message("WARNING: No anno.db file found for species '", db_species, "' in: ", anno_db_dir)
  message("Expected pattern: ", anno_db_pattern)
  message("Skipping enrichment.")
  saveRDS(list(), file = paste0(output_prefix, "_all.rds"))
  saveRDS(list(), file = paste0(output_prefix, "_onlyenriched.rds"))
  write.table(data.frame(), file = paste0(output_prefix, "_onlyenriched.txt"),
              quote = FALSE, sep = "\t", col.names = TRUE, row.names = FALSE)
  quit(save = "no", status = 0)
}

anno_db_files <- sort(anno_db_files, decreasing = TRUE)
anno_db_file  <- anno_db_files[1]
message("  Loading: ", basename(anno_db_file))

anno.db.obj <- readRDS(anno_db_file)
message("  Available categories: ", length(names(anno.db.obj$annotations)))

# ==============================================================================
# [3/6] Read DEG results
# ==============================================================================
message("\n[3/6] Reading DEG results...")

deg_results <- read_tsv(results_file, show_col_types = FALSE)

if (nrow(deg_results) == 0) {
  message("WARNING: No DEGs found. Creating empty output files.")
  saveRDS(list(), file = paste0(output_prefix, "_all.rds"))
  saveRDS(list(), file = paste0(output_prefix, "_onlyenriched.rds"))
  write.table(data.frame(), file = paste0(output_prefix, "_onlyenriched.txt"),
              quote = FALSE, sep = "\t", col.names = TRUE, row.names = FALSE)
  quit(save = "no", status = 0)
}

message("  Found ", nrow(deg_results), " DEGs")

# ==============================================================================
# [4/6] Prepare gene lists
# ==============================================================================
message("\n[4/6] Preparing gene lists...")

# anno.db uses gene symbols — use gene_name column if available, else gene_id
if ("gene_name" %in% colnames(deg_results)) {
  deg_results_named <- deg_results %>% filter(!is.na(gene_name), gene_name != "")
} else if ("gene_id" %in% colnames(deg_results)) {
  deg_results_named <- deg_results %>% rename(gene_name = gene_id)
} else {
  stop("DEG results must have a 'gene_name' or 'gene_id' column")
}

# Detect log2FoldChange column (could be logFC:ComparisonID format)
lfc_col <- if ("log2FoldChange" %in% colnames(deg_results_named)) {
  "log2FoldChange"
} else {
  # Look for logFC:* columns
  lfc_candidates <- grep("^logFC:", colnames(deg_results_named), value = TRUE)
  if (length(lfc_candidates) > 0) lfc_candidates[1] else NULL
}

if (!is.null(lfc_col)) {
  all_genes  <- deg_results_named$gene_name
  up_genes   <- deg_results_named %>% filter(.data[[lfc_col]] > 0) %>% pull(gene_name)
  down_genes <- deg_results_named %>% filter(.data[[lfc_col]] < 0) %>% pull(gene_name)
} else {
  all_genes  <- deg_results_named$gene_name
  up_genes   <- character(0)
  down_genes <- character(0)
}

message("  All DEGs: ",      length(all_genes))
message("  Up-regulated: ",  length(up_genes))
message("  Down-regulated: ", length(down_genes))

list.groups <- list(
  "all_DEGs"       = all_genes,
  "up_regulated"   = up_genes,
  "down_regulated" = down_genes
)

# Background universe from anno.db
all_ids <- unique(unlist(lapply(anno.db.obj$annotations, function(x) {
  if (is.matrix(x) || inherits(x, "Matrix")) rownames(x)
  else if (is.data.frame(x)) unique(x[, 1])
  else NULL
})))
message("  Background universe: ", length(all_ids), " genes")

# ==============================================================================
# [5/6] Run enrichment analysis
# ==============================================================================
message("\n[5/6] Running enrichment analysis...")

# Determine which categories to run
all_categories <- names(anno.db.obj$annotations)

# Exclude low-level categories
ncbi_idx <- grep("^NCBI_|^chromosome_location$|^xrefs$|^interactions$|^regulations$", all_categories)
if (length(ncbi_idx) > 0) {
  all_categories <- all_categories[-ncbi_idx]
}

# Apply user-requested filter
if (!is.null(selected_databases)) {
  categories <- intersect(all_categories, selected_databases)
  missing_dbs <- setdiff(selected_databases, all_categories)
  if (length(missing_dbs) > 0) {
    message("  WARNING: requested databases not found in anno.db: ", paste(missing_dbs, collapse = ", "))
  }
} else {
  categories <- all_categories
}

message("  Analyzing ", length(categories), " categories")

all.annour.obj      <- list()
enriched.annour.obj <- list()
enriched.annour.df  <- list()

for (group.name in names(list.groups)) {
  if (length(list.groups[[group.name]]) == 0) {
    message("\n  Skipping empty group: ", group.name)
    enriched.annour.df[[group.name]] <- data.frame()
    next
  }

  message("\n  Processing group: ", group.name, " (", length(list.groups[[group.name]]), " genes)")
  all.annour.obj[[group.name]]      <- list()
  enriched.annour.obj[[group.name]] <- list()
  group.ids <- list.groups[[group.name]]

  for (category in categories) {
    anno_m <- anno.db.obj$annotations[[category]]
    if (is.null(anno_m) || ncol(anno_m) <= 0) { next }

    annour.df <- tryCatch(
      fa.function(group.ids, anno_m, all_ids, group.name, category, p_cutoff, r_cutoff),
      error = function(e) {
        message("    WARNING: enrichment failed for '", category, "': ", e$message)
        NULL
      }
    )
    if (is.null(annour.df)) { next }

    all.annour.obj[[group.name]][[category]]      <- annour.df
    enriched.annour.obj[[group.name]][[category]] <- annour.df[!is.na(annour.df$info), , drop = FALSE]

    n_enriched <- sum(!is.na(annour.df$info))
    if (n_enriched > 0) {
      message("    ", category, ": ", n_enriched, " enriched terms")
    }
  }

  non_empty <- which(sapply(enriched.annour.obj[[group.name]], nrow) > 0)
  if (length(non_empty) > 0) {
    enriched.annour.df[[group.name]] <- as.data.frame(
      data.table::rbindlist(enriched.annour.obj[[group.name]][non_empty])
    )
  } else {
    enriched.annour.df[[group.name]] <- data.frame()
  }
}

# Combine all enriched terms across groups
non_empty_groups <- which(sapply(enriched.annour.df, nrow) > 0)
if (length(non_empty_groups) > 0) {
  enriched.annour.df.combined <- as.data.frame(
    data.table::rbindlist(enriched.annour.df[non_empty_groups], fill = TRUE)
  )
  # Truncate gene lists for compatibility
  enriched.annour.df.combined$genes <- unlist(
    lapply(enriched.annour.df.combined$genes, substr, 1, 32000)
  )
} else {
  enriched.annour.df.combined <- data.frame()
}

# ==============================================================================
# [6/6] Save results
# ==============================================================================
message("\n[6/6] Saving results...")

saveRDS(all.annour.obj, file = paste0(output_prefix, "_all.rds"))
message("  Saved: ", basename(paste0(output_prefix, "_all.rds")))

saveRDS(enriched.annour.obj, file = paste0(output_prefix, "_onlyenriched.rds"))
message("  Saved: ", basename(paste0(output_prefix, "_onlyenriched.rds")))

write.table(enriched.annour.df.combined,
            file   = paste0(output_prefix, "_onlyenriched.txt"),
            quote  = FALSE, sep = "\t", col.names = TRUE, row.names = FALSE)
message("  Saved: ", basename(paste0(output_prefix, "_onlyenriched.txt")))

# Summary JSON for pipeline manifest
n_total_enriched <- nrow(enriched.annour.df.combined)
enrichment_summary <- list(
  n_total_enriched_terms = n_total_enriched,
  species = species,
  databases_used = categories,
  groups_analyzed = names(list.groups)
)
write(toJSON(enrichment_summary, auto_unbox = TRUE, pretty = TRUE),
      file.path(output_dir, "enrichment_summary.json"))

# Print final summary
message("\n================================================================")
message("ENRICHMENT SUMMARY — ", species)
message("================================================================")
for (group.name in names(list.groups)) {
  n <- if (!is.null(enriched.annour.obj[[group.name]])) {
    sum(sapply(enriched.annour.obj[[group.name]], nrow))
  } else { 0 }
  message(group.name, ": ", n, " enriched terms")
}
message("================================================================")
message("Functional enrichment analysis completed successfully!")
message("================================================================")
