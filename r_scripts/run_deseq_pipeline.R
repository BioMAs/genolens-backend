#!/usr/bin/env Rscript
# =============================================================================
# GenoLens Self-Service DESeq2 Analysis Pipeline
# =============================================================================
#
# Runs a complete DESeq2 differential expression analysis on user-provided data
# and outputs Genolens-compatible CSV files for each comparison.
#
# Usage:
#   Rscript run_deseq_pipeline.R \
#     --counts   counts.tsv        \
#     --samples  samples.tsv       \
#     --comparisons comparisons.tsv \
#     --outdir   output/            \
#     [options]
#
# Required input format:
#   counts.tsv  : TSV with first column 'gene_id', remaining columns = sample IDs
#                 Optional 'gene_name' column will be used for annotation.
#   samples.tsv : TSV with 'sample_id' and 'condition' columns.
#                 Optional: 'batch', 'sex', other covariates.
#   comparisons.tsv : TSV with 'comparison_id', 'condition1', 'condition2'.
#                     Optional 'perform_analysis' column (TRUE/FALSE).
#
# Output per comparison (in --outdir/comparisons/{comparison_id}/):
#   genolens_deg.csv   : Genolens-compatible DEG CSV
#   results_full.tsv   : Full DESeq2 results
#   results_filtered.tsv : Significant DEGs only
#   summary.txt        : Human-readable summary
# =============================================================================

suppressPackageStartupMessages({
  library(DESeq2)
  library(tidyverse)
  library(optparse)
  library(BiocParallel)
  library(jsonlite)
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

write_progress <- function(progress_file, msg) {
  ts <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
  line <- paste0(ts, " | ", msg)
  message(line)
  if (!is.null(progress_file) && nchar(progress_file) > 0) {
    cat(line, "\n", file = progress_file, append = TRUE)
  }
}

# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------

option_list <- list(
  make_option("--counts",        type = "character", help = "Count matrix TSV"),
  make_option("--samples",       type = "character", help = "Sample metadata TSV"),
  make_option("--comparisons",   type = "character", help = "Comparisons TSV"),
  make_option("--outdir",        type = "character", help = "Output directory"),
  make_option("--design",        type = "character", default = "auto",
              help = "DESeq2 design: auto | condition | batch_condition [default: auto]"),
  make_option("--min-reads",     type = "integer",   default = 100000,
              help = "Min total reads per sample [default: 100000]"),
  make_option("--min-genes",     type = "integer",   default = 500,
              help = "Min genes detected per sample [default: 500]"),
  make_option("--min-count",     type = "integer",   default = 10,
              help = "Min count for gene filtering [default: 10]"),
  make_option("--min-reps",      type = "integer",   default = 2,
              help = "Min replicates with min-count [default: 2]"),
  make_option("--fdr",           type = "double",    default = 0.05,
              help = "FDR threshold [default: 0.05]"),
  make_option("--min-log2fc",    type = "double",    default = 1.0,
              help = "Min |log2FoldChange| for filtered results [default: 1.0]"),
  make_option("--threads",       type = "integer",   default = 1,
              help = "Threads for BiocParallel [default: 1]"),
  make_option("--progress-file", type = "character", default = "",
              help = "Path to progress log file (optional)"),
  make_option("--species",       type = "character", default = "human",
              help = "Organism species for functional enrichment (human, mouse, rat, zebrafish, pig) [default: human]"),
  make_option("--anno-db-dir",   type = "character", default = "/app/anno_db",
              help = "Directory containing anno.db RDS files [default: /app/anno_db]"),
  make_option("--enrichment-databases", type = "character", default = "all",
              help = "Comma-separated annotation databases to use, or 'all' [default: all]")
)

opt <- parse_args(OptionParser(option_list = option_list))

# Validate required arguments
for (arg_name in c("counts", "samples", "comparisons", "outdir")) {
  if (is.null(opt[[arg_name]])) {
    stop("Missing required argument: --", arg_name)
  }
}

progress_file <- if (nchar(opt[["progress-file"]]) > 0) opt[["progress-file"]] else NULL

write_progress(progress_file, "=== GenoLens DESeq2 Pipeline Starting ===")

# Configure BiocParallel
if (opt$threads > 1) {
  register(MulticoreParam(workers = opt$threads))
} else {
  register(SerialParam())
}

# ---------------------------------------------------------------------------
# Create output directory structure
# ---------------------------------------------------------------------------

dir.create(opt$outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(opt$outdir, "comparisons"), recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# Step 0: Load and validate inputs
# ---------------------------------------------------------------------------

write_progress(progress_file, "[0/4] Loading input files...")

# Count matrix
counts_raw <- tryCatch(
  read_tsv(opt$counts, col_types = cols(), show_col_types = FALSE),
  error = function(e) stop("Cannot read counts file: ", e$message)
)

if (!"gene_id" %in% colnames(counts_raw)) {
  stop("Count matrix must have a 'gene_id' column as the first column.")
}

# Extract gene_name mapping if present (optional)
gene_names_df <- NULL
if ("gene_name" %in% colnames(counts_raw)) {
  gene_names_df <- counts_raw %>%
    select(gene_id, gene_name) %>%
    filter(!is.na(gene_id))
  counts_raw <- counts_raw %>% select(-gene_name)
}

# Convert to matrix with gene_id as rownames
counts_matrix <- counts_raw %>%
  column_to_rownames("gene_id") %>%
  as.data.frame()

message("  - Loaded count matrix: ", nrow(counts_matrix), " genes x ", ncol(counts_matrix), " samples")

# Sample metadata
samples_df <- tryCatch(
  read_tsv(opt$samples, col_types = cols(), show_col_types = FALSE),
  error = function(e) stop("Cannot read samples file: ", e$message)
)
for (col in c("sample_id", "condition")) {
  if (!col %in% colnames(samples_df)) {
    stop("Samples file is missing required column: '", col, "'")
  }
}
message("  - Loaded ", nrow(samples_df), " sample records")

# Comparisons
comparisons_df <- tryCatch(
  read_tsv(opt$comparisons, col_types = cols(), show_col_types = FALSE),
  error = function(e) stop("Cannot read comparisons file: ", e$message)
)
for (col in c("comparison_id", "condition1", "condition2")) {
  if (!col %in% colnames(comparisons_df)) {
    stop("Comparisons file is missing required column: '", col, "'")
  }
}

# Filter to active comparisons
if ("perform_analysis" %in% colnames(comparisons_df)) {
  comparisons_df <- comparisons_df %>%
    filter(
      perform_analysis == TRUE |
      as.character(perform_analysis) %in% c("True", "true", "TRUE", "1", "yes")
    )
}
if (nrow(comparisons_df) == 0) {
  stop("No comparisons to run (all have perform_analysis=FALSE).")
}
message("  - ", nrow(comparisons_df), " comparison(s) to run: ",
        paste(comparisons_df$comparison_id, collapse = ", "))

# ---------------------------------------------------------------------------
# Step 1: Sample QC and gene filtering
# ---------------------------------------------------------------------------

write_progress(progress_file, "[1/4] Filtering samples and genes...")

# Align samples between matrix and metadata
common_samples <- intersect(colnames(counts_matrix), samples_df$sample_id)
if (length(common_samples) == 0) {
  stop(
    "No matching samples between count matrix columns and samples file.\n",
    "  Matrix columns (first 5): ", paste(head(colnames(counts_matrix), 5), collapse = ", "), "\n",
    "  Samples file (first 5): ", paste(head(samples_df$sample_id, 5), collapse = ", ")
  )
}
message("  - Matched ", length(common_samples), " / ", ncol(counts_matrix), " samples")

counts_matrix <- counts_matrix[, common_samples, drop = FALSE]
samples_df <- samples_df %>% filter(sample_id %in% common_samples)

# Sample-level QC metrics
sample_qc <- data.frame(
  sample_id      = colnames(counts_matrix),
  total_reads    = as.integer(colSums(counts_matrix)),
  genes_detected = as.integer(colSums(counts_matrix > 0)),
  stringsAsFactors = FALSE
)

samples_pass <- sample_qc %>%
  filter(total_reads >= opt[["min-reads"]], genes_detected >= opt[["min-genes"]]) %>%
  pull(sample_id)

samples_fail <- setdiff(common_samples, samples_pass)
if (length(samples_fail) > 0) {
  message("  - Removed ", length(samples_fail), " low-quality sample(s): ",
          paste(samples_fail, collapse = ", "))
}
if (length(samples_pass) < 2) {
  stop("Fewer than 2 samples passed QC. Cannot run DESeq2.")
}

counts_matrix <- counts_matrix[, samples_pass, drop = FALSE]
samples_df    <- samples_df %>% filter(sample_id %in% samples_pass)

# Gene-level filtering: keep genes with >= min-count in >= min-reps samples
n_before <- nrow(counts_matrix)
gene_pass <- apply(counts_matrix, 1, function(x) sum(x >= opt[["min-count"]]) >= opt[["min-reps"]])
counts_matrix <- counts_matrix[gene_pass, , drop = FALSE]
message("  - Genes passing filter: ", nrow(counts_matrix),
        " (removed ", n_before - nrow(counts_matrix), ")")

if (nrow(counts_matrix) < 100) {
  stop("Too few genes (", nrow(counts_matrix), ") after filtering. Check thresholds.")
}

# Ensure integer storage for DESeq2
# as.matrix() must precede storage.mode() — data.frames are 'list' objects
# and cannot be coerced to integer via storage.mode() directly.
counts_matrix <- as.matrix(round(counts_matrix))
storage.mode(counts_matrix) <- "integer"

# Write filtering report
writeLines(c(
  paste("Samples before QC:", length(common_samples)),
  paste("Samples passing QC:", length(samples_pass)),
  paste("Samples removed:", length(samples_fail)),
  paste("Genes after filtering:", nrow(counts_matrix)),
  paste("Total counts after filtering:", sum(colSums(counts_matrix)))
), file.path(opt$outdir, "filtering_report.txt"))

# ---------------------------------------------------------------------------
# Step 2: DESeq2 normalization
# ---------------------------------------------------------------------------

write_progress(progress_file, "[2/4] Normalizing with DESeq2...")

# Build colData aligned to count matrix column order
col_data <- samples_df %>%
  filter(sample_id %in% colnames(counts_matrix)) %>%
  arrange(match(sample_id, colnames(counts_matrix))) %>%
  as.data.frame()
rownames(col_data) <- col_data$sample_id

col_data$condition <- factor(col_data$condition)

# Determine design formula
has_batch <- "batch" %in% colnames(col_data) && length(unique(col_data$batch)) > 1

design_str <- opt$design
if (design_str == "auto") {
  design_str <- if (has_batch) "~ batch + condition" else "~ condition"
} else if (design_str == "batch_condition") {
  design_str <- "~ batch + condition"
} else {
  design_str <- "~ condition"
}

if (has_batch && grepl("batch", design_str)) {
  col_data$batch <- factor(col_data$batch)
}

message("  - Design formula: ", design_str)

design_formula <- as.formula(design_str)

# Create DESeqDataSet and run full pipeline
dds <- DESeqDataSetFromMatrix(
  countData = as.matrix(counts_matrix),
  colData   = col_data,
  design    = design_formula
)

dds <- DESeq(dds, parallel = (opt$threads > 1), quiet = FALSE)

message("  - DESeq2 normalization complete")

# Export normalized counts
norm_counts_df <- as.data.frame(counts(dds, normalized = TRUE)) %>%
  rownames_to_column("gene_id")
if (!is.null(gene_names_df)) {
  norm_counts_df <- norm_counts_df %>%
    left_join(gene_names_df, by = "gene_id") %>%
    select(gene_id, gene_name, everything())
}
write_tsv(norm_counts_df, file.path(opt$outdir, "normalized_counts.tsv"))

# ---------------------------------------------------------------------------
# Step 2b: VST transformation + PCA + QC report
# ---------------------------------------------------------------------------

write_progress(progress_file, "[2b/4] Computing VST transformation and PCA...")

# VST-stabilized counts (for visualization / clustering / PCA)
vst_obj <- tryCatch(
  vst(dds, blind = FALSE),
  error = function(e) {
    message("  - VST failed, falling back to rlog: ", e$message)
    rlog(dds, blind = FALSE)
  }
)
vst_mat <- assay(vst_obj)
vst_df <- as.data.frame(vst_mat) %>% rownames_to_column("gene_id")
if (!is.null(gene_names_df)) {
  vst_df <- vst_df %>%
    left_join(gene_names_df, by = "gene_id") %>%
    select(gene_id, gene_name, everything())
}
write_tsv(vst_df, file.path(opt$outdir, "vst_counts.tsv"))
message("  - VST counts exported: ", nrow(vst_df), " genes x ", ncol(vst_df) - 1, " samples")

# PCA on top variable genes
top_n_pca <- min(500, nrow(vst_mat))
gene_vars <- apply(vst_mat, 1, var)
top_genes <- names(sort(gene_vars, decreasing = TRUE))[seq_len(top_n_pca)]
pca_res <- prcomp(t(vst_mat[top_genes, ]), scale. = FALSE)

# Variance explained
var_explained <- (pca_res$sdev^2) / sum(pca_res$sdev^2)
n_pcs <- min(4, ncol(pca_res$x))

# Build PCA data for frontend
pca_samples <- rownames(pca_res$x)
pca_coords <- as.data.frame(pca_res$x[, seq_len(n_pcs), drop = FALSE])
pca_coords$sample_id <- pca_samples

# Merge with sample metadata
if ("condition" %in% colnames(col_data)) {
  pca_coords <- pca_coords %>%
    left_join(col_data %>% select(sample_id, condition, any_of("batch")), by = "sample_id")
}

pca_json <- list(
  variance_explained = as.numeric(var_explained[seq_len(n_pcs)]),
  pc_labels = colnames(pca_res$x)[seq_len(n_pcs)],
  samples = lapply(seq_len(nrow(pca_coords)), function(i) {
    row <- pca_coords[i, ]
    out <- list(sample_id = row$sample_id)
    for (pc in colnames(pca_res$x)[seq_len(n_pcs)]) {
      out[[pc]] <- as.numeric(row[[pc]])
    }
    if ("condition" %in% colnames(row)) out$condition <- as.character(row$condition)
    if ("batch" %in% colnames(row))     out$batch     <- as.character(row$batch)
    out
  })
)
write(toJSON(pca_json, auto_unbox = TRUE, digits = 6),
      file.path(opt$outdir, "pca_data.json"))
message("  - PCA data exported (PC1-PC", n_pcs, " on top ", top_n_pca, " variable genes)")

# QC report JSON
qc_json <- list(
  total_input_samples  = length(common_samples),
  samples_passed       = length(samples_pass),
  samples_removed      = length(samples_fail),
  removed_sample_ids   = samples_fail,
  genes_before_filter  = n_before,
  genes_after_filter   = nrow(counts_matrix),
  genes_removed        = n_before - nrow(counts_matrix),
  min_reads_threshold  = opt[["min-reads"]],
  min_genes_threshold  = opt[["min-genes"]],
  min_count_threshold  = opt[["min-count"]],
  min_reps_threshold   = opt[["min-reps"]],
  design_formula       = design_str,
  has_batch_correction = has_batch
)
write(toJSON(qc_json, auto_unbox = TRUE),
      file.path(opt$outdir, "qc_report.json"))
message("  - QC report exported")

# ---------------------------------------------------------------------------
# Step 3+5: Per-comparison DE analysis and Genolens export
# ---------------------------------------------------------------------------

write_progress(progress_file, paste0(
  "[3/4] Running DE analysis for ", nrow(comparisons_df), " comparison(s)..."
))

condition_levels <- levels(col_data$condition)
fdr_threshold    <- opt$fdr
min_log2fc       <- opt[["min-log2fc"]]

for (i in seq_len(nrow(comparisons_df))) {
  comp     <- comparisons_df[i, ]
  comp_id  <- comp$comparison_id
  cond1    <- comp$condition1
  cond2    <- comp$condition2

  comp_dir <- file.path(opt$outdir, "comparisons", comp_id)
  dir.create(comp_dir, recursive = TRUE, showWarnings = FALSE)

  write_progress(progress_file,
    paste0("  [", i, "/", nrow(comparisons_df), "] ", comp_id, ": ", cond1, " vs ", cond2))

  # Validate conditions exist in samples
  if (!cond1 %in% condition_levels) {
    msg <- paste0("condition '", cond1, "' not found in samples (available: ",
                  paste(condition_levels, collapse = ", "), ")")
    write_progress(progress_file, paste0("  SKIP: ", msg))
    writeLines(paste("SKIPPED:", msg), file.path(comp_dir, "error.txt"))
    next
  }
  if (!cond2 %in% condition_levels) {
    msg <- paste0("condition '", cond2, "' not found in samples")
    write_progress(progress_file, paste0("  SKIP: ", msg))
    writeLines(paste("SKIPPED:", msg), file.path(comp_dir, "error.txt"))
    next
  }

  tryCatch({
    # Wald test for this contrast
    res <- results(
      dds,
      contrast = c("condition", cond1, cond2),
      alpha    = fdr_threshold
    )

    # LFC shrinkage with apeglm (more accurate for visualization)
    # apeglm requires a coef name, not a contrast vector — find matching coef
    coef_name <- tryCatch({
      all_coefs <- resultsNames(dds)
      # Try exact match: condition_cond1_vs_cond2
      candidate <- paste0("condition_", cond1, "_vs_", cond2)
      if (candidate %in% all_coefs) {
        candidate
      } else {
        NULL
      }
    }, error = function(e) NULL)

    lfc_shrunk_col <- if (!is.null(coef_name)) {
      tryCatch({
        shrunk <- lfcShrink(dds, coef = coef_name, type = "apeglm", quiet = TRUE)
        as.data.frame(shrunk)$log2FoldChange
      }, error = function(e) {
        message("  - apeglm shrinkage failed for ", comp_id, ": ", e$message)
        as.data.frame(res)$log2FoldChange
      })
    } else {
      as.data.frame(res)$log2FoldChange
    }

    # Format results data frame
    res_df <- as.data.frame(res) %>%
      rownames_to_column("gene_id") %>%
      mutate(logFC_shrunk = lfc_shrunk_col) %>%
      arrange(padj)

    # Add gene_name (from count matrix annotation or fall back to gene_id)
    if (!is.null(gene_names_df)) {
      res_df <- res_df %>%
        left_join(gene_names_df, by = "gene_id") %>%
        mutate(gene_name = if_else(is.na(gene_name) | gene_name == "", gene_id, gene_name))
    } else {
      res_df <- res_df %>% mutate(gene_name = gene_id)
    }

    # Save full results
    write_tsv(res_df, file.path(comp_dir, "results_full.tsv"))

    # Save filtered results (significant + fold-change threshold)
    res_filtered <- res_df %>%
      filter(!is.na(padj), padj <= fdr_threshold, abs(log2FoldChange) >= min_log2fc)
    write_tsv(res_filtered, file.path(comp_dir, "results_filtered.tsv"))

    # ----- Genolens-compatible DEG CSV -----
    # Column naming convention: logFC:ComparisonID, pvalue:ComparisonID, padj:ComparisonID
    comp_clean <- gsub("[^A-Za-z0-9_]", "_", comp_id)

    genolens_deg <- res_df %>%
      filter(!is.na(padj)) %>%
      select(
        gene_id,
        gene_name,
        baseMean,
        !!paste0("logFC:", comp_clean) := log2FoldChange,
        !!paste0("logFC_shrunk:", comp_clean) := logFC_shrunk,
        !!paste0("pvalue:", comp_clean) := pvalue,
        !!paste0("padj:", comp_clean) := padj
      )

    write_csv(genolens_deg, file.path(comp_dir, "genolens_deg.csv"))

    # Comparison summary
    n_tested  <- sum(!is.na(res_df$padj))
    n_sig     <- sum(res_df$padj < fdr_threshold, na.rm = TRUE)
    n_up      <- sum(res_df$padj < fdr_threshold & res_df$log2FoldChange > 0, na.rm = TRUE)
    n_down    <- sum(res_df$padj < fdr_threshold & res_df$log2FoldChange < 0, na.rm = TRUE)

    writeLines(c(
      paste("comparison_id:",  comp_id),
      paste("condition1:",     cond1),
      paste("condition2:",     cond2),
      paste("genes_tested:",   n_tested),
      paste("significant:",    n_sig),
      paste("up_regulated:",   n_up),
      paste("down_regulated:", n_down),
      paste("fdr_threshold:",  fdr_threshold),
      paste("min_log2fc:",     min_log2fc)
    ), file.path(comp_dir, "summary.txt"))

    write_progress(progress_file,
      paste0("    Done: ", n_sig, " significant / ", n_tested, " tested  (up:", n_up, " dn:", n_down, ")"))

  }, error = function(e) {
    write_progress(progress_file, paste0("  ERROR in comparison '", comp_id, "': ", e$message))
    writeLines(paste("ERROR:", e$message), file.path(comp_dir, "error.txt"))
  })
}

write_progress(progress_file, "[4/5] Running functional enrichment analysis...")

# Locate the enrichment script: same directory as this pipeline script
r_scripts_path <- Sys.getenv("R_SCRIPTS_PATH", unset = "/app/r_scripts")
enrichment_script <- file.path(r_scripts_path, "functional_enrichment_legacy.R")

if (!file.exists(enrichment_script)) {
  write_progress(progress_file, paste0("  WARNING: functional_enrichment_legacy.R not found at ",
                                       enrichment_script, " — skipping enrichment"))
} else {
  anno_db_dir <- opt[["anno-db-dir"]]
  species     <- opt$species
  enrich_dbs  <- opt[["enrichment-databases"]]

  # Run enrichment for each comparison that produced filtered DEG results
  comp_dirs <- list.dirs(file.path(opt$outdir, "comparisons"), full.names = TRUE, recursive = FALSE)

  for (comp_dir in comp_dirs) {
    comp_id      <- basename(comp_dir)
    deg_filtered <- file.path(comp_dir, "results_filtered.tsv")

    if (!file.exists(deg_filtered)) {
      write_progress(progress_file, paste0("  Skipping enrichment for '", comp_id, "' (no filtered results)"))
      next
    }

    enrichment_outdir <- file.path(comp_dir, "enrichment_legacy")
    dir.create(enrichment_outdir, showWarnings = FALSE, recursive = TRUE)
    enrichment_prefix <- file.path(enrichment_outdir, "functional_enrichment")

    write_progress(progress_file, paste0("  [enrichment] ", comp_id))

    enrich_exit <- system2(
      "Rscript",
      args = c(
        enrichment_script,
        "--deg-results",   deg_filtered,
        "--anno-db-dir",   anno_db_dir,
        "--species",       species,
        "--output-prefix", enrichment_prefix,
        "--pvalue-cutoff", "0.05",
        "--r-cutoff",      "3",
        "--databases",     enrich_dbs
      ),
      stdout = FALSE,
      stderr = FALSE
    )

    if (!is.null(enrich_exit) && enrich_exit != 0) {
      write_progress(progress_file, paste0("    WARNING: enrichment returned non-zero exit for '", comp_id, "'"))
    } else {
      write_progress(progress_file, paste0("    Enrichment done for '", comp_id, "'"))
    }
  }
}

write_progress(progress_file, "[5/5] Pipeline complete!")
write_progress(progress_file, paste0("Results written to: ", opt$outdir))

# Write a final manifest of outputs for the Python service
manifest <- list(
  has_vst             = file.exists(file.path(opt$outdir, "vst_counts.tsv")),
  has_normalized      = file.exists(file.path(opt$outdir, "normalized_counts.tsv")),
  has_pca             = file.exists(file.path(opt$outdir, "pca_data.json")),
  has_qc              = file.exists(file.path(opt$outdir, "qc_report.json")),
  n_comparisons_run   = sum(file.exists(file.path(opt$outdir, "comparisons",
    comparisons_df$comparison_id, "genolens_deg.csv")))
)
write(toJSON(manifest, auto_unbox = TRUE),
      file.path(opt$outdir, "pipeline_manifest.json"))

message("\n=== GenoLens DESeq2 Pipeline Finished ===")
