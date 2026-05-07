#!/usr/bin/env Rscript
# =============================================================================
# GenoLens Self-Service Multi-Method Analysis Pipeline
# =============================================================================
#
# Combines DESeq2, edgeR (QLF) and limma-voom with Stouffer's method for
# robust differential expression analysis, plus PCA and UMAP.
#
# Usage:
#   Rscript run_multimethod_pipeline.R \
#     --counts      counts.tsv        \
#     --samples     samples.tsv       \
#     --comparisons comparisons.tsv   \
#     --outdir      output/           \
#     [options]
#
# Required input format:
#   counts.tsv       : TSV with first column 'gene_id', remaining = sample IDs.
#                      Optional 'gene_name' column used for annotation.
#   samples.tsv      : TSV with 'sample_id' and 'condition' columns.
#                      Optional: 'batch', 'sex', other covariates.
#   comparisons.tsv  : TSV with 'comparison_id', 'condition1', 'condition2'.
#                      Optional 'perform_analysis' column (TRUE/FALSE).
#
# Output (at --outdir level):
#   filtering_report.txt
#   normalized_counts.tsv
#   vst_counts.tsv
#   pca_data.json
#   umap_data.json        (NEW)
#   qc_report.json
#   pipeline_manifest.json
#
# Output per comparison (in --outdir/comparisons/{comparison_id}/):
#   genolens_deg.csv          : GenoLens-compatible DEG CSV (Stouffer padj, limma logFC)
#   results_filtered.tsv      : Significant DEGs (Stouffer padj + |LFC| thresholds)
#   method_pvalues.tsv        : Per-method p-values for all tested genes (NEW)
#   summary.txt
#   enrichment_legacy/        : Functional enrichment outputs
# =============================================================================

suppressPackageStartupMessages({
  library(DESeq2)
  library(edgeR)
  library(limma)
  library(tidyverse)
  library(optparse)
  library(BiocParallel)
  library(jsonlite)
  library(uwot)
})

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

write_progress <- function(progress_file, msg) {
  ts <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
  line <- paste0(ts, " | ", msg)
  message(line)
  if (!is.null(progress_file) && nchar(progress_file) > 0) {
    cat(line, "\n", file = progress_file, append = TRUE)
  }
}

weighted_stouffer_pvalue <- function(pvalues, weights = NULL) {
  # Combine p-values using weighted Stouffer's method (Stouffer-Liptak).
  if (is.null(weights)) weights <- rep(1, length(pvalues))
  idx <- which(!is.na(pvalues))
  pvalues <- pvalues[idx]
  weights <- weights[idx]
  n <- length(pvalues)
  if (n == 1) return(pvalues)
  if (n == 0) return(NA_real_)
  # Constrain for numerical stability
  pvalues <- pmin(pvalues, 0.95)
  pvalues <- pmax(pvalues, 1e-10)
  z_scores <- qnorm(1 - pvalues)
  combined_z <- sum(weights * z_scores) / sqrt(sum(weights^2))
  1 - pnorm(combined_z)
}

# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------

option_list <- list(
  make_option("--counts",                type = "character", help = "Count matrix TSV"),
  make_option("--samples",               type = "character", help = "Sample metadata TSV"),
  make_option("--comparisons",           type = "character", help = "Comparisons TSV"),
  make_option("--outdir",                type = "character", help = "Output directory"),
  make_option("--design",                type = "character", default = "auto",
              help = "Design: auto | condition | batch_condition [default: auto]"),
  make_option("--min-reads",             type = "integer",   default = 100000),
  make_option("--min-genes",             type = "integer",   default = 500),
  make_option("--min-count",             type = "integer",   default = 10),
  make_option("--min-reps",              type = "integer",   default = 2),
  make_option("--fdr",                   type = "double",    default = 0.05),
  make_option("--min-log2fc",            type = "double",    default = 1.0),
  make_option("--threads",               type = "integer",   default = 1),
  make_option("--progress-file",         type = "character", default = ""),
  make_option("--species",               type = "character", default = "human"),
  make_option("--anno-db-dir",           type = "character", default = "/app/anno_db"),
  make_option("--enrichment-databases",  type = "character", default = "all")
)

opt <- parse_args(OptionParser(option_list = option_list))

for (arg_name in c("counts", "samples", "comparisons", "outdir")) {
  if (is.null(opt[[arg_name]])) stop("Missing required argument: --", arg_name)
}

progress_file <- if (nchar(opt[["progress-file"]]) > 0) opt[["progress-file"]] else NULL

write_progress(progress_file, "=== GenoLens Multi-Method Pipeline Starting ===")

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

write_progress(progress_file, "[0/5] Loading input files...")

counts_raw <- tryCatch(
  read_tsv(opt$counts, col_types = cols(), show_col_types = FALSE),
  error = function(e) stop("Cannot read counts file: ", e$message)
)

if (!"gene_id" %in% colnames(counts_raw)) {
  stop("Count matrix must have a 'gene_id' column as the first column.")
}

# Extract gene_name mapping if present (replaces GTF dependency)
gene_names_df <- NULL
if ("gene_name" %in% colnames(counts_raw)) {
  gene_names_df <- counts_raw %>%
    select(gene_id, gene_name) %>%
    filter(!is.na(gene_id))
  counts_raw <- counts_raw %>% select(-gene_name)
}

counts_matrix <- counts_raw %>%
  column_to_rownames("gene_id") %>%
  as.data.frame()

message("  - Loaded count matrix: ", nrow(counts_matrix), " genes x ", ncol(counts_matrix), " samples")

samples_df <- tryCatch(
  read_tsv(opt$samples, col_types = cols(), show_col_types = FALSE),
  error = function(e) stop("Cannot read samples file: ", e$message)
)
for (col in c("sample_id", "condition")) {
  if (!col %in% colnames(samples_df)) stop("Samples file missing required column: '", col, "'")
}
message("  - Loaded ", nrow(samples_df), " sample records")

comparisons_df <- tryCatch(
  read_tsv(opt$comparisons, col_types = cols(), show_col_types = FALSE),
  error = function(e) stop("Cannot read comparisons file: ", e$message)
)
for (col in c("comparison_id", "condition1", "condition2")) {
  if (!col %in% colnames(comparisons_df)) stop("Comparisons file missing required column: '", col, "'")
}

if ("perform_analysis" %in% colnames(comparisons_df)) {
  comparisons_df <- comparisons_df %>%
    filter(
      perform_analysis == TRUE |
      as.character(perform_analysis) %in% c("True", "true", "TRUE", "1", "yes")
    )
}
if (nrow(comparisons_df) == 0) stop("No comparisons to run.")
message("  - ", nrow(comparisons_df), " comparison(s): ",
        paste(comparisons_df$comparison_id, collapse = ", "))

# ---------------------------------------------------------------------------
# Step 1: Sample QC and gene filtering
# ---------------------------------------------------------------------------

write_progress(progress_file, "[1/5] Filtering samples and genes...")

common_samples <- intersect(colnames(counts_matrix), samples_df$sample_id)
if (length(common_samples) == 0) {
  stop(
    "No matching samples between count matrix and samples file.\n",
    "  Matrix columns (first 5): ", paste(head(colnames(counts_matrix), 5), collapse = ", "), "\n",
    "  Samples file (first 5): ", paste(head(samples_df$sample_id, 5), collapse = ", ")
  )
}
message("  - Matched ", length(common_samples), " / ", ncol(counts_matrix), " samples")

counts_matrix <- counts_matrix[, common_samples, drop = FALSE]
samples_df    <- samples_df %>% filter(sample_id %in% common_samples)

# Sample-level QC
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
if (length(samples_pass) < 2) stop("Fewer than 2 samples passed QC.")

counts_matrix <- counts_matrix[, samples_pass, drop = FALSE]
samples_df    <- samples_df %>% filter(sample_id %in% samples_pass)

# Gene-level filtering
n_before   <- nrow(counts_matrix)
gene_pass  <- apply(counts_matrix, 1, function(x) sum(x >= opt[["min-count"]]) >= opt[["min-reps"]])
counts_matrix <- counts_matrix[gene_pass, , drop = FALSE]
message("  - Genes passing filter: ", nrow(counts_matrix),
        " (removed ", n_before - nrow(counts_matrix), ")")

if (nrow(counts_matrix) < 100) stop("Too few genes (", nrow(counts_matrix), ") after filtering.")

counts_matrix <- as.matrix(round(counts_matrix))
storage.mode(counts_matrix) <- "integer"

writeLines(c(
  paste("Samples before QC:", length(common_samples)),
  paste("Samples passing QC:", length(samples_pass)),
  paste("Samples removed:", length(samples_fail)),
  paste("Genes after filtering:", nrow(counts_matrix))
), file.path(opt$outdir, "filtering_report.txt"))

# ---------------------------------------------------------------------------
# Step 2: DESeq2 normalization (whole dataset — for VST / PCA / UMAP)
# ---------------------------------------------------------------------------

write_progress(progress_file, "[2/5] Normalizing with DESeq2 (whole dataset)...")

col_data <- samples_df %>%
  filter(sample_id %in% colnames(counts_matrix)) %>%
  arrange(match(sample_id, colnames(counts_matrix))) %>%
  as.data.frame()
rownames(col_data) <- col_data$sample_id
col_data$condition <- factor(col_data$condition)

has_batch  <- "batch" %in% colnames(col_data) && length(unique(col_data$batch)) > 1
design_str <- opt$design
if (design_str == "auto") {
  design_str <- if (has_batch) "~ batch + condition" else "~ condition"
} else if (design_str == "batch_condition") {
  design_str <- "~ batch + condition"
} else {
  design_str <- "~ condition"
}
if (has_batch && grepl("batch", design_str)) col_data$batch <- factor(col_data$batch)
message("  - Design formula: ", design_str)

dds_global <- DESeqDataSetFromMatrix(
  countData = counts_matrix,
  colData   = col_data,
  design    = as.formula(design_str)
)
dds_global <- DESeq(dds_global, parallel = (opt$threads > 1), quiet = FALSE)
message("  - DESeq2 normalization complete")

# Export normalized counts
norm_counts_df <- as.data.frame(counts(dds_global, normalized = TRUE)) %>%
  rownames_to_column("gene_id")
if (!is.null(gene_names_df)) {
  norm_counts_df <- norm_counts_df %>%
    left_join(gene_names_df, by = "gene_id") %>%
    select(gene_id, gene_name, everything())
}
write_tsv(norm_counts_df, file.path(opt$outdir, "normalized_counts.tsv"))

# ---------------------------------------------------------------------------
# Step 2b: VST + PCA
# ---------------------------------------------------------------------------

write_progress(progress_file, "[2b/5] Computing VST + PCA...")

vst_obj <- tryCatch(
  vst(dds_global, blind = FALSE),
  error = function(e) {
    message("  - VST failed, falling back to rlog: ", e$message)
    rlog(dds_global, blind = FALSE)
  }
)
vst_mat <- assay(vst_obj)
vst_df  <- as.data.frame(vst_mat) %>% rownames_to_column("gene_id")
if (!is.null(gene_names_df)) {
  vst_df <- vst_df %>%
    left_join(gene_names_df, by = "gene_id") %>%
    select(gene_id, gene_name, everything())
}
write_tsv(vst_df, file.path(opt$outdir, "vst_counts.tsv"))
message("  - VST counts exported: ", nrow(vst_df), " genes x ", ncol(vst_df) - 1, " samples")

# PCA on top 500 variable genes
top_n_pca  <- min(500, nrow(vst_mat))
gene_vars  <- apply(vst_mat, 1, var)
top_genes  <- names(sort(gene_vars, decreasing = TRUE))[seq_len(top_n_pca)]
pca_res    <- prcomp(t(vst_mat[top_genes, ]), scale. = FALSE)
var_expl   <- (pca_res$sdev^2) / sum(pca_res$sdev^2)
n_pcs      <- min(4, ncol(pca_res$x))
pca_coords <- as.data.frame(pca_res$x[, seq_len(n_pcs), drop = FALSE]) %>%
  rownames_to_column("sample_id") %>%
  left_join(col_data %>% select(sample_id, condition, any_of("batch")), by = "sample_id")

pca_json <- list(
  variance_explained = as.numeric(var_expl[seq_len(n_pcs)]),
  pc_labels          = colnames(pca_res$x)[seq_len(n_pcs)],
  samples = lapply(seq_len(nrow(pca_coords)), function(i) {
    row <- pca_coords[i, ]
    out <- list(sample_id = row$sample_id)
    for (pc in colnames(pca_res$x)[seq_len(n_pcs)]) out[[pc]] <- as.numeric(row[[pc]])
    if ("condition" %in% colnames(row)) out$condition <- as.character(row$condition)
    if ("batch" %in% colnames(row))     out$batch     <- as.character(row$batch)
    out
  })
)
write(toJSON(pca_json, auto_unbox = TRUE, digits = 6),
      file.path(opt$outdir, "pca_data.json"))
message("  - PCA data exported (PC1-PC", n_pcs, " on top ", top_n_pca, " variable genes)")

# QC report
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
  has_batch_correction = has_batch,
  pipeline             = "multimethod"
)
write(toJSON(qc_json, auto_unbox = TRUE),
      file.path(opt$outdir, "qc_report.json"))

# ---------------------------------------------------------------------------
# Step 2c: UMAP via uwot
# ---------------------------------------------------------------------------

write_progress(progress_file, "[2c/5] Computing UMAP...")

tryCatch({
  # Use same top variable genes as PCA, transpose: samples × genes
  umap_input <- t(vst_mat[top_genes, ])

  # n_neighbors must be < n_samples
  n_samp     <- nrow(umap_input)
  n_neigh    <- min(15L, max(2L, n_samp - 1L))

  set.seed(42)
  umap_coords <- uwot::umap(
    umap_input,
    n_components = 2,
    n_neighbors  = n_neigh,
    metric       = "euclidean",
    verbose      = FALSE
  )

  umap_data <- data.frame(
    sample_id = rownames(umap_input),
    UMAP1     = umap_coords[, 1],
    UMAP2     = umap_coords[, 2],
    stringsAsFactors = FALSE
  ) %>%
    left_join(col_data %>% select(sample_id, condition, any_of("batch")), by = "sample_id")

  umap_json <- lapply(seq_len(nrow(umap_data)), function(i) {
    row <- umap_data[i, ]
    out <- list(sample_id = row$sample_id,
                UMAP1     = round(as.numeric(row$UMAP1), 6),
                UMAP2     = round(as.numeric(row$UMAP2), 6))
    if ("condition" %in% colnames(row)) out$condition <- as.character(row$condition)
    if ("batch" %in% colnames(row))     out$batch     <- as.character(row$batch)
    out
  })
  write(toJSON(umap_json, auto_unbox = TRUE),
        file.path(opt$outdir, "umap_data.json"))
  message("  - UMAP data exported (", n_samp, " samples)")
}, error = function(e) {
  write_progress(progress_file, paste0("  WARNING: UMAP failed — ", e$message))
  # Write empty file so manifest can detect failure without crashing
  write("{}", file.path(opt$outdir, "umap_data.json"))
})

# ---------------------------------------------------------------------------
# Step 3: Per-comparison Multi-Method DE
# ---------------------------------------------------------------------------

write_progress(progress_file, paste0(
  "[3/5] Running multi-method DE for ", nrow(comparisons_df), " comparison(s)..."
))

condition_levels <- levels(col_data$condition)
fdr_threshold    <- opt$fdr
min_log2fc       <- opt[["min-log2fc"]]

for (i in seq_len(nrow(comparisons_df))) {
  comp    <- comparisons_df[i, ]
  comp_id <- comp$comparison_id
  cond1   <- comp$condition1
  cond2   <- comp$condition2

  comp_dir <- file.path(opt$outdir, "comparisons", comp_id)
  dir.create(comp_dir, recursive = TRUE, showWarnings = FALSE)

  write_progress(progress_file,
    paste0("  [", i, "/", nrow(comparisons_df), "] ", comp_id, ": ", cond1, " vs ", cond2))

  if (!cond1 %in% condition_levels) {
    msg <- paste0("condition '", cond1, "' not found (available: ",
                  paste(condition_levels, collapse = ", "), ")")
    write_progress(progress_file, paste0("  SKIP: ", msg))
    writeLines(paste("SKIPPED:", msg), file.path(comp_dir, "error.txt"))
    next
  }
  if (!cond2 %in% condition_levels) {
    msg <- paste0("condition '", cond2, "' not found")
    write_progress(progress_file, paste0("  SKIP: ", msg))
    writeLines(paste("SKIPPED:", msg), file.path(comp_dir, "error.txt"))
    next
  }

  tryCatch({
    # --- Subset samples for this comparison ---
    samples_comp <- col_data %>%
      filter(condition %in% c(cond1, cond2)) %>%
      pull(sample_id)

    counts_sub <- counts_matrix[, samples_comp, drop = FALSE]
    meta_sub   <- col_data[samples_comp, , drop = FALSE]
    meta_sub$condition <- factor(as.character(meta_sub$condition), levels = c(cond2, cond1))

    # ===== Method 1: DESeq2 =====
    dds_sub <- DESeqDataSetFromMatrix(
      countData = counts_sub,
      colData   = meta_sub,
      design    = ~ condition
    )
    dds_sub <- DESeq(dds_sub, parallel = (opt$threads > 1), quiet = TRUE)

    deseq_res <- results(dds_sub, contrast = c("condition", cond1, cond2), alpha = fdr_threshold)
    deseq_df  <- as.data.frame(deseq_res) %>%
      rownames_to_column("gene_id") %>%
      select(gene_id, pvalue, padj) %>%
      rename(pvalue_deseq2 = pvalue, padj_deseq2 = padj)

    message("    DESeq2: ", sum(deseq_df$padj_deseq2 < fdr_threshold, na.rm = TRUE), " sig genes")

    # ===== Method 2: edgeR QLF =====
    dge        <- DGEList(counts = counts_sub, group = meta_sub$condition)
    dge        <- calcNormFactors(dge, method = "TMM")
    design_edger <- model.matrix(~ condition, data = meta_sub)
    dge        <- estimateDisp(dge, design_edger)
    fit_edger  <- glmQLFit(dge, design_edger)

    # Coefficient for cond1 (reference = cond2 due to factor levels above)
    target_coef <- grep(paste0("condition", cond1), colnames(design_edger), value = TRUE)
    if (length(target_coef) == 0) {
      # Fallback: last non-intercept coefficient
      target_coef <- tail(colnames(design_edger), 1)
    }
    qlf       <- glmQLFTest(fit_edger, coef = target_coef)
    edger_res <- topTags(qlf, n = Inf, sort.by = "none")$table
    edger_df  <- edger_res %>%
      rownames_to_column("gene_id") %>%
      select(gene_id, PValue, FDR) %>%
      rename(pvalue_edger = PValue, padj_edger = FDR)

    message("    edgeR:  ", sum(edger_df$padj_edger < fdr_threshold, na.rm = TRUE), " sig genes")

    # ===== Method 3: limma-voom =====
    v            <- voom(dge, design_edger, plot = FALSE)
    fit_limma    <- lmFit(v, design_edger)
    fit_limma2   <- eBayes(fit_limma)
    limma_coef   <- grep(paste0("condition", cond1), colnames(design_edger), value = TRUE)
    if (length(limma_coef) == 0) limma_coef <- tail(colnames(design_edger), 1)

    limma_res <- topTable(fit_limma2, coef = limma_coef, number = Inf, sort.by = "none")
    limma_df  <- limma_res %>%
      rownames_to_column("gene_id") %>%
      select(gene_id, logFC, AveExpr, P.Value, adj.P.Val) %>%
      rename(
        log2FoldChange = logFC,
        baseMean       = AveExpr,
        pvalue_limma   = P.Value,
        padj_limma     = adj.P.Val
      )

    message("    limma:  ", sum(limma_df$padj_limma < fdr_threshold, na.rm = TRUE), " sig genes")

    # ===== Stouffer combination =====
    combined_df <- limma_df %>%
      left_join(deseq_df, by = "gene_id") %>%
      left_join(edger_df, by = "gene_id")

    combined_df$pvalue_stouffer <- apply(
      combined_df[, c("pvalue_limma", "pvalue_deseq2", "pvalue_edger")],
      1,
      weighted_stouffer_pvalue
    )
    combined_df$padj_stouffer <- p.adjust(combined_df$pvalue_stouffer, method = "BH")
    combined_df$n_significant_methods <- rowSums(
      combined_df[, c("padj_limma", "padj_deseq2", "padj_edger")] < fdr_threshold,
      na.rm = TRUE
    )

    message("    Stouffer: ", sum(combined_df$padj_stouffer < fdr_threshold, na.rm = TRUE), " sig genes")

    # ===== Add gene names =====
    if (!is.null(gene_names_df)) {
      combined_df <- combined_df %>%
        left_join(gene_names_df, by = "gene_id") %>%
        mutate(gene_name = if_else(is.na(gene_name) | gene_name == "", gene_id, gene_name))
    } else {
      combined_df <- combined_df %>% mutate(gene_name = gene_id)
    }
    combined_df <- combined_df %>%
      select(gene_id, gene_name, everything()) %>%
      arrange(padj_stouffer)

    # ===== Filtered results =====
    results_filtered <- combined_df %>%
      filter(
        !is.na(padj_stouffer),
        padj_stouffer < fdr_threshold,
        abs(log2FoldChange) >= min_log2fc
      ) %>%
      mutate(regulation = case_when(
        log2FoldChange > 0 ~ "Up",
        log2FoldChange < 0 ~ "Down",
        TRUE               ~ "Unchanged"
      ))

    write_tsv(results_filtered, file.path(comp_dir, "results_filtered.tsv"))

    # ===== method_pvalues.tsv (NEW) =====
    method_pvalues <- combined_df %>%
      select(gene_id, gene_name,
             pvalue_limma, padj_limma,
             pvalue_deseq2, padj_deseq2,
             pvalue_edger, padj_edger,
             pvalue_stouffer, padj_stouffer,
             n_significant_methods)
    write_tsv(method_pvalues, file.path(comp_dir, "method_pvalues.tsv"))

    # ===== genolens_deg.csv (compatible format, Stouffer padj + limma LFC) =====
    comp_clean  <- gsub("[^A-Za-z0-9_]", "_", comp_id)
    genolens_deg <- combined_df %>%
      filter(!is.na(padj_stouffer)) %>%
      select(
        gene_id,
        gene_name,
        baseMean,
        !!paste0("logFC:", comp_clean)           := log2FoldChange,
        !!paste0("pvalue:", comp_clean)           := pvalue_stouffer,
        !!paste0("padj:", comp_clean)             := padj_stouffer,
        !!paste0("pvalue_deseq2:", comp_clean)    := pvalue_deseq2,
        !!paste0("padj_deseq2:", comp_clean)      := padj_deseq2,
        !!paste0("pvalue_edger:", comp_clean)     := pvalue_edger,
        !!paste0("padj_edger:", comp_clean)       := padj_edger,
        !!paste0("pvalue_limma:", comp_clean)     := pvalue_limma,
        !!paste0("padj_limma:", comp_clean)       := padj_limma,
        !!paste0("n_methods:", comp_clean)        := n_significant_methods
      )
    write_csv(genolens_deg, file.path(comp_dir, "genolens_deg.csv"))

    # ===== Summary =====
    n_tested <- sum(!is.na(combined_df$padj_stouffer))
    n_sig    <- sum(combined_df$padj_stouffer < fdr_threshold, na.rm = TRUE)
    n_up     <- sum(results_filtered$regulation == "Up",   na.rm = TRUE)
    n_down   <- sum(results_filtered$regulation == "Down", na.rm = TRUE)

    writeLines(c(
      paste("comparison_id:",  comp_id),
      paste("condition1:",     cond1),
      paste("condition2:",     cond2),
      paste("genes_tested:",   n_tested),
      paste("significant:",    n_sig),
      paste("up_regulated:",   n_up),
      paste("down_regulated:", n_down),
      paste("fdr_threshold:",  fdr_threshold),
      paste("min_log2fc:",     min_log2fc),
      paste("pipeline:",       "multimethod")
    ), file.path(comp_dir, "summary.txt"))

    write_progress(progress_file,
      paste0("    Done: ", n_sig, " sig / ", n_tested, " tested  (up:", n_up, " dn:", n_down, ")"))

  }, error = function(e) {
    write_progress(progress_file, paste0("  ERROR in '", comp_id, "': ", e$message))
    writeLines(paste("ERROR:", e$message), file.path(comp_dir, "error.txt"))
  })
}

# ---------------------------------------------------------------------------
# Step 4: Functional enrichment (legacy anno.db)
# ---------------------------------------------------------------------------

write_progress(progress_file, "[4/5] Running functional enrichment...")

r_scripts_path   <- Sys.getenv("R_SCRIPTS_PATH", unset = "/app/r_scripts")
enrichment_script <- file.path(r_scripts_path, "functional_enrichment_legacy.R")

if (!file.exists(enrichment_script)) {
  write_progress(progress_file, paste0(
    "  WARNING: functional_enrichment_legacy.R not found at ",
    enrichment_script, " — skipping"))
} else {
  anno_db_dir <- opt[["anno-db-dir"]]
  species     <- opt$species
  enrich_dbs  <- opt[["enrichment-databases"]]

  comp_dirs <- list.dirs(file.path(opt$outdir, "comparisons"),
                         full.names = TRUE, recursive = FALSE)

  for (comp_dir in comp_dirs) {
    comp_id      <- basename(comp_dir)
    deg_filtered <- file.path(comp_dir, "results_filtered.tsv")

    if (!file.exists(deg_filtered)) {
      write_progress(progress_file,
        paste0("  Skipping enrichment for '", comp_id, "' (no filtered results)"))
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
      write_progress(progress_file,
        paste0("    WARNING: enrichment non-zero exit for '", comp_id, "'"))
    } else {
      write_progress(progress_file, paste0("    Enrichment done for '", comp_id, "'"))
    }
  }
}

# ---------------------------------------------------------------------------
# Step 5: Write manifest
# ---------------------------------------------------------------------------

write_progress(progress_file, "[5/5] Pipeline complete!")

manifest <- list(
  has_vst             = file.exists(file.path(opt$outdir, "vst_counts.tsv")),
  has_normalized      = file.exists(file.path(opt$outdir, "normalized_counts.tsv")),
  has_pca             = file.exists(file.path(opt$outdir, "pca_data.json")),
  has_umap            = file.exists(file.path(opt$outdir, "umap_data.json")),
  has_qc              = file.exists(file.path(opt$outdir, "qc_report.json")),
  n_comparisons_run   = sum(file.exists(file.path(opt$outdir, "comparisons",
    comparisons_df$comparison_id, "genolens_deg.csv"))),
  has_method_pvalues  = any(file.exists(file.path(opt$outdir, "comparisons",
    comparisons_df$comparison_id, "method_pvalues.tsv"))),
  pipeline            = "multimethod"
)
write(toJSON(manifest, auto_unbox = TRUE),
      file.path(opt$outdir, "pipeline_manifest.json"))

write_progress(progress_file, paste0("Results written to: ", opt$outdir))
message("\n=== GenoLens Multi-Method Pipeline Finished ===")
