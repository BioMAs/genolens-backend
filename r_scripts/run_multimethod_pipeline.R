#!/usr/bin/env Rscript
# =============================================================================
# GenoLens Multi-Method DEA Pipeline
# Wraps DESeq2 / edgeR / limma-voom with optional Stouffer combination.
# Outputs genolens_deg.csv per comparison in the format expected by the
# GenoLens Python backend (padj.Method:comparison, logFC:comparison columns).
# =============================================================================

suppressPackageStartupMessages({
  library(DESeq2)
  library(edgeR)
  library(limma)
  library(tidyverse)
  library(optparse)
  library(jsonlite)
})

# =============================================================================
# Stouffer's p-value combination
# =============================================================================

weighted_stouffer_pvalue <- function(pvalues, weights = NULL) {
  if (is.null(weights)) weights <- rep(1, length(pvalues))
  idx <- which(!is.na(pvalues))
  pvalues <- pvalues[idx]
  weights <- weights[idx]
  n <- length(pvalues)
  if (n == 0) return(NA)
  if (n == 1) return(pvalues)
  pvalues <- pmin(pvalues, 0.95)
  pvalues <- pmax(pvalues, 1e-10)
  # Use upper-tail forms to avoid catastrophic cancellation: 1 - pnorm(z) collapses
  # to exactly 0 for large z (pnorm rounds to 1.0), which made Stouffer p/padj show 0.
  z_scores <- qnorm(pvalues, lower.tail = FALSE)
  z_combined <- sum(weights * z_scores) / sqrt(sum(weights^2))
  p_combined <- pnorm(z_combined, lower.tail = FALSE)
  return(max(min(p_combined, 1), 0))
}

# =============================================================================
# CLI arguments
# =============================================================================

option_list <- list(
  make_option("--counts",      type = "character", help = "Count matrix TSV (genes x samples)"),
  make_option("--samples",     type = "character", help = "Sample metadata TSV"),
  make_option("--comparisons", type = "character", help = "Comparisons TSV (comparison, condition1, condition2)"),
  make_option("--outdir",      type = "character", help = "Output directory"),
  make_option("--design",      type = "character", default = "auto",  help = "Design: auto | condition | batch_condition"),
  make_option("--fdr",         type = "double",    default = 0.05,    help = "FDR threshold"),
  make_option("--min-log2fc",  type = "double",    default = 1.0,     help = "Minimum |log2FC| threshold"),
  make_option("--min-reads",   type = "integer",   default = 10L,     help = "Min total reads per sample"),
  make_option("--min-genes",   type = "integer",   default = 200L,    help = "Min expressed genes per sample"),
  make_option("--min-count",   type = "integer",   default = 5L,      help = "Min count per gene (CPM filter)"),
  make_option("--min-reps",    type = "integer",   default = 2L,      help = "Min replicates per condition"),
  make_option("--threads",     type = "integer",   default = 4L,      help = "Number of threads"),
  make_option("--species",     type = "character", default = "human", help = "Species"),
  make_option("--method",      type = "character", default = "all",
              help = "DEA method: 'all' (DESeq2+edgeR+limma+Stouffer), 'deseq2', 'edger', 'limma'")
)

opt <- parse_args(OptionParser(option_list = option_list))
stopifnot(!is.null(opt$counts), !is.null(opt$samples), !is.null(opt$comparisons), !is.null(opt$outdir))

run_deseq2 <- opt$method %in% c("all", "deseq2")
run_edger   <- opt$method %in% c("all", "edger")
run_limma   <- opt$method %in% c("all", "limma")

dir.create(opt$outdir, recursive = TRUE, showWarnings = FALSE)

message("\n=== GenoLens DEA Pipeline ===")
message("Method: ", opt$method)
message("FDR: ", opt$fdr, "  |  min_log2FC: ", opt$`min-log2fc`)

# =============================================================================
# Load data
# =============================================================================

message("\n[1/5] Loading input files...")
counts_raw <- read_tsv(opt$counts,     col_types = cols(), show_col_types = FALSE)
samples_df  <- read_tsv(opt$samples,   col_types = cols(), show_col_types = FALSE)
contrasts   <- read_tsv(opt$comparisons, col_types = cols(), show_col_types = FALSE)

# Normalise column names
colnames(samples_df) <- tolower(colnames(samples_df))

# Identify sample_id and condition columns
sid_col  <- intersect(c("sample_id", "sample", "sampleid", "id"), colnames(samples_df))[1]
cond_col <- intersect(c("condition", "group", "treatment", "genotype"), colnames(samples_df))[1]
batch_col <- intersect(c("batch", "run", "lane"), colnames(samples_df))
batch_col <- if (length(batch_col) > 0) batch_col[1] else NULL

if (is.na(sid_col) || is.na(cond_col)) {
  stop("samples.tsv must have 'sample_id' and 'condition' columns (case-insensitive)")
}

# Identify gene_id column in counts
gene_col <- intersect(c("gene_id", "geneid", "gene", "id"), colnames(counts_raw))[1]
if (is.na(gene_col)) gene_col <- colnames(counts_raw)[1]

gene_name_col <- intersect(c("gene_name", "symbol", "Symbol", "GeneName"), colnames(counts_raw))
gene_name_col <- if (length(gene_name_col) > 0) gene_name_col[1] else NULL

gene_ids   <- counts_raw[[gene_col]]
gene_names <- if (!is.null(gene_name_col)) counts_raw[[gene_name_col]] else gene_ids

# Numeric count matrix
sample_ids <- samples_df[[sid_col]]
count_cols <- intersect(sample_ids, colnames(counts_raw))
if (length(count_cols) < 2) {
  # try matching by column name directly
  count_cols <- setdiff(colnames(counts_raw), c(gene_col, gene_name_col))
}

counts_mat <- as.matrix(counts_raw[, count_cols, drop = FALSE])
rownames(counts_mat) <- gene_ids
storage.mode(counts_mat) <- "integer"

# Filter samples by min_reads / min_genes
message("[2/5] QC filtering...")
col_reads <- colSums(counts_mat)
col_genes <- colSums(counts_mat > 0)
keep_samples <- col_reads >= opt$`min-reads` & col_genes >= opt$`min-genes`
if (sum(keep_samples) < 2) {
  warning("All samples failed QC filter — relaxing to keep at least 2")
  keep_samples <- order(col_reads, decreasing = TRUE)[1:2]
  keep_mask <- seq_along(col_reads) %in% keep_samples
  keep_samples <- keep_mask
}
counts_mat  <- counts_mat[, keep_samples, drop = FALSE]
sample_ids  <- colnames(counts_mat)
samples_df  <- samples_df[samples_df[[sid_col]] %in% sample_ids, , drop = FALSE]

# Filter low-count genes (at least min_count in at least min_reps samples)
keep_genes <- rowSums(counts_mat >= opt$`min-count`) >= opt$`min-reps`
counts_mat  <- counts_mat[keep_genes, , drop = FALSE]
gene_ids    <- rownames(counts_mat)
gene_names  <- if (!is.null(gene_name_col)) {
  counts_raw[[gene_name_col]][counts_raw[[gene_col]] %in% gene_ids]
} else gene_ids

message("  Retained ", nrow(counts_mat), " genes and ", ncol(counts_mat), " samples")

# Normalise contrast table
colnames(contrasts) <- tolower(colnames(contrasts))
comp_col  <- intersect(c("comparison", "comparison_id", "comparison_name", "name", "contrast"), colnames(contrasts))[1]
cond1_col <- intersect(c("condition1", "test", "numerator",  "group1"), colnames(contrasts))[1]
cond2_col <- intersect(c("condition2", "ref",  "denominator","group2"), colnames(contrasts))[1]
if (is.na(comp_col) || is.na(cond1_col) || is.na(cond2_col)) {
  stop("comparisons.tsv must have comparison/condition1/condition2 columns (case-insensitive)")
}

# Honour an optional perform_analysis / run flag column (skip rows set to false)
flag_col <- intersect(c("perform_analysis", "perform", "run", "include", "active"), colnames(contrasts))[1]
if (!is.na(flag_col)) {
  keep <- tolower(trimws(as.character(contrasts[[flag_col]]))) %in% c("true", "t", "1", "yes", "y")
  contrasts <- contrasts[keep, , drop = FALSE]
  message("  perform_analysis filter: kept ", nrow(contrasts), " comparison(s)")
  if (nrow(contrasts) == 0) stop("No comparisons left to run after perform_analysis filter")
}

manifest <- list(has_vst = FALSE, comparisons = c())
results_dir <- file.path(opt$outdir, "comparisons")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

# =============================================================================
# VST normalisation (for PCA / heatmap)
# =============================================================================

message("[3/5] VST normalisation...")
tryCatch({
  col_data_all <- data.frame(
    condition = samples_df[[cond_col]],
    row.names = samples_df[[sid_col]]
  )[colnames(counts_mat), , drop = FALSE]
  col_data_all$condition <- factor(col_data_all$condition)

  dds_all <- DESeqDataSetFromMatrix(
    countData = counts_mat,
    colData   = col_data_all,
    design    = ~ condition
  )
  vst_mat <- assay(vst(dds_all, blind = TRUE))
  vst_df  <- as.data.frame(vst_mat)
  vst_df  <- cbind(gene_id = rownames(vst_df), vst_df)
  write_tsv(vst_df, file.path(opt$outdir, "vst_counts.tsv"))
  manifest$has_vst <- TRUE
  message("  VST written")
}, error = function(e) message("  VST skipped: ", conditionMessage(e)))

# =============================================================================
# Per-comparison DEA
# =============================================================================

message("[4/5] Running DEA (method=", opt$method, ")...")

for (i in seq_len(nrow(contrasts))) {
  comp_name <- contrasts[[comp_col]][i]
  cond1     <- contrasts[[cond1_col]][i]
  cond2     <- contrasts[[cond2_col]][i]

  message("\n  Comparison: ", comp_name, "  (", cond1, " vs ", cond2, ")")

  # Subset samples for this comparison
  smp <- samples_df[samples_df[[cond_col]] %in% c(cond1, cond2), ]
  sids_comp <- smp[[sid_col]]
  sids_comp <- intersect(sids_comp, colnames(counts_mat))

  if (length(sids_comp) < 2) {
    message("    Skipping: <2 samples found for conditions")
    next
  }

  cnt_sub <- counts_mat[, sids_comp, drop = FALSE]
  smp_sub <- smp[smp[[sid_col]] %in% sids_comp, ]

  cond_vec <- factor(smp_sub[[cond_col]], levels = c(cond2, cond1))  # ref=cond2, test=cond1
  has_batch <- !is.null(batch_col) && batch_col %in% colnames(smp_sub) && opt$design != "condition" &&
    length(unique(smp_sub[[batch_col]])) > 1
  n_per_cond <- table(cond_vec)

  # ── Build result frame base ─────────────────────────────────────────────────
  base_df <- data.frame(gene_id = rownames(cnt_sub), stringsAsFactors = FALSE)
  if (!is.null(gene_name_col)) {
    gn_map <- setNames(counts_raw[[gene_name_col]], counts_raw[[gene_col]])
    base_df$gene_name <- gn_map[base_df$gene_id]
  }

  lfc_col        <- paste0("logFC:",         comp_name)
  padj_deseq_col <- paste0("padj.DESeq2:",   comp_name)
  padj_edger_col <- paste0("padj.edgeR:",    comp_name)
  padj_limma_col <- paste0("padj.limma:",    comp_name)
  padj_stouf_col <- paste0("padj.Stouffer:", comp_name)
  pval_deseq_col <- paste0("pvalue.DESeq2:", comp_name)
  pval_edger_col <- paste0("pvalue.edgeR:",  comp_name)
  pval_limma_col <- paste0("pvalue.limma:",  comp_name)
  pval_stouf_col <- paste0("pvalue.Stouffer:", comp_name)
  contrast_col   <- paste0("contrast:",      comp_name)

  # ── DESeq2 ──────────────────────────────────────────────────────────────────
  deseq_padj   <- rep(NA_real_, nrow(cnt_sub))
  deseq_pvalue <- rep(NA_real_, nrow(cnt_sub))
  deseq_lfc    <- rep(NA_real_, nrow(cnt_sub))
  basemean     <- rep(NA_real_, nrow(cnt_sub))

  if (run_deseq2) {
    tryCatch({
      col_data <- data.frame(condition = cond_vec, row.names = sids_comp)
      if (has_batch) {
        batch_vec <- factor(smp_sub[[batch_col]])
        col_data$batch <- batch_vec
        design_formula <- ~ batch + condition
      } else {
        design_formula <- ~ condition
      }
      dds <- DESeqDataSetFromMatrix(countData = cnt_sub, colData = col_data, design = design_formula)
      dds <- DESeq(dds, quiet = TRUE)
      res <- results(dds, contrast = c("condition", cond1, cond2), alpha = opt$fdr)
      res_df <- as.data.frame(res)
      deseq_padj   <- res_df$padj
      deseq_pvalue <- res_df$pvalue
      deseq_lfc    <- res_df$log2FoldChange
      basemean     <- res_df$baseMean
      message("    DESeq2: ", sum(!is.na(deseq_padj) & deseq_padj < opt$fdr, na.rm = TRUE), " sig genes")
    }, error = function(e) message("    DESeq2 failed: ", conditionMessage(e)))
  }

  # ── edgeR ────────────────────────────────────────────────────────────────────
  edger_padj   <- rep(NA_real_, nrow(cnt_sub))
  edger_pvalue <- rep(NA_real_, nrow(cnt_sub))
  edger_lfc    <- rep(NA_real_, nrow(cnt_sub))

  if (run_edger) {
    tryCatch({
      dge <- DGEList(counts = cnt_sub, group = cond_vec)
      dge <- calcNormFactors(dge, method = "TMM")
      if (has_batch) {
        batch_vec <- factor(smp_sub[[batch_col]])
        design_e <- model.matrix(~ batch_vec + cond_vec)
      } else {
        design_e <- model.matrix(~ cond_vec)
      }
      dge <- estimateDisp(dge, design_e, robust = TRUE)
      fit <- glmQLFit(dge, design_e)
      qlt <- glmQLFTest(fit, coef = ncol(design_e))
      tt  <- topTags(qlt, n = Inf, sort.by = "none")$table
      edger_padj   <- tt$FDR
      edger_pvalue <- tt$PValue
      edger_lfc    <- tt$logFC
      if (all(is.na(basemean)))  basemean  <- rowMeans(cpm(dge))
      message("    edgeR:  ", sum(!is.na(edger_padj) & edger_padj < opt$fdr, na.rm = TRUE), " sig genes")
    }, error = function(e) message("    edgeR failed: ", conditionMessage(e)))
  }

  # ── limma-voom ───────────────────────────────────────────────────────────────
  limma_padj   <- rep(NA_real_, nrow(cnt_sub))
  limma_pvalue <- rep(NA_real_, nrow(cnt_sub))
  limma_lfc    <- rep(NA_real_, nrow(cnt_sub))

  if (run_limma) {
    tryCatch({
      dge_l <- DGEList(counts = cnt_sub, group = cond_vec)
      dge_l <- calcNormFactors(dge_l, method = "TMM")
      if (has_batch) {
        batch_vec <- factor(smp_sub[[batch_col]])
        design_l <- model.matrix(~ batch_vec + cond_vec)
      } else {
        design_l <- model.matrix(~ cond_vec)
      }
      v   <- voom(dge_l, design_l, plot = FALSE)
      fit <- lmFit(v, design_l)
      fit <- eBayes(fit)
      tt  <- topTable(fit, coef = ncol(design_l), n = Inf, sort.by = "none")
      limma_padj   <- tt$adj.P.Val
      limma_pvalue <- tt$P.Value
      limma_lfc    <- tt$logFC
      if (all(is.na(basemean)))  basemean  <- rowMeans(cpm(dge_l))
      message("    limma:  ", sum(!is.na(limma_padj) & limma_padj < opt$fdr, na.rm = TRUE), " sig genes")
    }, error = function(e) message("    limma failed: ", conditionMessage(e)))
  }

  # ── Stouffer combination (only if method=all) ──────────────────────────────
  stouffer_padj   <- rep(NA_real_, nrow(cnt_sub))
  stouffer_pvalue <- rep(NA_real_, nrow(cnt_sub))
  n_sig_methods   <- rep(0L, nrow(cnt_sub))

  if (opt$method == "all") {
    avail_pvals <- Filter(function(v) !all(is.na(v)), list(deseq_pvalue, edger_pvalue, limma_pvalue))
    if (length(avail_pvals) >= 2) {
      pval_mat <- do.call(cbind, avail_pvals)
      stouffer_pvalue <- apply(pval_mat, 1, weighted_stouffer_pvalue)
      stouffer_padj   <- p.adjust(stouffer_pvalue, method = "BH")

      sig_mat <- cbind(
        !is.na(deseq_padj) & deseq_padj < opt$fdr,
        !is.na(edger_padj) & edger_padj < opt$fdr,
        !is.na(limma_padj) & limma_padj < opt$fdr
      )
      n_sig_methods <- rowSums(sig_mat, na.rm = TRUE)
      message("    Stouffer: ", sum(!is.na(stouffer_padj) & stouffer_padj < opt$fdr, na.rm = TRUE), " sig genes")
    } else if (length(avail_pvals) == 1) {
      stouffer_pvalue <- avail_pvals[[1]]
      stouffer_padj   <- p.adjust(stouffer_pvalue, method = "BH")
    }
  }

  # Choose primary padj/pvalue/lfc for the contrast column
  if (opt$method == "all") {
    primary_padj   <- stouffer_padj
    primary_pvalue <- stouffer_pvalue
  } else if (opt$method == "deseq2") {
    primary_padj   <- deseq_padj
    primary_pvalue <- deseq_pvalue
  } else if (opt$method == "edger") {
    primary_padj   <- edger_padj
    primary_pvalue <- edger_pvalue
  } else if (opt$method == "limma") {
    primary_padj   <- limma_padj
    primary_pvalue <- limma_pvalue
  }

  # Canonical log2FoldChange for filtering/reporting: limma (matches pipe_scilicium's
  # step3_multimethod_comparison.R, which always uses limma-voom's logFC), falling
  # back to edgeR then DESeq2 if limma didn't run.
  if (!all(is.na(limma_lfc))) {
    canonical_lfc <- limma_lfc
  } else if (!all(is.na(edger_lfc))) {
    canonical_lfc <- edger_lfc
  } else {
    canonical_lfc <- deseq_lfc
  }

  # ── Contrast column (UP/DOWN regulation) ───────────────────────────────────
  regulation <- rep(NA_character_, nrow(cnt_sub))
  sig_mask   <- !is.na(primary_padj) & primary_padj < opt$fdr & !is.na(canonical_lfc) & abs(canonical_lfc) >= opt$`min-log2fc`
  regulation[sig_mask & canonical_lfc > 0]  <- "UP"
  regulation[sig_mask & canonical_lfc <= 0] <- "DOWN"

  # ── Assemble output data frame ──────────────────────────────────────────────
  out_df <- base_df
  out_df$baseMean <- basemean

  out_df[[lfc_col]]        <- canonical_lfc
  out_df[[padj_deseq_col]] <- deseq_padj
  out_df[[pval_deseq_col]] <- deseq_pvalue

  if (run_edger) {
    out_df[[padj_edger_col]] <- edger_padj
    out_df[[pval_edger_col]] <- edger_pvalue
  }
  if (run_limma) {
    out_df[[padj_limma_col]] <- limma_padj
    out_df[[pval_limma_col]] <- limma_pvalue
  }
  if (opt$method == "all") {
    out_df[[padj_stouf_col]]  <- stouffer_padj
    out_df[[pval_stouf_col]]  <- stouffer_pvalue
    out_df$n_significant_methods <- n_sig_methods
  }

  out_df[[contrast_col]] <- regulation

  # ── Write output ────────────────────────────────────────────────────────────
  comp_dir <- file.path(results_dir, comp_name)
  dir.create(comp_dir, recursive = TRUE, showWarnings = FALSE)
  write_csv(out_df, file.path(comp_dir, "genolens_deg.csv"))

  n_up   <- sum(regulation == "UP",   na.rm = TRUE)
  n_down <- sum(regulation == "DOWN", na.rm = TRUE)
  message("    Output: ", nrow(out_df), " genes | UP=", n_up, " DOWN=", n_down)
  manifest$comparisons <- c(manifest$comparisons, comp_name)
}

# =============================================================================
# Manifest
# =============================================================================

message("\n[5/5] Writing manifest...")
write_json(manifest, file.path(opt$outdir, "pipeline_manifest.json"), auto_unbox = TRUE)
message("=== Pipeline complete ===\n")
