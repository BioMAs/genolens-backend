#!/usr/bin/env Rscript
# =============================================================================
# Checks how run_multimethod_pipeline.R decides which genes are DEGs.
#
# - With --method all, DESeq2 decides (padj and log2FC). edgeR, limma and the
#   Stouffer combination are concordance columns only: Stouffer assumes
#   independent tests, while the three methods analyse the same counts.
# - --min-reps auto keeps a gene when it reaches --min-count in as many samples
#   as the smallest compared group holds (edgeR::filterByExpr's rule).
#
# Runs the real pipeline on small synthetic datasets.
#
#   Rscript r_scripts/tests/test_deg_criterion.R
#
# Needs the pipeline's R packages (DESeq2, edgeR, limma, tidyverse, optparse,
# jsonlite), i.e. the r-worker image or a local R with them installed.
# Exits non-zero if any check fails.
# =============================================================================

suppressPackageStartupMessages({
  library(jsonlite)
  library(readr)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", file_arg)))
pipeline <- normalizePath(file.path(here, "..", "run_multimethod_pipeline.R"))

set.seed(7)
work <- tempfile("degcrit_")
dir.create(work)

write_tsv_file <- function(name, df) {
  path <- file.path(work, name)
  write.table(df, path, sep = "\t", quote = FALSE, row.names = FALSE)
  path
}

write_counts <- function(name, counts) {
  write_tsv_file(name, data.frame(gene_id = rownames(counts), counts, check.names = FALSE))
}

comparisons_path <- write_tsv_file("comparisons.tsv", data.frame(
  comparison = "KO_vs_WT", condition1 = "KO", condition2 = "WT"
))

# KO x4, WT x3, plus one sample of a condition no comparison uses. KO
# over-expresses the first 40 genes. The last two genes reach 50 counts in two
# KO samples only: a 2-sample filter keeps them, the auto filter must not.
samples <- sprintf("S%d", 1:8)
condition <- c(rep("KO", 4), rep("WT", 3), "X")
counts <- matrix(rnbinom(300 * 8, mu = 200, size = 10), nrow = 300,
                 dimnames = list(sprintf("G%03d", 1:300), samples))
counts[1:40, condition == "KO"] <- counts[1:40, condition == "KO"] * 6L
counts[299:300, ] <- 0L
counts[299:300, c("S1", "S2")] <- 50L
counts_path <- write_counts("counts.tsv", counts)
samples_path <- write_tsv_file("samples.tsv", data.frame(sample_id = samples, condition = condition))

run_pipeline <- function(method, extra = character(), counts_file = counts_path,
                         samples_file = samples_path) {
  outdir <- tempfile("out_", tmpdir = work)
  args <- c(pipeline,
            "--counts", counts_file, "--samples", samples_file,
            "--comparisons", comparisons_path, "--outdir", outdir,
            "--method", method, "--design", "condition", "--threads", "1",
            "--min-genes", "10", "--min-reads", "10", "--min-count", "10", extra)
  log <- suppressWarnings(system2("Rscript", args, stdout = TRUE, stderr = TRUE))
  status <- attr(log, "status")
  manifest_path <- file.path(outdir, "pipeline_manifest.json")
  deg_path <- file.path(outdir, "comparisons", "KO_vs_WT", "genolens_deg.csv")
  list(
    status = if (is.null(status)) 0L else status,
    log = paste(log, collapse = "\n"),
    manifest = if (file.exists(manifest_path)) fromJSON(manifest_path) else NULL,
    deg = if (file.exists(deg_path)) read_csv(deg_path, show_col_types = FALSE) else NULL
  )
}

failures <- 0L
check <- function(label, ok, res) {
  if (isTRUE(ok)) {
    cat("PASS ", label, "\n", sep = "")
  } else {
    failures <<- failures + 1L
    cat("FAIL ", label, "\n--- pipeline output ---\n", res$log, "\n-----------------------\n", sep = "")
  }
}

# The labels the pipeline should have written for `padj_col` at the default
# thresholds (FDR 0.05, |log2FC| >= 1), from the table's own columns.
expected_labels <- function(deg, padj_col) {
  padj <- deg[[padj_col]]
  lfc <- deg[["logFC:KO_vs_WT"]]
  sig <- !is.na(padj) & padj < 0.05 & !is.na(lfc) & abs(lfc) >= 1
  ifelse(sig, ifelse(lfc > 0, "UP", "DOWN"), NA_character_)
}
labels_follow <- function(res, padj_col) {
  !is.null(res$deg) && padj_col %in% names(res$deg) &&
    identical(as.character(res$deg[["contrast:KO_vs_WT"]]), expected_labels(res$deg, padj_col))
}
stats_of <- function(res) res$manifest$comparison_stats$KO_vs_WT

# 1. --method all: DESeq2 decides; the other methods stay as columns.
all_res <- run_pipeline("all")
deseq_res <- run_pipeline("deseq2")
check("--method all runs", all_res$status == 0 && !is.null(all_res$deg), all_res)
check("--method all labels DEGs on padj.DESeq2",
      labels_follow(all_res, "padj.DESeq2:KO_vs_WT"), all_res)
check("--method all reports DESeq2 as the deciding method",
      identical(stats_of(all_res)$primary_method, "DESeq2"), all_res)
check("--method all takes its log2FC from DESeq2",
      !is.null(deseq_res$deg) &&
        isTRUE(all.equal(all_res$deg[["logFC:KO_vs_WT"]], deseq_res$deg[["logFC:KO_vs_WT"]])),
      all_res)
check("--method all keeps edgeR, limma, Stouffer and n_significant_methods as columns",
      all(c("padj.edgeR:KO_vs_WT", "padj.limma:KO_vs_WT", "padj.Stouffer:KO_vs_WT",
            "pvalue.Stouffer:KO_vs_WT", "n_significant_methods") %in% names(all_res$deg)),
      all_res)
check("manifest n_up / n_down count the written labels",
      identical(as.integer(stats_of(all_res)$n_up),
                sum(all_res$deg[["contrast:KO_vs_WT"]] == "UP", na.rm = TRUE)) &&
        identical(as.integer(stats_of(all_res)$n_down),
                  sum(all_res$deg[["contrast:KO_vs_WT"]] == "DOWN", na.rm = TRUE)),
      all_res)

# 2. A single method decides alone and leaves no empty column for the others.
edger_res <- run_pipeline("edger")
check("--method edger labels DEGs on padj.edgeR",
      labels_follow(edger_res, "padj.edgeR:KO_vs_WT"), edger_res)
check("--method edger reports edgeR as the deciding method",
      identical(stats_of(edger_res)$primary_method, "edgeR"), edger_res)
check("--method edger writes no padj.DESeq2 column",
      !is.null(edger_res$deg) && !("padj.DESeq2:KO_vs_WT" %in% names(edger_res$deg)), edger_res)
limma_res <- run_pipeline("limma")
check("--method limma labels DEGs on padj.limma",
      labels_follow(limma_res, "padj.limma:KO_vs_WT") &&
        identical(stats_of(limma_res)$primary_method, "limma"), limma_res)

# 3. --method all falls back to edgeR when DESeq2 cannot run: every gene has a
#    zero somewhere, so DESeq2 has no geometric mean to size-factor on.
zero_counts <- counts
for (g in seq_len(nrow(zero_counts))) zero_counts[g, ((g - 1L) %% 7L) + 1L] <- 0L
zero_res <- run_pipeline("all", counts_file = write_counts("zero_counts.tsv", zero_counts))
check("--method all falls back to edgeR when DESeq2 fails",
      identical(stats_of(zero_res)$primary_method, "edgeR") &&
        labels_follow(zero_res, "padj.edgeR:KO_vs_WT"), zero_res)

# 4. Low-count filter.
fixed_res <- run_pipeline("deseq2", c("--min-reps", "2"))
check("--min-reps 2 keeps genes seen in two samples",
      identical(as.integer(fixed_res$manifest$qc_report$genes_after_filter), 300L) &&
        identical(fixed_res$manifest$qc_report$min_reps_mode, "fixed") &&
        identical(as.integer(fixed_res$manifest$qc_report$min_reps_threshold), 2L),
      fixed_res)
auto_res <- run_pipeline("deseq2", c("--min-reps", "auto"))
check("--min-reps auto uses the smallest compared group (WT, 3), not the unused X",
      identical(as.integer(auto_res$manifest$qc_report$min_reps_threshold), 3L) &&
        identical(auto_res$manifest$qc_report$min_reps_mode, "auto") &&
        identical(as.integer(auto_res$manifest$qc_report$genes_after_filter), 298L),
      auto_res)

# 20 vs 20: like filterByExpr, groups above 10 samples relax the requirement to
# 10 + 0.7 * (n - 10) = 17 samples.
big_samples <- sprintf("B%02d", 1:40)
big_counts <- matrix(rnbinom(200 * 40, mu = 200, size = 10), nrow = 200,
                     dimnames = list(sprintf("G%03d", 1:200), big_samples))
big_res <- run_pipeline(
  "deseq2", c("--min-reps", "auto"),
  counts_file = write_counts("big_counts.tsv", big_counts),
  samples_file = write_tsv_file("big_samples.tsv", data.frame(
    sample_id = big_samples, condition = rep(c("KO", "WT"), each = 20)
  ))
)
check("--min-reps auto relaxes large groups like filterByExpr (20 -> 17)",
      identical(as.integer(big_res$manifest$qc_report$min_reps_threshold), 17L), big_res)

for (bad in c("0", "abc", "2.5")) {
  bad_res <- run_pipeline("deseq2", c("--min-reps", bad))
  check(paste0("--min-reps ", bad, " is an explicit error"),
        bad_res$status != 0 && grepl("--min-reps", bad_res$log, fixed = TRUE), bad_res)
}

unlink(work, recursive = TRUE)
if (failures > 0) {
  cat(failures, "check(s) failed\n")
  quit(status = 1)
}
cat("All DEG-criterion checks passed\n")
