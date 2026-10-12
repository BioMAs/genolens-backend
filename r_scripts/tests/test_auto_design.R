#!/usr/bin/env Rscript
# =============================================================================
# Checks when `--design auto` keeps the batch column in run_multimethod_pipeline.R.
#
# `auto` used to add batch as soon as a batch/run/lane column had more than one
# level in the comparison. On an unbalanced donor design (SLC-26-158: 6 samples,
# 4 donors, one donor unique to each arm) ~ batch + condition leaves 1 residual
# degree of freedom and the DEG count collapses. `auto` must now fall back to
# ~ condition there, keep batch on a balanced paired design, and say which
# design it used in pipeline_manifest.json. The explicit `condition` and
# `batch_condition` modes are unchanged.
#
# Runs the real pipeline with every method (DESeq2, edgeR, limma, Stouffer: the
# design reaches each of them) on a small synthetic dataset and compares its
# DEG tables across designs.
#
#   Rscript r_scripts/tests/test_auto_design.R
#
# Needs the pipeline's R packages (DESeq2, edgeR, limma, tidyverse, optparse,
# jsonlite), i.e. the r-worker image or a local R with them installed.
# Exits non-zero if any check fails.
# =============================================================================

suppressPackageStartupMessages(library(jsonlite))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", file_arg)))
pipeline <- normalizePath(file.path(here, "..", "run_multimethod_pipeline.R"))

set.seed(158)
work <- tempfile("design_")
dir.create(work)

# 6 samples, 3 per arm, 300 genes. Each donor shifts every gene by its own
# log-normal factor (so blocking on donor changes the results) and Sertoli
# over-expresses the first 40 genes.
samples <- sprintf("S%d", 1:6)
condition <- rep(c("iso", "Sertoli"), each = 3)
donor_of_sample <- c(1, 2, 3, 1, 2, 3)
donor_effect <- matrix(exp(rnorm(300 * 3, sd = 0.6)), nrow = 300)
mu <- 150 * donor_effect[, donor_of_sample]
mu[1:40, condition == "Sertoli"] <- mu[1:40, condition == "Sertoli"] * 4
counts <- matrix(rnbinom(length(mu), mu = mu, size = 20), nrow = 300,
                 dimnames = list(sprintf("G%03d", 1:300), samples))
counts_df <- data.frame(gene_id = rownames(counts), counts, check.names = FALSE)
counts_path <- file.path(work, "counts.tsv")
write.table(counts_df, counts_path, sep = "\t", quote = FALSE, row.names = FALSE)

write_sheet <- function(name, df) {
  path <- file.path(work, name)
  write.table(df, path, sep = "\t", quote = FALSE, row.names = FALSE)
  path
}

comparisons_path <- write_sheet("comparisons.tsv", data.frame(
  comparison = "Sertoli_vs_iso", condition1 = "Sertoli", condition2 = "iso"
))

# Paired: the same 3 donors in both arms -> ~ batch + condition has 6 - 4 = 2
# residual degrees of freedom.
balanced_path <- write_sheet("balanced.tsv", data.frame(
  sample_id = samples, condition = condition, batch = c("D1", "D2", "D3", "D1", "D2", "D3")
))
# SLC-26-158 shape: D3 only in iso, D4 only in Sertoli -> 6 - 5 = 1 residual df.
unbalanced_path <- write_sheet("unbalanced.tsv", data.frame(
  sample_id = samples, condition = condition, batch = c("D1", "D2", "D3", "D1", "D2", "D4")
))
# Each arm processed in its own batch: batch and condition cannot be separated.
confounded_path <- write_sheet("confounded.tsv", data.frame(
  sample_id = samples, condition = condition, batch = c("B1", "B1", "B1", "B2", "B2", "B2")
))

run_pipeline <- function(samples_path, design) {
  outdir <- tempfile("out_", tmpdir = work)
  args <- c(pipeline,
            "--counts", counts_path, "--samples", samples_path,
            "--comparisons", comparisons_path, "--outdir", outdir,
            "--design", design, "--method", "all", "--threads", "1",
            "--min-genes", "10", "--min-reads", "10")
  log <- suppressWarnings(system2("Rscript", args, stdout = TRUE, stderr = TRUE))
  status <- attr(log, "status")
  manifest_path <- file.path(outdir, "pipeline_manifest.json")
  deg_path <- file.path(outdir, "comparisons", "Sertoli_vs_iso", "genolens_deg.csv")
  manifest <- if (file.exists(manifest_path)) fromJSON(manifest_path) else NULL
  list(
    status = if (is.null(status)) 0L else status,
    log = paste(log, collapse = "\n"),
    stats = manifest$comparison_stats$Sertoli_vs_iso,
    deg = if (file.exists(deg_path)) readLines(deg_path) else NULL
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
used <- function(res, design, df, fallback) {
  res$status == 0 && identical(res$stats$design_used, design) &&
    isTRUE(res$stats$residual_df == df) && identical(res$stats$design_fallback, fallback)
}
same_deg <- function(a, b) !is.null(a$deg) && identical(a$deg, b$deg)
# Each method swallows its own errors, so identical tables could just be two
# all-NA columns: make sure every method produced p-values.
all_methods_ran <- function(res) {
  if (is.null(res$deg)) return(FALSE)
  deg <- read.csv(text = res$deg, check.names = FALSE)
  all(vapply(c("DESeq2", "edgeR", "limma", "Stouffer"), function(m) {
    col <- paste0("padj.", m, ":Sertoli_vs_iso")
    col %in% names(deg) && any(!is.na(deg[[col]]))
  }, logical(1)))
}

# 1. Unbalanced donors: auto falls back to ~ condition, logs it, records why,
#    and produces exactly the ~ condition results.
res_auto <- run_pipeline(unbalanced_path, "auto")
res_cond <- run_pipeline(unbalanced_path, "condition")
check("auto falls back to ~ condition on an unbalanced donor design",
      used(res_auto, "~ condition", 4, TRUE) &&
        identical(res_auto$stats$design_fallback_reason, "too_few_residual_df") &&
        isTRUE(res_auto$stats$batch_residual_df == 1), res_auto)
check("the fallback is logged",
      grepl("falling back to ~ condition", res_auto$log, fixed = TRUE), res_auto)
check("the fallback gives the ~ condition results",
      same_deg(res_auto, res_cond) && all_methods_ran(res_auto), res_auto)

# 2. Explicit batch_condition is unchanged: batch is kept even with 1 residual df.
res <- run_pipeline(unbalanced_path, "batch_condition")
check("batch_condition keeps batch on an unbalanced donor design",
      used(res, "~ batch + condition", 1, FALSE), res)

# 3. Balanced paired donors: auto keeps batch and matches batch_condition, whose
#    results differ from ~ condition on this data.
res_auto <- run_pipeline(balanced_path, "auto")
res_batch <- run_pipeline(balanced_path, "batch_condition")
res_cond <- run_pipeline(balanced_path, "condition")
check("auto keeps batch on a balanced paired design",
      used(res_auto, "~ batch + condition", 2, FALSE) &&
        is.null(res_auto$stats$design_fallback_reason), res_auto)
check("auto gives the batch_condition results on a balanced paired design",
      same_deg(res_auto, res_batch) && !same_deg(res_auto, res_cond) &&
        all_methods_ran(res_auto), res_auto)

# 4. Explicit condition is unchanged: batch is never added.
check("condition ignores batch on a balanced paired design",
      used(res_cond, "~ condition", 4, FALSE), res_cond)

# 5. Batch confounded with condition: the batch design is not of full rank, so
#    auto falls back instead of letting every method fail.
res <- run_pipeline(confounded_path, "auto")
check("auto falls back when batch is confounded with condition",
      used(res, "~ condition", 4, TRUE) &&
        identical(res$stats$design_fallback_reason, "batch_confounded_with_condition") &&
        !is.null(res$deg), res)

unlink(work, recursive = TRUE)
if (failures > 0) {
  cat(failures, "check(s) failed\n")
  quit(status = 1)
}
cat("All auto-design checks passed\n")
