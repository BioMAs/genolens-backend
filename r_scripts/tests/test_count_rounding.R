#!/usr/bin/env Rscript
# =============================================================================
# Checks that run_multimethod_pipeline.R rounds non-integer counts.
#
# salmon/tximport estimates are not integers. Casting them straight to integer
# truncates (as.integer(2.9) == 2) and biases shallow samples downwards; the
# legacy pipe_scilicium step3 rounds them. The pipeline (DESeq2 only, to stay
# fast) runs on one fractional matrix, on its rounded copy and on its truncated
# copy: the fractional run must reproduce the rounded one.
#
#   Rscript r_scripts/tests/test_count_rounding.R
#
# Needs the pipeline's R packages (DESeq2, edgeR, limma, tidyverse, optparse,
# jsonlite), i.e. the r-worker image or a local R with them installed.
# Exits non-zero if any check fails.
# =============================================================================

suppressPackageStartupMessages(library(jsonlite))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", file_arg)))
pipeline <- normalizePath(file.path(here, "..", "run_multimethod_pipeline.R"))

set.seed(7)
work <- tempfile("rounding_")
dir.create(work)

# 8 samples, 300 low-count genes, KO over-expressing the first 40. Adding
# U(0, 1) to integer counts gives tximport-like estimates whose rounded and
# truncated versions differ for about half of the cells.
samples <- sprintf("S%d", 1:8)
genotype <- rep(c("KO", "WT"), each = 4)
base <- matrix(rnbinom(300 * 8, mu = 12, size = 5), nrow = 300,
               dimnames = list(sprintf("G%03d", 1:300), samples))
base[1:40, genotype == "KO"] <- base[1:40, genotype == "KO"] * 4L
fractional <- base + matrix(runif(length(base)), nrow = nrow(base), dimnames = dimnames(base))

write_counts <- function(name, mat) {
  path <- file.path(work, name)
  df <- data.frame(gene_id = rownames(mat), mat, check.names = FALSE)
  write.table(df, path, sep = "\t", quote = FALSE, row.names = FALSE)
  path
}
write_sheet <- function(name, df) {
  path <- file.path(work, name)
  write.table(df, path, sep = "\t", quote = FALSE, row.names = FALSE)
  path
}

samples_path <- write_sheet("samples.tsv", data.frame(sample_id = samples, condition = genotype))
comparisons_path <- write_sheet("comparisons.tsv", data.frame(
  comparison = "KO_vs_WT", condition1 = "KO", condition2 = "WT"
))

run_pipeline <- function(counts_path) {
  outdir <- tempfile("out_", tmpdir = work)
  args <- c(pipeline,
            "--counts", counts_path, "--samples", samples_path,
            "--comparisons", comparisons_path, "--outdir", outdir,
            "--method", "deseq2", "--threads", "1",
            "--min-genes", "10", "--min-reads", "10")
  log <- suppressWarnings(system2("Rscript", args, stdout = TRUE, stderr = TRUE))
  status <- attr(log, "status")
  manifest_path <- file.path(outdir, "pipeline_manifest.json")
  deg_path <- file.path(outdir, "comparisons", "KO_vs_WT", "genolens_deg.csv")
  vst_path <- file.path(outdir, "vst_counts.tsv")
  list(
    status = if (is.null(status)) 0L else status,
    log = paste(log, collapse = "\n"),
    manifest = if (file.exists(manifest_path)) fromJSON(manifest_path) else NULL,
    deg = if (file.exists(deg_path)) readLines(deg_path) else NULL,
    vst = if (file.exists(vst_path)) readLines(vst_path) else NULL
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
same_outputs <- function(a, b) {
  !is.null(a$deg) && identical(a$deg, b$deg) && identical(a$vst, b$vst) &&
    identical(a$manifest$qc_report$genes_after_filter, b$manifest$qc_report$genes_after_filter)
}

res_frac  <- run_pipeline(write_counts("fractional.tsv", fractional))
res_round <- run_pipeline(write_counts("rounded.tsv", round(fractional)))
res_trunc <- run_pipeline(write_counts("truncated.tsv", trunc(fractional)))

# 0. Sanity: rounding and truncating give different results on this matrix,
#    otherwise check 1 would prove nothing.
check("rounded and truncated inputs give different results",
      res_round$status == 0 && res_trunc$status == 0 && !same_outputs(res_round, res_trunc),
      res_trunc)

# 1. Fractional counts are rounded, not truncated: same DEG table, same VST
#    matrix and same number of genes kept as the pre-rounded input.
check("fractional counts are rounded to the nearest integer",
      res_frac$status == 0 && same_outputs(res_frac, res_round), res_frac)

# 2. The rounding is logged when, and only when, the counts are fractional.
check("rounding is logged for fractional counts",
      grepl("Counts are not integers", res_frac$log, fixed = TRUE), res_frac)
check("nothing is logged for integer counts",
      !grepl("Counts are not integers", res_round$log, fixed = TRUE), res_round)

unlink(work, recursive = TRUE)
if (failures > 0) {
  cat(failures, "check(s) failed\n")
  quit(status = 1)
}
cat("All count-rounding checks passed\n")
