#!/usr/bin/env Rscript
# =============================================================================
# Checks how run_multimethod_pipeline.R picks the condition column.
#
# Runs the real pipeline (DESeq2 only, to stay fast) on a small synthetic
# dataset and asserts which column ends up in pipeline_manifest.json.
#
#   Rscript r_scripts/tests/test_condition_column.R
#
# Needs the pipeline's R packages (DESeq2, edgeR, limma, tidyverse, optparse,
# jsonlite), i.e. the r-worker image or a local R with them installed.
# Exits non-zero on the first failed check.
# =============================================================================

suppressPackageStartupMessages(library(jsonlite))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", file_arg)))
pipeline <- normalizePath(file.path(here, "..", "run_multimethod_pipeline.R"))

set.seed(42)
work <- tempfile("condcol_")
dir.create(work)

# 8 samples, 300 genes. KO samples over-express the first 40 genes, so a
# KO-vs-WT comparison has real signal.
samples <- sprintf("S%d", 1:8)
genotype <- rep(c("KO", "WT"), each = 4)
counts <- matrix(rnbinom(300 * 8, mu = 200, size = 10), nrow = 300,
                 dimnames = list(sprintf("G%03d", 1:300), samples))
counts[1:40, genotype == "KO"] <- counts[1:40, genotype == "KO"] * 6L
counts_df <- data.frame(gene_id = rownames(counts), counts, check.names = FALSE)
counts_path <- file.path(work, "counts.tsv")
write.table(counts_df, counts_path, sep = "\t", quote = FALSE, row.names = FALSE)

write_sheet <- function(name, df) {
  path <- file.path(work, name)
  write.table(df, path, sep = "\t", quote = FALSE, row.names = FALSE)
  path
}

comparisons_path <- write_sheet("comparisons.tsv", data.frame(
  comparison = "KO_vs_WT", condition1 = "KO", condition2 = "WT"
))

run_pipeline <- function(samples_path, extra = character()) {
  outdir <- tempfile("out_", tmpdir = work)
  args <- c(pipeline,
            "--counts", counts_path, "--samples", samples_path,
            "--comparisons", comparisons_path, "--outdir", outdir,
            "--method", "deseq2", "--threads", "1",
            "--min-genes", "10", "--min-reads", "10", extra)
  log <- suppressWarnings(system2("Rscript", args, stdout = TRUE, stderr = TRUE))
  status <- attr(log, "status")
  manifest_path <- file.path(outdir, "pipeline_manifest.json")
  list(
    status = if (is.null(status)) 0L else status,
    log = paste(log, collapse = "\n"),
    manifest = if (file.exists(manifest_path)) fromJSON(manifest_path) else NULL
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
ran_comparison <- function(res) {
  stats <- res$manifest$comparison_stats$KO_vs_WT
  !is.null(stats) && !isTRUE(stats$skipped)
}

# 1. French alias `groupe`, no flag: auto-detected (was rejected before).
res <- run_pipeline(write_sheet("groupe.tsv", data.frame(sample_id = samples, groupe = genotype)))
check("'groupe' is auto-detected",
      res$status == 0 && identical(res$manifest$qc_report$condition_column, "groupe") &&
        ran_comparison(res), res)

# A sheet where an alias column (`condition`) exists but is NOT the grouping
# the comparisons were built on — the wizard user picked Genotype_Detail.
decoy_path <- write_sheet("decoy.tsv", data.frame(
  Sample_ID = samples,
  condition = rep(c("day1", "day2"), times = 4),
  Genotype_Detail = genotype
))

# 2. --condition-col wins over the alias column (case-insensitive).
res <- run_pipeline(decoy_path, "--condition-col=Genotype_Detail")
check("--condition-col overrides the alias column",
      res$status == 0 && identical(res$manifest$qc_report$condition_column, "genotype_detail") &&
        ran_comparison(res), res)

# 3. Without the flag, the old behaviour is kept: the alias column is used, so
#    the KO/WT comparison finds no samples and is skipped.
res <- run_pipeline(decoy_path)
check("no flag keeps alias detection",
      res$status == 0 && identical(res$manifest$qc_report$condition_column, "condition") &&
        !ran_comparison(res), res)

# 4. A requested column that is absent fails loudly instead of falling back.
res <- run_pipeline(decoy_path, "--condition-col=treatment_group")
check("missing --condition-col column is an explicit error",
      res$status != 0 && grepl("Condition column 'treatment_group' was not found", res$log), res)

# 5. No alias and no flag: the error names the columns and the flag.
res <- run_pipeline(write_sheet("noalias.tsv", data.frame(sample = samples, genotype_detail = genotype)))
check("no alias column and no flag is an explicit error",
      res$status != 0 && grepl("--condition-col", res$log, fixed = TRUE) &&
        grepl("genotype_detail", res$log, fixed = TRUE), res)

unlink(work, recursive = TRUE)
if (failures > 0) {
  cat(failures, "check(s) failed\n")
  quit(status = 1)
}
cat("All condition-column checks passed\n")
