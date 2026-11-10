#!/usr/bin/env Rscript
# =============================================================================
# Checks what the R scripts record about their own run, for the LaTeX report to
# print as observed (never as a default):
#
#   run_multimethod_pipeline.R -> pipeline_manifest.json
#     software_versions   R, DESeq2, edgeR and limma, as installed where it ran
#     design_used         per comparison (comparison_stats)
#     qc_relaxed          TRUE when fewer than 2 samples passed the thresholds
#   functional_enrichment.R --meta-out <json>
#     annodb_file         the anno.db file actually loaded (the newest)
#     categories          the anno.db categories enriched (after --databases)
#     background          what the hypergeometric background is
#
# Runs the real scripts on a small synthetic dataset (DESeq2 only, to stay fast).
#
#   Rscript r_scripts/tests/test_manifest_provenance.R
#
# Needs the pipeline's R packages (DESeq2, edgeR, limma, tidyverse, optparse,
# jsonlite, Matrix, data.table), i.e. the r-worker image. Exits non-zero on the
# first failed check.
# =============================================================================

suppressPackageStartupMessages({
  library(jsonlite)
  library(Matrix)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", file_arg)))
pipeline <- normalizePath(file.path(here, "..", "run_multimethod_pipeline.R"))
enrichment <- normalizePath(file.path(here, "..", "functional_enrichment.R"))

set.seed(7)
work <- tempfile("provenance_")
dir.create(work)

failures <- 0L
check <- function(label, ok, detail = "") {
  if (isTRUE(ok)) {
    cat("PASS ", label, "\n", sep = "")
  } else {
    failures <<- failures + 1L
    cat("FAIL ", label, "\n", detail, "\n", sep = "")
  }
}

# -----------------------------------------------------------------------------
# The pipeline manifest
# -----------------------------------------------------------------------------

# 8 samples, 300 genes. KO samples over-express the first 40 genes.
samples <- sprintf("S%d", 1:8)
genotype <- rep(c("KO", "WT"), each = 4)
counts <- matrix(rnbinom(300 * 8, mu = 200, size = 10), nrow = 300,
                 dimnames = list(sprintf("G%03d", 1:300), samples))
counts[1:40, genotype == "KO"] <- counts[1:40, genotype == "KO"] * 6L
counts_path <- file.path(work, "counts.tsv")
write.table(data.frame(gene_id = rownames(counts), counts, check.names = FALSE),
            counts_path, sep = "\t", quote = FALSE, row.names = FALSE)

write_sheet <- function(name, df) {
  path <- file.path(work, name)
  write.table(df, path, sep = "\t", quote = FALSE, row.names = FALSE)
  path
}
comparisons_path <- write_sheet("comparisons.tsv", data.frame(
  comparison = "KO_vs_WT", condition1 = "KO", condition2 = "WT"
))
plain_samples <- write_sheet("samples.tsv", data.frame(sample_id = samples, condition = genotype))
batch_samples <- write_sheet("samples_batch.tsv", data.frame(
  sample_id = samples, condition = genotype, batch = rep(c("b1", "b2"), 4)
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

res <- run_pipeline(plain_samples)
m <- res$manifest
versions <- m$software_versions
installed <- list(
  R = paste(R.version$major, R.version$minor, sep = "."),
  DESeq2 = as.character(packageVersion("DESeq2")),
  edgeR = as.character(packageVersion("edgeR")),
  limma = as.character(packageVersion("limma"))
)
check("the manifest records the versions of R, DESeq2, edgeR and limma as installed",
      res$status == 0 && !is.null(versions) &&
        all(vapply(names(installed), function(k) identical(versions[[k]], installed[[k]]), logical(1))),
      paste(c(res$log, toJSON(versions, auto_unbox = TRUE)), collapse = "\n"))
check("every recorded version is a plain string",
      !is.null(versions) && all(vapply(versions, function(v) is.character(v) && length(v) == 1, logical(1))))
check("the design each comparison was fitted with is recorded",
      identical(m$comparison_stats$KO_vs_WT$design_used, "~ condition"), res$log)
check("the sample thresholds were not relaxed when two samples or more passed",
      identical(m$qc_relaxed, FALSE) && identical(m$qc_report$samples_passed, 8L), res$log)

res_batch <- run_pipeline(batch_samples, c("--design", "batch_condition"))
check("a batch design is recorded as the one fitted",
      identical(res_batch$manifest$comparison_stats$KO_vs_WT$design_used, "~ batch + condition"),
      res_batch$log)

# Every sample is below the read threshold: the script keeps the two largest libraries.
res_relaxed <- run_pipeline(plain_samples, c("--min-reads", "100000000"))
check("relaxed sample thresholds are recorded",
      res_relaxed$status == 0 && identical(res_relaxed$manifest$qc_relaxed, TRUE) &&
        identical(res_relaxed$manifest$qc_report$samples_passed, 2L), res_relaxed$log)

# -----------------------------------------------------------------------------
# The enrichment run record
# -----------------------------------------------------------------------------

genes <- sprintf("GENE%02d", 1:40)
# Terms of different sizes, as real gene sets are. (The script's apply() call used to return a
# matrix for equally sized sets; it no longer does, and test_enrichment_one_term_category.R
# is the test of that case.)
term_matrix <- function(sizes, start, terms) {
  rows <- unlist(Map(function(n, from) seq(from, length.out = n), sizes, start))
  sparseMatrix(i = rows, j = rep(seq_along(sizes), sizes), x = 1, dims = c(40, length(sizes)),
               dimnames = list(genes, terms))
}
anno <- term_matrix(c(12, 8), c(1, 13), c("TERM_A", "TERM_B"))
anno_kegg <- term_matrix(c(16, 6), c(5, 21), c("PATH_A", "PATH_B"))
describe <- function(ids) data.frame(term = ids, description = paste("Term", ids))
db <- list(annotations = list(GO_BP = anno, kegg_pathway = anno_kegg),
           descriptions = list(GO_BP = describe(c("TERM_A", "TERM_B")),
                               kegg_pathway = describe(c("PATH_A", "PATH_B"))))
anno_dir <- file.path(work, "anno")
dir.create(anno_dir)
saveRDS(db, file.path(anno_dir, "anno.db.human.gene_symbol.20250101.rds"))
saveRDS(db, file.path(anno_dir, "anno.db.human.gene_symbol.20260101.rds"))

deg <- data.frame(
  gene_id = genes, gene_name = genes,
  `logFC:C` = c(rep(2, 20), rep(-2, 10), rep(0.1, 10)),
  `padj.DESeq2:C` = ifelse(seq_along(genes) <= 30, 0.001, 0.9),
  check.names = FALSE
)
deg_path <- file.path(work, "deg.csv")
write.csv(deg, deg_path, row.names = FALSE)

run_enrichment <- function(extra = character()) {
  meta_path <- tempfile("enrichment_meta_", tmpdir = work, fileext = ".json")
  args <- c(enrichment, "--deg", deg_path, "--anno-db-dir", anno_dir, "--species", "human",
            "--comparison", "C", "--output", tempfile("enr_", tmpdir = work, fileext = ".csv"),
            "--fdr", "0.05", "--min-log2fc", "1", "--meta-out", meta_path, extra)
  log <- suppressWarnings(system2("Rscript", args, stdout = TRUE, stderr = TRUE))
  status <- attr(log, "status")
  list(
    status = if (is.null(status)) 0L else status,
    log = paste(log, collapse = "\n"),
    meta = if (file.exists(meta_path)) fromJSON(meta_path) else NULL,
    raw = if (file.exists(meta_path)) paste(readLines(meta_path, warn = FALSE), collapse = "") else ""
  )
}

res <- run_enrichment()
check("the enrichment records the anno.db file it loaded (the newest)",
      res$status == 0 && identical(res$meta$annodb_file, "anno.db.human.gene_symbol.20260101.rds"),
      res$log)
check("the enrichment records the categories it ran over",
      identical(sort(res$meta$categories), c("GO_BP", "kegg_pathway")), res$log)
check("the enrichment records its background as the genes annotated in the category",
      identical(res$meta$background, "genes annotated to at least one term of the category tested"),
      res$log)

res <- run_enrichment(c("--databases", "kegg_pathway"))
check("a database pick narrows the recorded categories",
      res$status == 0 && identical(as.character(res$meta$categories), "kegg_pathway"), res$log)
check("a single category stays a list in the record",
      grepl("\"categories\":\\s*\\[\\s*\"kegg_pathway\"\\s*\\]", res$raw), res$raw)

unlink(work, recursive = TRUE)
if (failures > 0) {
  cat(failures, "check(s) failed\n")
  quit(status = 1)
}
cat("All manifest-provenance checks passed\n")
