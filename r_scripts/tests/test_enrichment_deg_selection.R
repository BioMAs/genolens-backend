#!/usr/bin/env Rscript
# =============================================================================
# Checks which padj column functional_enrichment.R uses to pick the DEGs it
# enriches from a genolens_deg.csv: the method that decided in
# run_multimethod_pipeline.R (DESeq2, then edgeR, then limma), never the
# Stouffer combination while a per-method column has results.
#
# Uses a tiny fake anno.db; only the DEG selection is asserted.
#
#   Rscript r_scripts/tests/test_enrichment_deg_selection.R
#
# Needs the script's R packages (optparse, tidyverse, Matrix, data.table),
# i.e. the r-worker image. Exits non-zero if any check fails.
# =============================================================================

suppressPackageStartupMessages(library(Matrix))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", file_arg)))
script <- normalizePath(file.path(here, "..", "functional_enrichment.R"))

work <- tempfile("enrsel_")
dir.create(work)

genes <- sprintf("GENE%02d", 1:40)
anno <- sparseMatrix(i = 1:20, j = rep(1:2, each = 10), x = 1, dims = c(40, 2),
                     dimnames = list(genes, c("TERM_A", "TERM_B")))
saveRDS(list(annotations = list(GO_BP = anno),
             descriptions = list(GO_BP = data.frame(term = c("TERM_A", "TERM_B"),
                                                    description = c("Term A", "Term B")))),
        file.path(work, "anno.db.human.gene_symbol.test.rds"))

# Significant at FDR 0.05 with |log2FC| >= 1: DESeq2 5 genes, edgeR 7, limma 3,
# Stouffer 12.
sig_first <- function(n) ifelse(seq_along(genes) <= n, 0.01, 0.5)
base_deg <- data.frame(
  gene_id = genes, gene_name = genes,
  `logFC:C` = c(rep(2, 20), rep(-2, 10), rep(0.1, 10)),
  `padj.DESeq2:C` = sig_first(5),
  `padj.edgeR:C` = sig_first(7),
  `padj.limma:C` = sig_first(3),
  `padj.Stouffer:C` = sig_first(12),
  check.names = FALSE
)

run_enrichment <- function(deg) {
  deg_path <- tempfile("deg_", tmpdir = work, fileext = ".csv")
  write.csv(deg, deg_path, row.names = FALSE, na = "NA")
  args <- c(script, "--deg", deg_path, "--anno-db-dir", work, "--species", "human",
            "--comparison", "C", "--output", tempfile("enr_", tmpdir = work, fileext = ".csv"),
            "--fdr", "0.05", "--min-log2fc", "1")
  paste(suppressWarnings(system2("Rscript", args, stdout = TRUE, stderr = TRUE)), collapse = "\n")
}

failures <- 0L
check <- function(label, log, padj_col, n_deg) {
  ok <- grepl(paste0("DEG padj column: ", padj_col), log, fixed = TRUE) &&
    grepl(paste0("Significant DEGs: ", n_deg), log, fixed = TRUE)
  if (ok) {
    cat("PASS ", label, "\n", sep = "")
  } else {
    failures <<- failures + 1L
    cat("FAIL ", label, "\n--- script output ---\n", log, "\n---------------------\n", sep = "")
  }
}

check("DESeq2 decides when present", run_enrichment(base_deg), "padj.DESeq2:C", 5)

no_deseq <- base_deg
no_deseq$`padj.DESeq2:C` <- NA
check("edgeR decides when DESeq2 gave no result", run_enrichment(no_deseq), "padj.edgeR:C", 7)

stouffer_only <- base_deg[, c("gene_id", "gene_name", "logFC:C", "padj.Stouffer:C")]
check("a Stouffer-only table still enriches on Stouffer", run_enrichment(stouffer_only),
      "padj.Stouffer:C", 12)

unlink(work, recursive = TRUE)
if (failures > 0) {
  cat(failures, "check(s) failed\n")
  quit(status = 1)
}
cat("All enrichment DEG-selection checks passed\n")
