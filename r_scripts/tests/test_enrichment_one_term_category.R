#!/usr/bin/env Rscript
# =============================================================================
# Checks that functional_enrichment.R keeps one gene list per term when an
# anno.db category has a single term, or terms that all hit the same number of
# DEGs. fa.function used to build its per-term gene lists with apply(), which
# collapses equal-length results into a matrix; the genes / r columns then no
# longer had one entry per term and the script stopped.
#
# Uses a tiny fake anno.db with three categories:
#   GO_BP     one term of 10 genes, 4 of them DEGs
#   KEGG      two terms of 5 genes each, 3 DEGs in each (equal-length hits)
#   REACTOME  two terms that no DEG touches
#
#   Rscript r_scripts/tests/test_enrichment_one_term_category.R
#
# Needs the script's R packages (optparse, tidyverse, Matrix, data.table),
# i.e. the r-worker image. Exits non-zero if any check fails.
# =============================================================================

suppressPackageStartupMessages(library(Matrix))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", file_arg)))
script <- normalizePath(file.path(here, "..", "functional_enrichment.R"))

work <- tempfile("enrone_")
dir.create(work)

genes <- sprintf("GENE%02d", 1:40)

# One category = one sparse genes x terms matrix; term_genes maps each term id to
# the genes it annotates.
make_category <- function(term_genes) {
  sparseMatrix(i = match(unlist(term_genes), genes),
               j = rep(seq_along(term_genes), lengths(term_genes)), x = 1,
               dims = c(length(genes), length(term_genes)),
               dimnames = list(genes, names(term_genes)))
}
describe <- function(term_genes) {
  data.frame(term = names(term_genes), description = paste("Term", names(term_genes)))
}

categories <- list(
  GO_BP    = list(TERM_SOLO = genes[1:10]),
  KEGG     = list(TERM_K1 = genes[11:15], TERM_K2 = genes[16:20]),
  REACTOME = list(TERM_R1 = genes[31:35], TERM_R2 = genes[36:40])
)
saveRDS(list(annotations = lapply(categories, make_category),
             descriptions = lapply(categories, describe)),
        file.path(work, "anno.db.human.gene_symbol.test.rds"))

# Ten DEGs (FDR 0.05, |log2FC| >= 1): GENE01-04 and GENE11-13 up, GENE16-18 down.
deg_idx <- c(1:4, 11:13, 16:18)
deg <- data.frame(
  gene_id = genes, gene_name = genes,
  `logFC:C` = ifelse(seq_along(genes) %in% 16:18, -2, ifelse(seq_along(genes) %in% deg_idx, 2, 0.1)),
  `padj.DESeq2:C` = ifelse(seq_along(genes) %in% deg_idx, 0.01, 0.5),
  check.names = FALSE
)

# A category's background is the genes annotated to its own terms, so a one-term
# category can never reach significance (p = 1). The cutoffs are opened so that
# every term with at least one DEG is reported and its gene list can be read:
# 1.1 and not 1, because the script keeps a term only when its p-value and its
# adjusted p-value are strictly below the cutoff (`<`), and a p-value of 1 is
# not below 1.
deg_path <- file.path(work, "deg.csv")
out_path <- file.path(work, "enrichment.csv")
write.csv(deg, deg_path, row.names = FALSE, na = "NA")
args <- c(script, "--deg", deg_path, "--anno-db-dir", work, "--species", "human",
          "--comparison", "C", "--output", out_path, "--fdr", "0.05", "--min-log2fc", "1",
          "--p-cutoff", "1.1", "--padj-cutoff", "1.1", "--r-cutoff", "1")
log <- suppressWarnings(system2("Rscript", args, stdout = TRUE, stderr = TRUE))
status <- attr(log, "status")
if (is.null(status)) status <- 0L
log <- paste(log, collapse = "\n")

failures <- 0L
check <- function(label, ok) {
  if (isTRUE(ok)) {
    cat("PASS ", label, "\n", sep = "")
  } else {
    failures <<- failures + 1L
    cat("FAIL ", label, "\n", sep = "")
  }
}

check("the script finishes with exit code 0", status == 0L)
if (status != 0L || !file.exists(out_path)) {
  cat("--- script output ---\n", log, "\n---------------------\n", sep = "")
  cat("no output to read, the row checks are skipped\n")
  unlink(work, recursive = TRUE)
  cat(failures, "check(s) failed\n")
  quit(status = 1)
}

res <- read.csv(out_path, stringsAsFactors = FALSE)
rows_of <- function(term, cluster) res[res$term == term & res$gene.cluster == cluster, , drop = FALSE]
check_term <- function(label, term, cluster, expected) {
  row <- rows_of(term, cluster)
  ok <- nrow(row) == 1L && row$Count == length(expected) &&
    identical(sort(strsplit(row$genes, "|", fixed = TRUE)[[1]]), sort(expected))
  check(label, ok)
}
check_absent <- function(label, terms, cluster) {
  check(label, all(vapply(terms, function(t) nrow(rows_of(t, cluster)) == 0L, logical(1))))
}

check_term("one-term category: r counts the DEGs of the term, genes lists them", "TERM_SOLO", "C",
           genes[1:4])
check_term("KEGG term 1 keeps its own genes", "TERM_K1", "C", genes[11:13])
check_term("KEGG term 2 keeps its own genes", "TERM_K2", "C", genes[16:18])
check_absent("a category no DEG touches adds no row", c("TERM_R1", "TERM_R2"), "C")

check_term("up group: one-term category keeps its gene list", "TERM_SOLO", "C (up)", genes[1:4])
check_term("up group: KEGG term 1 keeps its own genes", "TERM_K1", "C (up)", genes[11:13])
check_absent("up group: a term with no up DEG adds no row", "TERM_K2", "C (up)")
check_term("down group: KEGG term 2 keeps its own genes", "TERM_K2", "C (down)", genes[16:18])
check_absent("down group: the categories without a down DEG add no row",
             c("TERM_SOLO", "TERM_K1", "TERM_R1", "TERM_R2"), "C (down)")

unlink(work, recursive = TRUE)
if (failures > 0) {
  cat(failures, "check(s) failed\n")
  quit(status = 1)
}
cat("All enrichment one-term-category checks passed\n")
