#!/usr/bin/env Rscript
# =============================================================================
# GenoLens — annoDB Functional Enrichment (multi-organism, multi-database)
#
# Ports the SciLicium pipeline `fa.function` (hypergeometric over-representation
# against the local anno.db) into GenoLens. Runs enrichment for ALL annotation
# categories available for the species (GO BP/MF/CC, KEGG, Reactome, Hallmark,
# WikiPathways, …) on the significant DEGs of a single comparison, and writes a
# `genolens_enrichment.csv` whose columns are consumed by
# DataProcessorService.extract_enrichment_pathways_for_db (tasks.py ingestion).
#
# Reference: pipe_scilicium/workflow/scripts/r/functional_enrichment_legacy.R
# =============================================================================

suppressPackageStartupMessages({
  library(optparse)
  library(tidyverse)
  library(Matrix)
  library(data.table)
})

option_list <- list(
  make_option("--deg",         type = "character", default = NULL, help = "Per-comparison genolens_deg.csv"),
  make_option("--gene-list",   type = "character", default = NULL,
              help = "Plain gene-symbol list (one per line). When set, enrich exactly this set and ignore --deg / thresholds (ad-hoc intersection enrichment)."),
  make_option("--anno-db-dir", type = "character", help = "Directory holding anno.db.<species>.gene_symbol.*.rds"),
  make_option("--species",     type = "character", default = "human", help = "Species (human, mouse, rat, zebrafish, medaka)"),
  make_option("--comparison",  type = "character", help = "Comparison name (used for the gene.cluster column)"),
  make_option("--output",      type = "character", help = "Output genolens_enrichment.csv path"),
  make_option("--fdr",         type = "double",    default = 0.05, help = "DEG padj threshold for significance"),
  make_option("--min-log2fc",  type = "double",    default = 1.0,  help = "DEG |log2FC| threshold for significance"),
  make_option("--p-cutoff",    type = "double",    default = 0.05, help = "Enrichment p-value cutoff"),
  make_option("--padj-cutoff", type = "double",    default = 0.05, help = "Adjusted p-value (BH) cutoff for output"),
  make_option("--r-cutoff",    type = "integer",   default = 3L,   help = "Minimum query genes per term")
)
opt <- parse_args(OptionParser(option_list = option_list))
stopifnot(!is.null(opt$`anno-db-dir`), !is.null(opt$output), !is.null(opt$comparison))
if (is.null(opt$deg) && is.null(opt$`gene-list`)) stop("Provide either --deg or --gene-list")

message("=== GenoLens annoDB Enrichment ===")
message("DEG file: ", opt$deg)
message("anno.db dir: ", opt$`anno-db-dir`, "  |  species: ", opt$species, "  |  comparison: ", opt$comparison)

# -----------------------------------------------------------------------------
# Map the species label → anno.db file species token
# -----------------------------------------------------------------------------
species_mapping <- list(
  "hg38" = "human", "hg19" = "human", "human" = "human", "homo_sapiens" = "human",
  "mm10" = "mouse", "mm39" = "mouse", "mouse" = "mouse", "mus_musculus" = "mouse",
  "rn6" = "rat", "rn7" = "rat", "rat" = "rat",
  "medaka" = "medaka",
  "drerio" = "zebrafish", "zebrafish" = "zebrafish"
)
db_species <- species_mapping[[tolower(opt$species)]]
if (is.null(db_species)) stop("Unsupported species: ", opt$species)

# Map anno.db internal category names → the values used by the GenoLens frontend
# selector. Unknown categories fall through unchanged (the UI derives chips
# dynamically from the results, so they still render).
map_category <- function(cat) {
  key <- tolower(cat)
  dplyr::case_when(
    key %in% c("biological_process", "go_bp", "go:bp")            ~ "GO:BP",
    key %in% c("molecular_function", "go_mf", "go:mf")            ~ "GO:MF",
    key %in% c("cellular_component", "go_cc", "go:cc")            ~ "GO:CC",
    grepl("kegg", key)                                            ~ "KEGG",
    grepl("reactome", key)                                        ~ "REACTOME",
    grepl("hallmark", key)                                        ~ "HALLMARK",
    grepl("wikipathway", key)                                     ~ "WIKIPATHWAYS",
    TRUE                                                          ~ cat
  )
}

# -----------------------------------------------------------------------------
# fa.function — hypergeometric ORA against an anno.db category matrix
# (ported verbatim in spirit from functional_enrichment_legacy.R)
# -----------------------------------------------------------------------------
fa.function <- function(group.ids, anno.db.m, all.ids = NULL, group.name = "gene.cluster",
                        category = "category", p.cutoff = 0.05, r.cutoff = 3,
                        description_map = NULL) {
  if (ncol(anno.db.m) <= 0) return(NULL)

  N.IDs <- unique(names(which(Matrix::rowSums(anno.db.m) > 0)))
  if (!is.null(all.ids)) N.IDs <- intersect(N.IDs, all.ids)
  R.IDs <- intersect(unique(group.ids), N.IDs)

  if (length(R.IDs) <= 0) {
    r.IDs <- lapply(colnames(anno.db.m), function(x) NULL)
    names(r.IDs) <- colnames(anno.db.m)
  } else {
    r.IDs <- apply(anno.db.m[R.IDs, , drop = FALSE], 2, function(x) unique(R.IDs[which(x > 0)]))
  }

  df <- data.frame(term = colnames(anno.db.m), stringsAsFactors = FALSE)
  # Normalise the description map to a named character vector (it may arrive as a
  # data.frame [id, label] or a list in the anno.db object).
  if (is.data.frame(description_map) && ncol(description_map) >= 2) {
    description_map <- stats::setNames(as.character(description_map[[2]]), as.character(description_map[[1]]))
  } else if (is.list(description_map) && !is.null(names(description_map))) {
    description_map <- unlist(description_map, use.names = TRUE)
  }
  df$description <- if (is.atomic(description_map) && !is.null(names(description_map))) {
    lbl <- unname(description_map[df$term])
    ifelse(is.na(lbl) | lbl == "", df$term, lbl)
  } else df$term
  df$category     <- category
  df$gene.cluster <- group.name
  df$genes <- unlist(lapply(r.IDs, paste, collapse = "|"))
  df$r <- unlist(lapply(r.IDs, length))
  df$R <- length(R.IDs)
  df$n <- Matrix::colSums(anno.db.m[N.IDs, , drop = FALSE], na.rm = TRUE)
  df$N <- length(N.IDs)

  df$rExpected <- round(df$n * df$R / df$N, 0)
  sd <- (df$rExpected * (1 - df$R / df$N) * (1 - (df$n - 1) / (df$N - 1)))^0.5
  df$zscore <- (df$r - df$rExpected) / sd
  df$p.hyper.enri <- phyper(df$r - 1, df$n, df$N - df$n, df$R, lower.tail = FALSE)

  df$info <- NA_character_
  df$pvalue <- NA_real_
  enri.idx <- which(df$p.hyper.enri < p.cutoff & df$r >= r.cutoff)
  if (length(enri.idx) > 0) {
    df$info[enri.idx]   <- "enriched"
    df$pvalue[enri.idx] <- df$p.hyper.enri[enri.idx]
  }
  df$p.adjust <- if (length(enri.idx) > 0) p.adjust(df$p.hyper.enri, method = "BH") else rep(1, nrow(df))
  df[order(df$pvalue, na.last = TRUE), ]
}

# -----------------------------------------------------------------------------
# Load the anno.db object for this species (newest file wins)
# -----------------------------------------------------------------------------
anno_db_files <- list.files(
  opt$`anno-db-dir`,
  pattern = sprintf("anno.db.%s.gene_symbol.*\\.rds$", db_species),
  full.names = TRUE
)
if (length(anno_db_files) == 0) {
  message("WARNING: no anno.db file for species '", db_species, "' in ", opt$`anno-db-dir`, " — writing empty output")
  readr::write_csv(tibble(term = character(), Description = character(), category = character(),
                          pvalue = double(), p.adjust = double(), genes = character(),
                          Count = integer(), GeneRatio = character(), BgRatio = character(),
                          gene.cluster = character()), opt$output)
  quit(save = "no", status = 0)
}
anno.db.obj <- readRDS(sort(anno_db_files, decreasing = TRUE)[1])
description_sources <- anno.db.obj$descriptions
message("Loaded anno.db with ", length(names(anno.db.obj$annotations)), " categories")

# -----------------------------------------------------------------------------
# Read DEG results and select significant genes for this single comparison
# -----------------------------------------------------------------------------
if (!is.null(opt$`gene-list`)) {
  # Ad-hoc mode: enrich exactly the provided gene-symbol set (e.g. a Venn
  # intersection). No DEG file, no thresholds, no up/down split.
  genes <- readLines(opt$`gene-list`, warn = FALSE)
  genes <- unique(trimws(genes))
  genes <- genes[nzchar(genes)]
  message("Gene-list mode: ", length(genes), " genes")
  list.groups   <- list(all = genes)
  cluster_label <- list(all = opt$comparison)
} else {
  deg <- readr::read_csv(opt$deg, show_col_types = FALSE)

  # The per-comparison genolens_deg.csv has dynamic, comparison-suffixed columns.
  logfc_col <- grep("^log2FoldChange|^logFC", names(deg), value = TRUE)[1]
  padj_priority <- c("^padj\\.Stouffer", "^padj\\.DESeq2", "^padj\\.edgeR", "^padj\\.limma", "^padj")
  padj_col <- NA_character_
  for (pat in padj_priority) {
    hit <- grep(pat, names(deg), value = TRUE)
    if (length(hit) > 0) { padj_col <- hit[1]; break }
  }
  if (is.na(logfc_col) || is.na(padj_col) || !("gene_name" %in% names(deg))) {
    stop("DEG file missing logFC / padj / gene_name columns")
  }

  deg <- deg %>%
    mutate(
      .lfc  = suppressWarnings(as.numeric(.data[[logfc_col]])),
      .padj = suppressWarnings(as.numeric(.data[[padj_col]]))
    ) %>%
    filter(!is.na(gene_name), !is.na(.padj), !is.na(.lfc),
           .padj < opt$fdr, abs(.lfc) >= opt$`min-log2fc`)

  message("Significant DEGs: ", nrow(deg))

  list.groups <- list(
    all  = unique(deg$gene_name),
    up   = unique(deg$gene_name[deg$.lfc > 0]),
    down = unique(deg$gene_name[deg$.lfc < 0])
  )
  # gene.cluster suffix understood by extract_enrichment_pathways_for_db
  cluster_label <- list(all = opt$comparison,
                        up = paste0(opt$comparison, " (up)"),
                        down = paste0(opt$comparison, " (down)"))
}

all_ids <- unique(unlist(lapply(anno.db.obj$annotations, function(x) {
  if (inherits(x, "Matrix") || is.matrix(x)) rownames(x) else NULL
})))

categories <- names(anno.db.obj$annotations)
categories <- categories[!grepl("^NCBI_|^chromosome_location$|^xrefs$|^interactions$|^regulations$", categories)]

# -----------------------------------------------------------------------------
# Run enrichment for every category × {all, up, down}
# -----------------------------------------------------------------------------
rows <- list()
for (grp in names(list.groups)) {
  ids <- list.groups[[grp]]
  if (length(ids) == 0) next
  for (category in categories) {
    m <- anno.db.obj$annotations[[category]]
    if (is.null(m) || ncol(m) <= 0) next
    res <- fa.function(ids, m, all_ids, cluster_label[[grp]], category,
                       opt$`p-cutoff`, opt$`r-cutoff`, description_sources[[category]])
    if (is.null(res)) next
    res <- res[!is.na(res$info) & !is.na(res$p.adjust) & res$p.adjust < opt$`padj-cutoff`, , drop = FALSE]
    if (nrow(res) > 0) rows[[length(rows) + 1]] <- res
  }
}

if (length(rows) == 0) {
  message("No enriched terms — writing empty output")
  out <- tibble(term = character(), Description = character(), category = character(),
                pvalue = double(), p.adjust = double(), genes = character(),
                Count = integer(), GeneRatio = character(), BgRatio = character(),
                gene.cluster = character())
} else {
  combined <- as.data.frame(data.table::rbindlist(rows, fill = TRUE))
  out <- tibble(
    term         = combined$term,
    Description  = combined$description,
    category     = map_category(combined$category),
    pvalue       = combined$pvalue,
    p.adjust     = combined$p.adjust,
    genes        = substr(combined$genes, 1, 32000),
    Count        = combined$r,
    GeneRatio    = paste0(combined$r, "/", combined$R),
    BgRatio      = paste0(combined$n, "/", combined$N),
    gene.cluster = combined$gene.cluster
  )
}

dir.create(dirname(opt$output), recursive = TRUE, showWarnings = FALSE)
readr::write_csv(out, opt$output)
message("Wrote ", nrow(out), " enriched terms → ", opt$output)
