#!/usr/bin/env Rscript
# Install R packages required for GenoLens self-service analysis.
# Run once during Docker image build.
# Exits with code 1 if any package cannot be loaded after installation.

options(repos = c(CRAN = "https://cloud.r-project.org"))
options(Ncpus = 4)
# Treat warnings as errors so BiocManager failures surface immediately
options(warn = 2)

message("=== Installing CRAN packages ===")
install.packages(c("optparse", "tidyverse", "BiocManager", "Matrix", "data.table", "uwot"), quiet = FALSE)

message("=== Installing Bioconductor packages ===")
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}
BiocManager::install(
  c("DESeq2", "BiocParallel", "S4Vectors", "SummarizedExperiment", "edgeR", "limma", "apeglm"),
  ask    = FALSE,
  update = FALSE,
  quiet  = FALSE
)

message("=== Verifying package availability ===")
required_packages <- c("DESeq2", "BiocParallel", "S4Vectors", "SummarizedExperiment",
                        "edgeR", "limma", "apeglm", "uwot",
                        "optparse", "tidyverse", "Matrix", "data.table", "jsonlite")
failed <- character(0)
for (pkg in required_packages) {
  tryCatch(
    library(pkg, character.only = TRUE),
    error = function(e) {
      failed <<- c(failed, pkg)
      message("MISSING: ", pkg, " — ", conditionMessage(e))
    }
  )
}
if (length(failed) > 0) {
  message("=== FATAL: the following packages could not be loaded: ",
          paste(failed, collapse = ", "), " ===")
  quit(status = 1)
}

message("=== All packages installed and verified successfully ===")
