# Multi-stage Dockerfile for GenoLens Next
# Builds both API and Worker targets

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# ================================
# API Target
# ================================
FROM base as api

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ================================
# Worker Target
# ================================
FROM base as worker

# Run Celery worker
CMD ["celery", "-A", "app.worker.celery_app", "worker", "--loglevel=info", "-Q", "default,data_processing"]

# ================================
# R Worker Target (DEG pipeline + annoDB functional enrichment)
# ================================
# R and the CRAN/Bioconductor packages live in their own stage so the slow
# install layer stays cached across application-code changes (the final
# r-worker stage only re-copies the source).
FROM python:3.11-slim as r-deps

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Python build deps + R and the system libraries the R packages need to compile
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    libffi-dev \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libuv1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install CRAN + Bioconductor R packages (see r_scripts/install_packages.R).
# Cached unless install_packages.R changes.
COPY r_scripts/install_packages.R /app/r_scripts/install_packages.R
RUN Rscript /app/r_scripts/install_packages.R

# LaTeX toolchain for the SciLicium PDF report (rendered + compiled on this worker).
# Distinct layer so it caches independently of the app source / R packages.
# Covers the packages required by the SciLicium template: tcolorbox/pgf, siunitx,
# longtable/booktabs, txfonts, tocloft, datetime2, fancyhdr, geometry, Carlito, …
RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-science \
    texlive-pictures \
    texlive-plain-generic \
    texlive-lang-french \
    lmodern \
    fonts-crosextra-carlito \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

FROM r-deps as r-worker

# Application source (and the rest of r_scripts, incl. the pipeline + enrichment)
COPY . .

CMD ["celery", "-A", "app.worker.celery_app", "worker", "--loglevel=info", "-Q", "r_analysis", "--concurrency=1"]
