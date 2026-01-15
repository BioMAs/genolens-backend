"""
Data processing service for converting files to Parquet and querying.
"""
import io
from typing import Any, Optional
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from app.core.config import settings


class DataProcessorService:
    """Service for data processing operations (CSV/Excel -> Parquet)."""

    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.compression = settings.PARQUET_COMPRESSION

    async def convert_to_parquet(
        self,
        file_data: bytes,
        file_extension: str
    ) -> bytes:
        """
        Convert CSV/Excel/TSV file to Parquet format.

        Args:
            file_data: Raw file bytes
            file_extension: File extension (.csv, .xlsx, .tsv, etc.)

        Returns:
            bytes: Parquet file as bytes

        Raises:
            ValueError: If file format is not supported
        """
        # Read the file based on extension
        df = await self._read_file(file_data, file_extension)

        # Standardize gene column
        # Rename 'Unnamed: 0' or 'gene' to 'gene_id'
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "gene_id"})
        elif "gene" in df.columns:
            df = df.rename(columns={"gene": "gene_id"})
        
        # Ensure gene_id is the first column if it exists
        if "gene_id" in df.columns:
            cols = ["gene_id"] + [c for c in df.columns if c != "gene_id"]
            df = df[cols]

        # Convert to Parquet
        parquet_buffer = io.BytesIO()
        df.to_parquet(
            parquet_buffer,
            engine="pyarrow",
            compression=self.compression,
            index=False
        )
        parquet_buffer.seek(0)

        return parquet_buffer.read()

    async def _read_file(self, file_data: bytes, file_extension: str) -> pd.DataFrame:
        """
        Read file into pandas DataFrame based on extension.

        Args:
            file_data: Raw file bytes
            file_extension: File extension

        Returns:
            pd.DataFrame: Loaded data

        Raises:
            ValueError: If file format is not supported
        """
        file_buffer = io.BytesIO(file_data)

        if file_extension in [".csv"]:
            return pd.read_csv(file_buffer)
        elif file_extension in [".tsv", ".txt"]:
            return pd.read_csv(file_buffer, sep="\t")
        elif file_extension in [".xlsx", ".xls"]:
            return pd.read_excel(file_buffer)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

    async def query_parquet(
        self,
        parquet_data: bytes,
        gene_ids: Optional[list[str]] = None,
        sample_ids: Optional[list[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> dict[str, Any]:
        """
        Query a Parquet file with filters.

        Args:
            parquet_data: Parquet file bytes
            gene_ids: Filter by gene IDs (assumes 'gene_id' column exists)
            sample_ids: Filter by sample IDs (column names)
            limit: Maximum rows to return
            offset: Number of rows to skip

        Returns:
            dict: Query results with columns and data
        """
        parquet_buffer = io.BytesIO(parquet_data)
        df = pd.read_parquet(parquet_buffer)

        # Apply filters
        if gene_ids:
            if "gene_id" in df.columns:
                df = df[df["gene_id"].isin(gene_ids)]

        if sample_ids:
            # Filter columns to include gene_id + requested samples
            available_samples = [s for s in sample_ids if s in df.columns]
            columns_to_keep = ["gene_id"] if "gene_id" in df.columns else []
            columns_to_keep.extend(available_samples)
            df = df[columns_to_keep]

        # Get total before pagination
        total_rows = len(df)

        # Apply pagination
        df = df.iloc[offset:offset + limit]

        # Handle NaN and Infinity for JSON serialization
        # Manual cleaning to be absolutely sure
        data = df.to_dict(orient="records")
        cleaned_data = []
        for row in data:
            cleaned_row = {}
            for k, v in row.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    cleaned_row[k] = None
                else:
                    cleaned_row[k] = v
            cleaned_data.append(cleaned_row)

        # Convert to dict format
        return {
            "columns": df.columns.tolist(),
            "data": cleaned_data,
            "total_rows": total_rows,
            "returned_rows": len(df)
        }

    async def get_dataframe(self, parquet_data: bytes) -> pd.DataFrame:
        """
        Get the full DataFrame from Parquet data.
        
        Args:
            parquet_data: Parquet file bytes
            
        Returns:
            pd.DataFrame: The loaded DataFrame
        """
        parquet_buffer = io.BytesIO(parquet_data)
        return pd.read_parquet(parquet_buffer)

    async def get_file_metadata(self, file_data: bytes, file_extension: str) -> dict[str, Any]:
        """
        Extract metadata from a file.

        Args:
            file_data: Raw file bytes
            file_extension: File extension

        Returns:
            dict: Metadata (rows, columns, size, etc.)
        """
        df = await self._read_file(file_data, file_extension)

        metadata = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
        }

        # Attempt to detect comparisons if it looks like a DEG file
        comparisons = self._detect_comparisons(df.columns.tolist())
        if comparisons:
            metadata["comparisons"] = comparisons
            # Also populate columns_info for consistency with frontend/API expectation
            if "columns_info" not in metadata:
                metadata["columns_info"] = {}
            metadata["columns_info"]["comparisons"] = comparisons
        
        # Detect comparisons from gene_cluster column for enrichment files
        enrichment_comparisons = self._detect_enrichment_comparisons(df)
        if enrichment_comparisons:
            metadata["enrichment_comparisons"] = enrichment_comparisons

        return metadata

    def _detect_comparisons(self, columns: list[str]) -> dict[str, dict[str, str]]:
        """
        Detect comparisons from column names.
        Looks for patterns like 'log2FoldChange:ComparisonName', 'padj.Stouffer:ComparisonName', etc.
        Handles contrast: prefix in comparison names.
        """
        comparisons = {}
        
        # Common prefixes for DEG columns
        # Format: (prefix, key_in_metadata)
        # Note: Order matters, longer prefixes first to avoid partial matches
        patterns = [
            # Log Fold Change - with test methods
            ('log2FoldChange:', 'logFC'),
            ('logFC:', 'logFC'),
            ('logFC_', 'logFC'),
            ('log2FoldChange_', 'logFC'),
            ('FoldChange_', 'logFC'),
            
            # Adjusted P-value - with test methods (Stouffer, Fisher, etc.)
            ('padj.Stouffer:', 'padj'),
            ('padj.Fisher:', 'padj'),
            ('padj.edgeR:', 'padj'),
            ('padj.DESeq2:', 'padj'),
            ('padj:', 'padj'),
            ('padj_', 'padj'),
            ('adj.P.Val_', 'padj'),
            ('FDR_', 'padj'),
            
            # Raw P-value - with test methods
            ('pvalue.Stouffer:', 'pvalue'),
            ('pvalue.Fisher:', 'pvalue'),
            ('pvalue.edgeR:', 'pvalue'),
            ('pvalue.DESeq2:', 'pvalue'),
            ('pvalue:', 'pvalue'),
            ('pvalue_', 'pvalue'),
            ('P.Value_', 'pvalue'),
        ]
        
        # Track available test methods per comparison
        test_methods = {}
        
        for col in columns:
            # Skip 'contrast:' columns as they are just markers, not data columns
            if col.startswith('contrast:'):
                continue
                
            for prefix, key in patterns:
                if col.startswith(prefix):
                    comp_name_raw = col[len(prefix):]
                    
                    # Remove 'contrast:' prefix from comparison name if present
                    comp_name = comp_name_raw
                    if comp_name.startswith('contrast:'):
                        comp_name = comp_name[9:]  # len('contrast:') = 9
                    
                    # Extract test method from prefix if present
                    test_method = None
                    if '.Stouffer:' in prefix:
                        test_method = 'Stouffer'
                    elif '.Fisher:' in prefix:
                        test_method = 'Fisher'
                    elif '.edgeR:' in prefix:
                        test_method = 'edgeR'
                    elif '.DESeq2:' in prefix:
                        test_method = 'DESeq2'
                    
                    if comp_name not in comparisons:
                        comparisons[comp_name] = {}
                        test_methods[comp_name] = set()
                    
                    # Store the column name
                    comparisons[comp_name][key] = col
                    
                    # Track test method
                    if test_method and key in ['padj', 'pvalue']:
                        test_methods[comp_name].add(test_method)
                    
                    break # Found a match for this column, move to next column

        # Filter out incomplete comparisons (must have at least logFC and padj)
        valid_comparisons = {
            k: v for k, v in comparisons.items() 
            if 'logFC' in v and 'padj' in v
        }
        
        # Add test method information to metadata
        for comp_name in valid_comparisons:
            if comp_name in test_methods and test_methods[comp_name]:
                valid_comparisons[comp_name]['test_methods'] = sorted(list(test_methods[comp_name]))
        
        print(f"[DETECT] Found {len(valid_comparisons)} comparisons:")
        for comp_name, cols in valid_comparisons.items():
            print(f"  - {comp_name}: logFC={cols.get('logFC')}, padj={cols.get('padj')}, tests={cols.get('test_methods', ['default'])}")
        
        return valid_comparisons

    def _parse_comparison_name(self, comp_name: str) -> list[str]:
        """
        Parse comparison name to extract experimental groups.
        Examples:
            "WT_vs_KO" -> ["WT", "KO"]
            "Treatment_vs_Control" -> ["Treatment", "Control"]
            "GroupA-GroupB" -> ["GroupA", "GroupB"]
            "Condition1.Condition2" -> ["Condition1", "Condition2"]
        
        Returns:
            list[str]: List of group names
        """
        # Common separators in comparison names
        separators = ['_vs_', '_VS_', '-vs-', '-VS-', '_', '-', '.', ' vs ', ' VS ']
        
        for sep in separators:
            if sep in comp_name:
                parts = comp_name.split(sep)
                # Filter out common suffixes like 'up', 'down', 'all'
                filtered_parts = [
                    p for p in parts 
                    if p.lower() not in ['up', 'down', 'all', 'deg', 'degs', 'vs']
                ]
                if filtered_parts:
                    return filtered_parts
        
        # If no separator found, return the whole name as single group
        return [comp_name]

    def _detect_enrichment_comparisons(self, df: pd.DataFrame) -> list[str]:
        """
        Detect comparisons from enrichment files by analyzing the gene_cluster column.
        Common patterns: 'contrast:ConditionA_vs_ConditionB_up', 'cluster:Treatment_vs_Control_down'
        Splits by ':' and takes the last part as the comparison name.
        
        Returns:
            list[str]: List of unique comparison names
        """
        # Look for gene_cluster column (case-insensitive)
        gene_cluster_col = None
        for col in df.columns:
            if col.lower() in ['gene_cluster', 'genecluster', 'gene.cluster', 'cluster', 'comparison']:
                gene_cluster_col = col
                break
        
        if not gene_cluster_col:
            return []
        
        # Extract unique values from gene_cluster column
        unique_clusters = df[gene_cluster_col].dropna().unique()
        
        # Parse comparison names
        comparisons = set()
        for cluster in unique_clusters:
            cluster_str = str(cluster)
            
            # Split by ':' and take the last part (handles 'contrast:name', 'cluster:name')
            if ':' in cluster_str:
                cluster_str = cluster_str.split(':')[-1]
            
            # Add the original comparison name (with regulation suffix)
            comparisons.add(cluster_str.strip())
            
            # Also add base comparison name without regulation suffix
            base_cluster = cluster_str.strip()
            # Remove regulation suffixes with parentheses: " (up)", " (down)"
            if base_cluster.endswith(' (up)') or base_cluster.endswith(' (down)'):
                base_cluster = base_cluster.rsplit(' (', 1)[0]
                comparisons.add(base_cluster)
            # Remove common underscore suffixes
            for suffix in ['_up', '_down', '_upregulated', '_downregulated', '_UP', '_DOWN']:
                if base_cluster.endswith(suffix):
                    base_cluster = base_cluster[:-len(suffix)]
                    comparisons.add(base_cluster)
                    break
        
        return sorted(list(comparisons))

    async def calculate_pca(self, parquet_data: bytes, n_components: int = 2) -> dict[str, Any]:
        """
        Calculate PCA on the dataset.
        Assumes the dataset is an expression matrix (Genes x Samples).
        """
        parquet_buffer = io.BytesIO(parquet_data)
        df = pd.read_parquet(parquet_buffer)

        # Prepare data: Set gene_id as index if exists
        if "gene_id" in df.columns:
            df = df.set_index("gene_id")
        
        # Select only numeric columns (samples)
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Transpose: Samples as rows, Genes as columns
        # PCA works on samples (rows)
        X = numeric_df.T
        
        # Handle missing values (impute with 0)
        X = X.fillna(0)

        # Standardize features (genes)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Run PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(X_scaled)
        
        # Format results
        results = []
        for i, sample_name in enumerate(X.index):
            results.append({
                "sample": sample_name,
                "x": float(principal_components[i, 0]),
                "y": float(principal_components[i, 1]) if n_components > 1 else 0,
                "z": float(principal_components[i, 2]) if n_components > 2 else 0
            })

        return {
            "data": results,
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "total_variance": float(sum(pca.explained_variance_ratio_))
        }

    async def calculate_umap(
        self, 
        parquet_data: bytes, 
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1
    ) -> dict[str, Any]:
        """
        Calculate UMAP on the dataset.
        Assumes the dataset is an expression matrix (Genes x Samples).
        """
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP is not installed. Install with: pip install umap-learn")
        
        parquet_buffer = io.BytesIO(parquet_data)
        df = pd.read_parquet(parquet_buffer)

        # Prepare data: Set gene_id as index if exists
        if "gene_id" in df.columns:
            df = df.set_index("gene_id")
        
        # Select only numeric columns (samples)
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Transpose: Samples as rows, Genes as columns
        X = numeric_df.T
        
        # Handle missing values (impute with 0)
        X = X.fillna(0)

        # Standardize features (genes)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Run UMAP
        umap = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        embedding = umap.fit_transform(X_scaled)
        
        # Format results
        results = []
        for i, sample_name in enumerate(X.index):
            results.append({
                "sample": sample_name,
                "x": float(embedding[i, 0]),
                "y": float(embedding[i, 1]) if n_components > 1 else 0,
                "z": float(embedding[i, 2]) if n_components > 2 else 0
            })

        return {
            "data": results,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist
        }

    async def calculate_library_size(self, parquet_data: bytes) -> list[dict[str, Any]]:
        """
        Calculate library size (total reads) for each sample.
        """
        parquet_buffer = io.BytesIO(parquet_data)
        df = pd.read_parquet(parquet_buffer)

        # Prepare data: Set gene_id as index if exists
        if "gene_id" in df.columns:
            df = df.set_index("gene_id")
        
        # Select only numeric columns (samples)
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Sum columns
        library_sizes = numeric_df.sum().sort_values(ascending=False)
        
        # Count genes with >= 3 reads per sample
        genes_detected = (numeric_df >= 3).sum()
        
        return [
            {
                "sample": sample, 
                "reads": float(reads),
                "genes_detected": int(genes_detected[sample])
            }
            for sample, reads in library_sizes.items()
        ]

    async def calculate_volcano_plots(self, parquet_data: bytes, comparisons: dict[str, dict[str, str]]) -> dict[str, list[dict]]:
        """
        Pre-calculate volcano plot data for all comparisons in a DEG dataset.
        
        Args:
            parquet_data: Parquet file bytes
            comparisons: Dict of comparison names to column mappings
            
        Returns:
            Dict mapping comparison names to volcano plot data
        """
        parquet_buffer = io.BytesIO(parquet_data)
        df = pd.read_parquet(parquet_buffer)
        
        print(f"[VOLCANO] Processing {len(comparisons)} comparisons")
        print(f"[VOLCANO] Available columns: {df.columns.tolist()}")
        
        volcano_plots = {}
        for comp_name, cols in comparisons.items():
            logfc_col = cols.get('logFC')
            padj_col = cols.get('padj')
            
            print(f"[VOLCANO] Comparison '{comp_name}': logFC={logfc_col}, padj={padj_col}")
            
            if not logfc_col or not padj_col:
                print(f"[VOLCANO] Skipping '{comp_name}': missing columns")
                continue
                
            # Extract data
            plot_data = []
            zero_pval_count = 0
            valid_count = 0
            
            # Optimization: Separate significant and non-significant genes
            sig_genes = []
            non_sig_genes = []
            
            for idx, row in df.iterrows():
                gene_id = row.get('gene_id', idx)
                logfc = row.get(logfc_col)
                padj = row.get(padj_col)
                
                if pd.notna(logfc) and pd.notna(padj):
                    # Skip genes with p-value = 0 to avoid contaminating the volcano plot
                    if padj == 0:
                        zero_pval_count += 1
                        continue
                    
                    if padj > 0:
                        point = {
                            'gene_id': str(gene_id),
                            'logFC': round(float(logfc), 4),
                            'padj': float(padj), # Keep original precision for filtering
                            'negLogPadj': round(-np.log10(float(padj)), 4)
                        }
                        
                        if point['padj'] < 0.05:
                            sig_genes.append(point)
                        else:
                            non_sig_genes.append(point)
                        valid_count += 1
            
            # Downsample if too many points (limit to ~5000 points total for good visualization)
            MAX_POINTS = 5000
            
            if valid_count > MAX_POINTS:
                # Prioritize significant genes
                if len(sig_genes) >= MAX_POINTS:
                    # If too many significant genes, take top most significant
                    sig_genes.sort(key=lambda x: x['padj'])
                    plot_data = sig_genes[:MAX_POINTS]
                else:
                    # Take all significant genes
                    plot_data = sig_genes
                    # Fill remaining slots with stratified sample of non-significant genes
                    remaining_slots = MAX_POINTS - len(sig_genes)
                    if remaining_slots > 0 and non_sig_genes:
                        import random
                        # Use stratified sampling for better distribution across logFC range
                        if len(non_sig_genes) > remaining_slots:
                            # Sort by absolute logFC to get diverse representation
                            non_sig_genes.sort(key=lambda x: abs(x['logFC']))
                            # Take evenly spaced samples
                            step = len(non_sig_genes) / remaining_slots
                            indices = [int(i * step) for i in range(remaining_slots)]
                            plot_data.extend([non_sig_genes[i] for i in indices])
                        else:
                            plot_data.extend(non_sig_genes)
            else:
                plot_data = sig_genes + non_sig_genes
                
            # Remove 'padj' from final output if not needed for display to save space, 
            # but usually it's useful for tooltips. We'll keep it but maybe round it?
            # Let's round padj for the final output to save space
            for p in plot_data:
                p['padj'] = float(f"{p['padj']:.6g}")

            print(f"[VOLCANO] Comparison '{comp_name}': {valid_count} valid genes, {zero_pval_count} excluded (p-value=0). Keeping {len(plot_data)} points.")
            volcano_plots[comp_name] = plot_data
        
        return volcano_plots

    async def calculate_enrichment_dotplots(self, parquet_data: bytes, enrichment_comparisons: list[str]) -> dict[str, list[dict]]:
        """
        Pre-calculate dotplot data for enrichment datasets.
        
        Args:
            parquet_data: Parquet file bytes
            enrichment_comparisons: List of comparison names
            
        Returns:
            Dict mapping comparison names to dotplot data
        """
        parquet_buffer = io.BytesIO(parquet_data)
        df = pd.read_parquet(parquet_buffer)
        
        # Find required columns
        term_col = None
        for col in df.columns:
            if col.lower() in ['term', 'description']:
                term_col = col
                break
        
        pval_col = None
        for col in df.columns:
            if col == 'adj.p.hyper.enri' or 'adj.p' in col.lower():
                pval_col = col
                break
                
        r_col = 'r' if 'r' in df.columns else None
        r_expected_col = 'rExpected' if 'rExpected' in df.columns else None
        cluster_col = None
        
        for col in df.columns:
            if col.lower() in ['gene_cluster', 'gene.cluster', 'cluster']:
                cluster_col = col
                break
        
        if not all([term_col, pval_col, r_col, r_expected_col, cluster_col]):
            return {}
        
        dotplots = {}
        for comp_name in enrichment_comparisons:
            # Filter rows for this comparison
            comp_df = df[df[cluster_col].astype(str).str.contains(comp_name, na=False)]
            
            plot_data = []
            for idx, row in comp_df.iterrows():
                r = row.get(r_col)
                r_expected = row.get(r_expected_col)
                pval = row.get(pval_col)
                
                if pd.notna(r) and pd.notna(r_expected) and pd.notna(pval) and r_expected > 0 and pval > 0:
                    plot_data.append({
                        'term': str(row[term_col]),
                        'geneRatio': float(r) / float(r_expected),
                        'count': float(r),
                        'pvalue': float(pval),
                        'negLogP': -np.log10(float(pval))
                    })
            
            # Sort by p-value and take top 20
            plot_data.sort(key=lambda x: x['pvalue'])
            dotplots[comp_name] = plot_data[:20]
        
        return dotplots

    async def calculate_deg_heatmaps(self, parquet_data: bytes, matrix_parquet_data: bytes, comparisons: dict[str, dict[str, str]]) -> dict[str, dict]:
        """
        Pre-calculate heatmap data for all DEG comparisons with hierarchical clustering.
        
        Args:
            parquet_data: DEG file parquet bytes
            matrix_parquet_data: Expression matrix parquet bytes
            comparisons: Dict of comparison names to column mappings
            
        Returns:
            Dict mapping comparison names to heatmap data clustered by up/down regulation
        """
        deg_buffer = io.BytesIO(parquet_data)
        deg_df = pd.read_parquet(deg_buffer)
        
        matrix_buffer = io.BytesIO(matrix_parquet_data)
        matrix_df = pd.read_parquet(matrix_buffer)
        
        if 'gene_id' in matrix_df.columns:
            matrix_df = matrix_df.set_index('gene_id')
        
        heatmaps = {}
        for comp_name, cols in comparisons.items():
            logfc_col = cols.get('logFC')
            padj_col = cols.get('padj')
            
            print(f"Processing comparison: {comp_name}")
            print(f"LogFC column: {logfc_col}, Padj column: {padj_col}")
            
            if not logfc_col or not padj_col:
                continue
            
            # Filter significant DEGs (0 < padj < 0.05, |logFC| > 1)
            # Exclude padj = 0 (invalid/missing values)
            sig_genes_data = deg_df[
                (deg_df[padj_col] > 0) &
                (deg_df[padj_col] < 0.05) &
                (deg_df[logfc_col].abs() > 1)
            ]
            
            if len(sig_genes_data) == 0:
                continue
            
            # Separate and sort by up/down regulation
            up_regulated = sig_genes_data[sig_genes_data[logfc_col] > 0].sort_values(
                by=logfc_col, ascending=False
            )['gene_id'].tolist() if 'gene_id' in sig_genes_data.columns else []
            
            down_regulated = sig_genes_data[sig_genes_data[logfc_col] < 0].sort_values(
                by=logfc_col, ascending=True
            )['gene_id'].tolist() if 'gene_id' in sig_genes_data.columns else []
            
            print(f"Found {len(up_regulated)} up-regulated and {len(down_regulated)} down-regulated genes")
            
            # Combine: up-regulated first, then down-regulated
            sig_genes = up_regulated + down_regulated
            
            # Create regulation mapping
            regulation_map = {}
            for gene in up_regulated:
                regulation_map[gene] = 'up'
            for gene in down_regulated:
                regulation_map[gene] = 'down'
            
            print(f"Regulation map created with {len(regulation_map)} genes")
            
            if len(sig_genes) == 0:
                continue
            
            # Get expression data for significant genes
            sig_matrix = matrix_df.loc[matrix_df.index.isin(sig_genes)]
            
            if sig_matrix.empty:
                continue
            
            # Extract comparison groups from name (e.g., "WT_vs_KO" -> ["WT", "KO"])
            comparison_groups = self._parse_comparison_name(comp_name)
            
            # Filter samples based on comparison groups
            filtered_samples = []
            for sample in sig_matrix.columns:
                # Check if sample name contains any of the comparison groups
                for group in comparison_groups:
                    if group.lower() in sample.lower():
                        filtered_samples.append(sample)
                        break
            
            # If we found matching samples, use them; otherwise use all samples
            if filtered_samples:
                sig_matrix = sig_matrix[filtered_samples]
            
            # Z-score normalization
            z_scored = sig_matrix.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
            z_scored = z_scored.fillna(0)
            
            # Reorder to match sig_genes order (already sorted by up/down)
            z_scored = z_scored.reindex(sig_genes)
            
            # Convert to list format
            heatmap_data = {
                'genes': z_scored.index.tolist(),
                'samples': z_scored.columns.tolist(),
                'values': z_scored.values.tolist(),
                'gene_count': len(z_scored),
                'comparison_groups': comparison_groups,
                'regulation': regulation_map
            }
            
            heatmaps[comp_name] = heatmap_data
        
        return heatmaps

    async def calculate_deg_statistics(
        self,
        parquet_data: bytes,
        comparisons: dict[str, dict[str, str]],
        logfc_threshold: float = 0.58,
        padj_threshold: float = 0.05
    ) -> dict[str, dict[str, Any]]:
        """
        Calculate DEG statistics for all comparisons in a DEG dataset.
        Matches the exact filtering logic used in the DEG table component.

        Args:
            parquet_data: Parquet file bytes
            comparisons: Dict of comparison names to column mappings
            logfc_threshold: Log2 fold change threshold for significance (default: 0.58)
            padj_threshold: Adjusted p-value threshold for significance (default: 0.05)

        Returns:
            Dict mapping comparison names to statistics:
            {
                'comparison_name': {
                    'deg_up': int,
                    'deg_down': int,
                    'deg_total': int,
                    'top_genes': [
                        {'gene': str, 'logFC': float, 'padj': float},
                        ...
                    ]
                }
            }
        """
        parquet_buffer = io.BytesIO(parquet_data)
        df = pd.read_parquet(parquet_buffer)

        print(f"[DEG_STATS] Processing {len(comparisons)} comparisons")

        statistics = {}
        for comp_name, cols in comparisons.items():
            logfc_col = cols.get('logFC')
            padj_col = cols.get('padj')

            print(f"[DEG_STATS] Comparison '{comp_name}': logFC={logfc_col}, padj={padj_col}")

            if not logfc_col or not padj_col:
                print(f"[DEG_STATS] Skipping '{comp_name}': missing columns")
                continue

            # Check if contrast column exists for this comparison
            contrast_col = f"contrast:{comp_name}"
            has_contrast_col = contrast_col in df.columns

            print(f"[DEG_STATS] Looking for contrast column: {contrast_col}, Found: {has_contrast_col}")

            if has_contrast_col:
                # Use contrast column to count (more accurate)
                # Filter genes that belong to this comparison (non-empty contrast value)
                comparison_genes = df[
                    (df[contrast_col].notna()) &
                    (df[contrast_col] != '') &
                    (df[contrast_col] != None)
                ].copy()

                print(f"[DEG_STATS] Total genes in contrast column: {len(comparison_genes)}")

                # Count UP and DOWN based on contrast column values
                deg_up = len(comparison_genes[comparison_genes[contrast_col].str.upper() == 'UP'])
                deg_down = len(comparison_genes[comparison_genes[contrast_col].str.upper() == 'DOWN'])
                deg_total = deg_up + deg_down

                # For top genes, filter by the same criteria as DEG table
                significant = comparison_genes[
                    (comparison_genes[padj_col].notna()) &
                    (comparison_genes[logfc_col].notna()) &
                    (comparison_genes[padj_col] < padj_threshold) &
                    (np.abs(comparison_genes[logfc_col]) > logfc_threshold)
                ].copy()
            else:
                # Fallback: use logFC sign (old method)
                print(f"[DEG_STATS] No contrast column found, using logFC sign")
                significant = df[
                    (df[padj_col].notna()) &
                    (df[logfc_col].notna()) &
                    (df[padj_col] < padj_threshold) &
                    (np.abs(df[logfc_col]) > logfc_threshold)
                ].copy()

                deg_up = len(significant[significant[logfc_col] > 0])
                deg_down = len(significant[significant[logfc_col] < 0])
                deg_total = len(significant)

            # Get top 10 genes by adjusted p-value
            top_genes_df = significant.nsmallest(10, padj_col)
            top_genes = []

            for idx, row in top_genes_df.iterrows():
                gene_id = row.get('gene_id', idx)
                logfc = row.get(logfc_col)
                padj = row.get(padj_col)

                top_genes.append({
                    'gene': str(gene_id),
                    'logFC': float(logfc),
                    'padj': float(padj)
                })

            statistics[comp_name] = {
                'deg_up': deg_up,
                'deg_down': deg_down,
                'deg_total': deg_total,
                'top_genes': top_genes
            }

            print(f"[DEG_STATS] Comparison '{comp_name}': {deg_up} up, {deg_down} down, {deg_total} total, {len(top_genes)} top genes")

        return statistics

    async def extract_deg_genes_for_db(
        self,
        parquet_data: bytes,
        comparisons: dict[str, dict[str, str]]
    ) -> dict[str, list[dict]]:
        """
        Extract all DEG genes for database storage.
        Returns all genes with their contrast column value for efficient querying.

        Args:
            parquet_data: Parquet file bytes
            comparisons: Dict of comparison names to column mappings

        Returns:
            Dict mapping comparison names to list of gene records
        """
        parquet_buffer = io.BytesIO(parquet_data)
        df = pd.read_parquet(parquet_buffer)

        print(f"[DEG_EXTRACT] Processing {len(comparisons)} comparisons for DB storage")

        all_genes = {}
        for comp_name, cols in comparisons.items():
            logfc_col = cols.get('logFC')
            padj_col = cols.get('padj')
            pvalue_col = cols.get('pvalue')  # Optional

            if not logfc_col or not padj_col:
                print(f"[DEG_EXTRACT] Skipping '{comp_name}': missing columns")
                continue

            # Check for contrast column
            contrast_col = f"contrast:{comp_name}"
            has_contrast_col = contrast_col in df.columns

            if has_contrast_col:
                # Filter genes that belong to this comparison
                comparison_genes = df[
                    (df[contrast_col].notna()) &
                    (df[contrast_col] != '') &
                    (df[contrast_col] != None)
                ].copy()

                genes_list = []
                for idx, row in comparison_genes.iterrows():
                    gene_id = row.get('gene_id', idx)
                    logfc = row.get(logfc_col)
                    padj = row.get(padj_col)
                    contrast_value = row.get(contrast_col)

                    # Skip invalid values
                    if pd.isna(logfc) or pd.isna(padj):
                        continue

                    gene_record = {
                        'gene_id': str(gene_id),
                        'log_fc': float(logfc),
                        'padj': float(padj),
                        'regulation': str(contrast_value).upper() if contrast_value else None,
                    }

                    # Add optional fields
                    if pvalue_col and pvalue_col in row.index:
                        pval = row.get(pvalue_col)
                        if not pd.isna(pval):
                            gene_record['pvalue'] = float(pval)
                    
                    # Add baseMean if available
                    if 'baseMean' in row.index:
                        base_mean = row.get('baseMean')
                        if not pd.isna(base_mean):
                            gene_record['base_mean'] = float(base_mean)

                    # Try to find gene name
                    gene_name = None
                    for col in ['gene_name', 'symbol', 'Symbol', 'GeneName', 'GENE_NAME']:
                        if col in row.index:
                            val = row.get(col)
                            if not pd.isna(val):
                                gene_name = str(val)
                                break
                    
                    if gene_name:
                        gene_record['gene_name'] = gene_name

                    genes_list.append(gene_record)

                all_genes[comp_name] = genes_list
                print(f"[DEG_EXTRACT] Comparison '{comp_name}': {len(genes_list)} genes extracted (using contrast column)")
            
            else:
                # Fallback: use logFC sign and thresholds if no contrast column
                # This ensures we still populate the DB even without explicit contrast columns
                print(f"[DEG_EXTRACT] No contrast column for '{comp_name}', using logFC/padj thresholds")
                
                # Default thresholds for "significant" genes to store
                # We store significant genes to populate the DEG table
                padj_threshold = 0.05
                logfc_threshold = 0.58 # approx 1.5 fold change
                
                comparison_genes = df[
                    (df[padj_col].notna()) &
                    (df[logfc_col].notna()) &
                    (df[padj_col] < padj_threshold) &
                    (np.abs(df[logfc_col]) > logfc_threshold)
                ].copy()
                
                genes_list = []
                for idx, row in comparison_genes.iterrows():
                    gene_id = row.get('gene_id', idx)
                    logfc = row.get(logfc_col)
                    padj = row.get(padj_col)
                    
                    # Determine regulation
                    regulation = "UP" if logfc > 0 else "DOWN"
                    
                    gene_record = {
                        'gene_id': str(gene_id),
                        'log_fc': float(logfc),
                        'padj': float(padj),
                        'regulation': regulation,
                    }
                    
                    # Add optional fields
                    if pvalue_col and pvalue_col in row.index:
                        pval = row.get(pvalue_col)
                        if not pd.isna(pval):
                            gene_record['pvalue'] = float(pval)
                            
                    # Add baseMean if available
                    if 'baseMean' in row.index:
                        base_mean = row.get('baseMean')
                        if not pd.isna(base_mean):
                            gene_record['base_mean'] = float(base_mean)

                    # Try to find gene name
                    gene_name = None
                    for col in ['gene_name', 'symbol', 'Symbol', 'GeneName', 'GENE_NAME']:
                        if col in row.index:
                            val = row.get(col)
                            if not pd.isna(val):
                                gene_name = str(val)
                                break
                    
                    if gene_name:
                        gene_record['gene_name'] = gene_name

                    genes_list.append(gene_record)
                
                all_genes[comp_name] = genes_list
                print(f"[DEG_EXTRACT] Comparison '{comp_name}': {len(genes_list)} genes extracted (using thresholds)")

        return all_genes


    async def extract_enrichment_pathways_for_db(
        self,
        parquet_data: bytes,
        comparisons: dict[str, dict[str, str]]
    ) -> dict[str, list[dict]]:
        """
        Extract enrichment pathways from Parquet file for database storage.

        Enrichment files typically have these columns:
        - pathway_id (e.g., GO:0006915, hsa04210)
        - description/pathway_name
        - GeneRatio (e.g., "5/100")
        - BgRatio (e.g., "50/10000")
        - pvalue
        - p.adjust/padj
        - geneID (e.g., "GENE1/GENE2/GENE3")
        - Count (number of genes)
        - Category (GO:BP, KEGG, etc.)

        Args:
            parquet_data: Parquet file as bytes
            comparisons: Dict of comparison names and their column mappings

        Returns:
            Dict[comparison_name, List[pathway_dict]]
        """
        import pandas as pd
        import io

        df = pd.read_parquet(io.BytesIO(parquet_data))
        result = {}

        # Common column name variations for enrichment files
        pathway_id_cols = ['ID', 'pathway_id', 'Term', 'term_id', 'term']
        pathway_name_cols = ['Description', 'pathway_name', 'Term', 'description', 'pathway', 'term']
        pvalue_cols = ['pvalue', 'PValue', 'p.value', 'p.hyper.enri', 'p.norm.enri', 'p.hyper.depl', 'p.norm.depl']
        padj_cols = ['p.adjust', 'padj', 'qvalue', 'FDR', 'adj.P.Val', 'adj.p.hyper.enri', 'adj.p.norm.enri', 'adj.p.hyper.depl', 'adj.p.norm.depl']
        gene_ratio_cols = ['GeneRatio', 'gene_ratio', 'ratio']
        bg_ratio_cols = ['BgRatio', 'bg_ratio', 'background_ratio']
        genes_cols = ['geneID', 'gene_id', 'genes', 'core_enrichment']
        count_cols = ['Count', 'count', 'gene_count']
        category_cols = ['Category', 'category', 'ONTOLOGY', 'ontology']

        # Helper to find first matching column
        def find_col(possible_names: list[str]) -> str | None:
            for name in possible_names:
                if name in df.columns:
                    return name
            return None

        # Detect column names
        pathway_id_col = find_col(pathway_id_cols)
        pathway_name_col = find_col(pathway_name_cols)
        pvalue_col = find_col(pvalue_cols)
        padj_col = find_col(padj_cols)
        gene_ratio_col = find_col(gene_ratio_cols)
        bg_ratio_col = find_col(bg_ratio_cols)
        genes_col = find_col(genes_cols)
        count_col = find_col(count_cols)
        category_col = find_col(category_cols)

        if not all([pathway_id_col, pathway_name_col, padj_col]):
            print(f"[DataProcessor] Enrichment file missing required columns")
            print(f"  - Pathway ID: {pathway_id_col}")
            print(f"  - Pathway Name: {pathway_name_col}")
            print(f"  - Adjusted P-value: {padj_col}")
            return {}

        # For enrichment files, we might have:
        # 1. One comparison per file (most common)
        # 2. Multiple comparisons with a "contrast" or "comparison" column

        comparison_col = None
        for col in ['gene.cluster', 'contrast', 'comparison', 'Comparison', 'cluster']:
            if col in df.columns:
                comparison_col = col
                break

        if comparison_col:
            # Multiple comparisons in one file
            # Handle UP/DOWN regulation from gene.cluster column
            unique_clusters = df[comparison_col].unique()
            
            for cluster_name in unique_clusters:
                comp_data = df[df[comparison_col] == cluster_name].copy()
                
                # Parse regulation and base comparison name
                cluster_str = str(cluster_name)
                base_name = cluster_str
                regulation = 'ALL'
                
                # Check for various regulation suffixes
                # 1. Parentheses: "Name (up)", "Name (down)"
                if cluster_str.endswith(' (up)'):
                    base_name = cluster_str.rsplit(' (', 1)[0]
                    regulation = 'UP'
                elif cluster_str.endswith(' (down)'):
                    base_name = cluster_str.rsplit(' (', 1)[0]
                    regulation = 'DOWN'
                    
                # 2. Underscores: "Name_up", "Name_down"
                elif cluster_str.endswith('_up') or cluster_str.endswith('_UP'):
                    base_name = cluster_str[:-3]
                    regulation = 'UP'
                elif cluster_str.endswith('_down') or cluster_str.endswith('_DOWN'):
                    base_name = cluster_str[:-5]
                    regulation = 'DOWN'
                
                # 3. Upregulated/Downregulated
                elif cluster_str.endswith('_upregulated'):
                    base_name = cluster_str[:-12]
                    regulation = 'UP'
                elif cluster_str.endswith('_downregulated'):
                    base_name = cluster_str[:-14]
                    regulation = 'DOWN'

                # Remove common prefixes from the base name (after stripping suffixes)
                if base_name.startswith('contrast:'):
                    base_name = base_name[9:]
                elif base_name.startswith('cluster:'):
                    base_name = base_name[8:]
                elif base_name.startswith('comb_prob_test:'):
                    base_name = base_name[15:]

                # Extract pathways
                pathways = self._extract_pathways_from_df(
                    comp_data,
                    pathway_id_col,
                    pathway_name_col,
                    pvalue_col,
                    padj_col,
                    gene_ratio_col,
                    bg_ratio_col,
                    genes_col,
                    count_col,
                    category_col
                )
                
                # Add regulation field to each pathway
                for pathway in pathways:
                    pathway['regulation'] = regulation
                
                # Merge into result dict using base_name as key
                if base_name in result:
                    result[base_name].extend(pathways)
                else:
                    result[base_name] = pathways
        else:
            # Single comparison - use first comparison name from comparisons dict
            if comparisons:
                comp_name = list(comparisons.keys())[0]
                pathways = self._extract_pathways_from_df(
                    df,
                    pathway_id_col,
                    pathway_name_col,
                    pvalue_col,
                    padj_col,
                    gene_ratio_col,
                    bg_ratio_col,
                    genes_col,
                    count_col,
                    category_col
                )
                
                # Add default regulation for single comparison
                for pathway in pathways:
                    pathway['regulation'] = 'ALL'
                
                result[comp_name] = pathways

        return result

    def _extract_pathways_from_df(
        self,
        df: pd.DataFrame,
        pathway_id_col: str,
        pathway_name_col: str,
        pvalue_col: str | None,
        padj_col: str,
        gene_ratio_col: str | None,
        bg_ratio_col: str | None,
        genes_col: str | None,
        count_col: str | None,
        category_col: str | None
    ) -> list[dict]:
        """Extract pathway data from a DataFrame."""
        pathways_list = []

        for _, row in df.iterrows():
            # Parse gene ratio (e.g., "5/100" -> 0.05)
            gene_ratio = None
            if gene_ratio_col and pd.notna(row.get(gene_ratio_col)):
                try:
                    ratio_str = str(row[gene_ratio_col])
                    if '/' in ratio_str:
                        num, denom = ratio_str.split('/')
                        gene_ratio = float(num) / float(denom) if float(denom) > 0 else 0
                except:
                    pass

            # Parse background ratio
            bg_ratio = None
            if bg_ratio_col and pd.notna(row.get(bg_ratio_col)):
                try:
                    ratio_str = str(row[bg_ratio_col])
                    if '/' in ratio_str:
                        num, denom = ratio_str.split('/')
                        bg_ratio = float(num) / float(denom) if float(denom) > 0 else 0
                except:
                    pass

            # Parse genes list (e.g., "GENE1/GENE2/GENE3" -> ["GENE1", "GENE2", "GENE3"])
            genes = None
            if genes_col and pd.notna(row.get(genes_col)):
                genes_str = str(row[genes_col])
                if '/' in genes_str:
                    genes = genes_str.split('/')
                elif '|' in genes_str:
                    genes = genes_str.split('|')
                elif ',' in genes_str:
                    genes = [g.strip() for g in genes_str.split(',')]
                else:
                    genes = [genes_str]

            # Get gene count
            gene_count = 0
            if count_col and pd.notna(row.get(count_col)):
                try:
                    gene_count = int(row[count_col])
                except (ValueError, TypeError):
                    gene_count = 0
            elif genes:
                gene_count = len(genes)

            # Prepare truncated and safe values
            p_id = str(row[pathway_id_col])[:255]
            p_name = str(row[pathway_name_col])[:500]
            
            p_val = 1.0
            if pvalue_col and pd.notna(row.get(pvalue_col)):
                try:
                    p_val = float(row[pvalue_col])
                except (ValueError, TypeError):
                    p_val = 1.0
            
            p_adj = 1.0
            if pd.notna(row.get(padj_col)):
                 try:
                    p_adj = float(row[padj_col])
                 except (ValueError, TypeError):
                    p_adj = 1.0

            cat = "General"
            if category_col and pd.notna(row.get(category_col)):
                cat = str(row[category_col])[:100]
            
            pathway_dict = {
                'pathway_id': p_id,
                'pathway_name': p_name,
                'pvalue': p_val,
                'padj': p_adj,
                'gene_ratio': gene_ratio,
                'bg_ratio': bg_ratio,
                'genes': genes,
                'gene_count': gene_count,
                'category': cat,
                'description': p_name  # Use pathway name as description
            }

            pathways_list.append(pathway_dict)

        return pathways_list


# Global instance
data_processor = DataProcessorService()
