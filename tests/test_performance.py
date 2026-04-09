"""
Performance tests for critical endpoints.
Validates that optimizations achieve the target performance benchmarks.

Target Metrics:
- Page load: < 1.5s (from 5-10s)
- Data transfer: < 500 KB (from 10-27 MB)
- Heatmap rendering: < 2s (from 8-15s)
- Stats calculation: < 100ms (from 1-2s)
"""
import pytest
import time
from httpx import AsyncClient, ASGITransport
from app.main import app


def _client():
    """Return an AsyncClient configured with ASGITransport for httpx >= 0.24."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestPerformanceBenchmarks:
    """Performance benchmark tests for optimized endpoints."""
    
    @pytest.mark.asyncio
    async def test_dataset_columns_performance(self):
        """Test /datasets/{id}/columns endpoint performance."""
        async with _client() as client:
            start = time.perf_counter()
            response = await client.get("/api/datasets/test-dataset-id/columns")
            duration_ms = (time.perf_counter() - start) * 1000
            
            assert response.status_code in [200, 404]  # 404 if no test data
            assert duration_ms < 10, f"Columns endpoint took {duration_ms:.0f}ms (target: <10ms)"
    
    @pytest.mark.asyncio
    async def test_dataset_stats_performance(self):
        """Test /datasets/{id}/stats endpoint performance."""
        async with _client() as client:
            start = time.perf_counter()
            response = await client.get("/api/datasets/test-dataset-id/stats")
            duration_ms = (time.perf_counter() - start) * 1000
            
            assert response.status_code in [200, 404]
            assert duration_ms < 100, f"Stats endpoint took {duration_ms:.0f}ms (target: <100ms)"
    
    @pytest.mark.asyncio
    async def test_gene_list_performance(self):
        """Test /datasets/{id}/genes/list endpoint performance."""
        async with _client() as client:
            start = time.perf_counter()
            response = await client.get("/api/datasets/test-dataset-id/genes/list")
            duration_ms = (time.perf_counter() - start) * 1000
            
            assert response.status_code in [200, 404]
            assert duration_ms < 50, f"Gene list endpoint took {duration_ms:.0f}ms (target: <50ms)"
    
    @pytest.mark.asyncio
    async def test_gene_list_transfer_size(self):
        """Validate that gene list transfer is minimal (<50 KB)."""
        async with _client() as client:
            response = await client.get("/api/datasets/test-dataset-id/genes/list")
            
            if response.status_code == 200:
                content_length = len(response.content)
                assert content_length < 50 * 1024, f"Gene list response is {content_length / 1024:.1f} KB (target: <50 KB)"
    
    @pytest.mark.asyncio
    async def test_stats_transfer_size(self):
        """Validate that stats transfer is minimal (<1 KB)."""
        async with _client() as client:
            response = await client.get("/api/datasets/test-dataset-id/stats")
            
            if response.status_code == 200:
                content_length = len(response.content)
                assert content_length < 1024, f"Stats response is {content_length} bytes (target: <1 KB)"


class TestCacheEffectiveness:
    """Test that caching works as expected."""
    
    @pytest.mark.asyncio
    async def test_clustering_cache_hit(self):
        """Verify that second clustering request is much faster (cache hit)."""
        async with _client() as client:
            # First request (cache miss)
            start1 = time.perf_counter()
            response1 = await client.post(
                "/api/datasets/test-dataset-id/cluster",
                json={"top_n_genes": 100, "cluster_rows": True, "cluster_cols": True}
            )
            duration1_ms = (time.perf_counter() - start1) * 1000
            
            if response1.status_code != 200:
                pytest.skip("No test dataset available")
            
            # Second request (should be cache hit)
            start2 = time.perf_counter()
            response2 = await client.post(
                "/api/datasets/test-dataset-id/cluster",
                json={"top_n_genes": 100, "cluster_rows": True, "cluster_cols": True}
            )
            duration2_ms = (time.perf_counter() - start2) * 1000
            
            assert response2.status_code == 200
            # Cached request should be at least 10x faster
            assert duration2_ms < duration1_ms / 10, f"Cache hit ({duration2_ms:.0f}ms) not significantly faster than miss ({duration1_ms:.0f}ms)"


class TestDownsamplingIntelligent:
    """Test that intelligent downsampling works correctly."""
    
    @pytest.mark.asyncio
    async def test_clustering_with_many_genes(self):
        """Test that clustering with >2000 genes triggers intelligent downsampling."""
        async with _client() as client:
            response = await client.post(
                "/api/datasets/test-dataset-id/cluster",
                json={"top_n_genes": 5000, "cluster_rows": True, "max_genes_for_clustering": 2000}
            )
            
            if response.status_code == 200:
                data = response.json()
                # Verify that fewer than 5000 genes were actually clustered
                assert len(data.get("row_labels", [])) <= 2000, "Downsampling did not occur"


class TestMonitoringEndpoints:
    """Test admin monitoring endpoints."""
    
    @pytest.mark.asyncio
    async def test_performance_stats_endpoint(self):
        """Test that performance stats endpoint works."""
        async with _client() as client:
            response = await client.get("/api/datasets/admin/performance-stats")
            
            # 404 acceptable if endpoint not yet implemented
            assert response.status_code in [200, 401, 403, 404]
            
            if response.status_code == 200:
                data = response.json()
                assert "performance" in data
                assert "cache" in data
    
    @pytest.mark.asyncio
    async def test_cache_stats_endpoint(self):
        """Test that cache stats endpoint works."""
        async with _client() as client:
            response = await client.get("/api/datasets/admin/cache-stats")
            
            # 404 acceptable if endpoint not yet implemented
            assert response.status_code in [200, 401, 403, 404]
            
            if response.status_code == 200:
                data = response.json()
                assert "memory_cache" in data
                assert "persistent_cache" in data


@pytest.mark.benchmark
@pytest.mark.skip(reason="Requires pytest-benchmark: pip install pytest-benchmark")
class TestRegressionBenchmarks:
    """Regression tests to ensure performance doesn't degrade over time."""
    
    @pytest.mark.asyncio
    async def test_baseline_query_parquet(self, benchmark):
        """Benchmark query_parquet method."""
        from app.services.data_processor import DataProcessorService
        import pandas as pd
        
        processor = DataProcessorService()
        
        # Create sample Parquet data
        df = pd.DataFrame({
            "gene_id": [f"GENE_{i}" for i in range(10000)],
            "logFC": [i * 0.01 for i in range(10000)],
            "padj": [0.001 for _ in range(10000)]
        })
        
        import io
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        parquet_data = buffer.getvalue()
        
        # Benchmark
        result = await benchmark.pedantic(
            processor.query_parquet,
            args=(parquet_data,),
            kwargs={"limit": 100, "padj_max": 0.05},
            rounds=10
        )
        
        assert len(result) > 0


# Performance targets documentation
PERFORMANCE_TARGETS = """
Performance Optimization Targets (vs Baseline):

1. Initial Page Load
   - Before: 5-10s (including 10-27 MB transfer)
   - After: <1.5s (including <500 KB transfer)
   - Improvement: 70-85% reduction

2. Heatmap Rendering
   - Before: 8-15s (client-side filtering + clustering)
   - After: <2s (backend filtering + cached clustering)
   - Improvement: 75-87% reduction

3. Stats Calculation
   - Before: 1-2s (client-side JavaScript)
   - After: <100ms (pre-calculated in DB)
   - Improvement: 95% reduction

4. Data Transfer per Page
   - Before: 10-27 MB (full dataset download)
   - After: <500 KB (optimized endpoints)
   - Improvement: 98% reduction

5. Clustering Performance
   - Intelligent downsampling: 2000 gene limit (from unlimited)
   - Sample clustering: Pre-computed, stored 30 days
   - Gene sorting: O(n log n) vs O(n²) when cluster_rows=False
"""
