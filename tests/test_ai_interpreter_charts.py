"""Tests for chart-type prompt builders in LocalAIInterpreter."""
import pytest
from unittest.mock import AsyncMock, patch
from app.services.ai_interpreter import LocalAIInterpreter


VOLCANO_CTX = {
    "comparison_name": "Treated_vs_Control",
    "up_count": 127,
    "down_count": 94,
    "top_up_genes": [{"gene_id": "BRCA1", "log_fc": 2.3}, {"gene_id": "MYC", "log_fc": 1.8}],
    "top_down_genes": [{"gene_id": "TP53", "log_fc": -1.5}],
}

PCA_CTX = {
    "variance_pc1": 42.1,
    "variance_pc2": 18.3,
    "n_samples": 12,
    "n_genes": 5000,
    "sample_groups": ["Treated", "Control"],
    "group_separation": True,
}

UMAP_CTX = {
    "n_clusters": 3,
    "cluster_sizes": [4, 5, 3],
    "n_samples": 12,
    "sample_groups": ["Treated", "Control"],
}

HEATMAP_CTX = {
    "n_genes": 500,
    "n_samples": 12,
    "n_clusters": 3,
    "top_varying_genes": ["BRCA1", "MYC", "TP53"],
    "sample_groups": ["Treated", "Control"],
    "method": "ward",
}

ENRICHMENT_CTX = {
    "category": "GO:BP",
    "comparison_name": "Treated_vs_Control",
    "regulation": "UP",
    "top_terms": [
        {"name": "DNA repair", "pvalue": 0.001, "gene_count": 12},
        {"name": "cell cycle", "pvalue": 0.005, "gene_count": 8},
    ],
}


def test_build_volcano_prompt_contains_key_data():
    interp = LocalAIInterpreter()
    prompt = interp._build_volcano_prompt(VOLCANO_CTX)
    assert "127" in prompt
    assert "94" in prompt
    assert "BRCA1" in prompt
    assert "Treated_vs_Control" in prompt


def test_build_pca_prompt_contains_variance():
    interp = LocalAIInterpreter()
    prompt = interp._build_pca_prompt(PCA_CTX)
    assert "42.1" in prompt
    assert "18.3" in prompt
    assert "Treated" in prompt


def test_build_umap_prompt_contains_clusters():
    interp = LocalAIInterpreter()
    prompt = interp._build_umap_prompt(UMAP_CTX)
    assert "3" in prompt
    assert "Treated" in prompt


def test_build_heatmap_prompt_contains_genes():
    interp = LocalAIInterpreter()
    prompt = interp._build_heatmap_prompt(HEATMAP_CTX)
    assert "500" in prompt
    assert "BRCA1" in prompt


def test_build_enrichment_prompt_contains_terms():
    interp = LocalAIInterpreter()
    prompt = interp._build_enrichment_prompt(ENRICHMENT_CTX)
    assert "DNA repair" in prompt
    assert "GO:BP" in prompt


def test_interpret_chart_raises_on_unknown_type():
    interp = LocalAIInterpreter()
    import asyncio
    with pytest.raises(ValueError, match="Unknown chart_type"):
        asyncio.get_event_loop().run_until_complete(
            interp.interpret_chart("unknown_chart", {})
        )


@pytest.mark.asyncio
async def test_interpret_chart_calls_ollama_raw():
    interp = LocalAIInterpreter()
    with patch.object(interp, '_call_ollama_raw', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = "Plain English interpretation."
        result = await interp.interpret_chart("volcano", VOLCANO_CTX)
    mock_call.assert_called_once()
    assert result == "Plain English interpretation."
