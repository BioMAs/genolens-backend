from unittest.mock import MagicMock
from app.services.report_service import ReportService


def svc():
    return ReportService()


def test_fig_to_b64_returns_nonempty_string():
    import matplotlib.pyplot as plt
    service = svc()
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    result = service._fig_to_b64(fig)
    assert isinstance(result, str) and len(result) > 100


def test_generate_volcano_plot_returns_b64():
    service = svc()
    genes = [
        MagicMock(log_fc=v, padj=p)
        for v, p in [(2.5, 0.001), (-1.8, 0.01), (0.2, 0.8), (3.1, 1e-20)]
    ]
    result = service._generate_volcano_plot(genes, "TestComp")
    assert result is not None and len(result) > 100


def test_generate_volcano_plot_empty_returns_none():
    service = svc()
    assert service._generate_volcano_plot([], "Empty") is None


def test_generate_enrichment_dotplot_top20():
    service = svc()
    pathways = [
        MagicMock(pathway_name=f"pathway_{i}", padj=0.001 * i, gene_count=10)
        for i in range(1, 35)
    ]
    result = service._generate_enrichment_dotplot(pathways, "TestComp")
    assert result is not None


def test_extract_qc_data_parses_metadata():
    service = svc()
    ds = MagicMock()
    ds.dataset_metadata = {
        "qc_report": {
            "SampleA": {
                "total_reads": 20_000_000,
                "mapped_reads": 19_000_000,
                "mapping_rate": 95.0,
                "detected_genes": 18_000,
            }
        }
    }
    rows = service._extract_qc_data([ds])
    assert len(rows) == 1
    assert rows[0]["sample"] == "SampleA"
    assert rows[0]["total_reads"] == 20_000_000


def test_extract_qc_data_empty_metadata():
    service = svc()
    ds = MagicMock()
    ds.dataset_metadata = {}
    assert service._extract_qc_data([ds]) == []
