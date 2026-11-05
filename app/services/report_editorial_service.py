"""
ReportEditorialService — the "OmicSkin Éditorial" comparison report (variant D): an HTML page set
rendered to PDF by headless Chromium (Playwright).

Data collection reuses the LaTeX report service (`collect_comparison_data`, `_load_branding`) and
the CosmOmics nova (`compute_nova`, the code path of the `cosmomics-nova` endpoint); the content is
built by `report_editorial_content`, the figures by `report_editorial_figures`.

Chromium rather than WeasyPrint: the pages rely on CSS grid, inline SVG with `paint-order`
halos and patterns, which WeasyPrint does not render faithfully. The LaTeX report is untouched.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import jinja2
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.models import AIInterpretation, CosmeticInterpretation, Dataset
from app.services import report_editorial_figures as figs
from app.services.cosmomics_nova import (
    NOVA_MODEL_TAG,
    TermHit,
    compute_nova,
    default_claim_subgroups,
    term_gene_map,
)
from app.services.cosmomics_nova_svg import render_nova_svg
from app.services.omicskin_score import score_claims
from app.services.report_common import merge_cover_info
from app.services.report_editorial_content import (
    EditorialInputs,
    analysis_thresholds,
    build_editorial_context,
    looks_english,
    sanitize_condition,
)
from app.services.report_latex_service import report_latex_service

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = APP_DIR / "templates" / "report_editorial"
FONT_DIR = APP_DIR / "assets" / "fonts" / "inter"
LOGO_PATH = APP_DIR / "assets" / "report_editorial" / "logo_scilicium.png"

PDF_TIMEOUT_MS = 90_000
AI_TIMEOUT_S = 120

_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=True,
)


def render_html(context: dict[str, Any], font_uri: Optional[str] = None) -> str:
    """The report as one HTML document (fonts referenced by file:// URL)."""
    return _env.get_template("report.html.j2").render(
        font_uri=font_uri or FONT_DIR.as_uri(), **context
    )


def data_uri(path: Optional[Path]) -> Optional[str]:
    if not path or not Path(path).exists():
        return None
    ext = Path(path).suffix.lower().lstrip(".") or "png"
    mime = {"jpg": "jpeg", "svg": "svg+xml"}.get(ext, ext)
    return f"data:image/{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


_OVERFLOW_JS = """() => Array.from(document.querySelectorAll('.page'))
  .map((el, i) => (el.scrollHeight > el.clientHeight + 2 ? i + 1 : 0)).filter(Boolean)"""


async def html_to_pdf(html: str, work_dir: str) -> tuple[bytes, list[int]]:
    """Print the HTML with headless Chromium. Returns the PDF and the pages whose content
    overflows the A4 box (those pages clip their content; logged by the caller)."""
    from playwright.async_api import async_playwright

    path = Path(work_dir) / "report.html"
    path.write_text(html, encoding="utf-8")
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            page = await browser.new_page()
            await page.goto(path.as_uri(), wait_until="load", timeout=PDF_TIMEOUT_MS)
            await page.evaluate("() => document.fonts.ready.then(() => true)")
            overflow = await page.evaluate(_OVERFLOW_JS)
            pdf = await page.pdf(prefer_css_page_size=True, print_background=True)
        finally:
            await browser.close()
    return pdf, list(overflow or [])


def chromium_available() -> bool:
    """True when Playwright and its Chromium build are installed (used to skip PDF tests)."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            return os.path.exists(p.chromium.executable_path)
    except Exception:  # noqa: BLE001
        return False


def split_comparison(name: str) -> tuple[str, str]:
    """("KO", "WT") from "KO_vs_WT"; (name, "") when there is no "vs"."""
    parts = re.split(r"[_\s]+vs[_\s]+", name or "", maxsplit=1, flags=re.I)
    return (parts[0], parts[1]) if len(parts) == 2 else (name, "")


def hits_to_enrichment(hits: list[TermHit]) -> tuple[list[dict], list[dict]]:
    """Enrichment rows (up, down) rebuilt from the nova's directional hits — used when the
    comparison has no analysis-linked ENRICHMENT dataset (rows stored on the DEG dataset)."""
    up, down = [], []
    for h in hits:
        if h.p is None:  # no adjusted p-value: left out of the nova, left out of the plots
            continue
        row = {
            "pathway_id": h.term_id,
            "pathway_name": h.label,
            "category": h.category,
            "padj": h.p,
            "pvalue": h.p,
            "gene_count": h.gene_count,
            "regulation": h.obs.upper(),
        }
        (up if h.obs == "up" else down).append(row)
    return up, down


def build_figures(
    degs: list[dict],
    enr_up: list[dict],
    enr_down: list[dict],
    expression: dict[str, Any],
    sample_conditions: dict[str, str],
    test: str,
    control: str,
    params: dict[str, Any],
) -> dict[str, Optional[str]]:
    fdr, min_lfc = analysis_thresholds(params)
    out: dict[str, Optional[str]] = {}
    for key, fn in (
        ("pca", lambda: figs.pca_svg(expression.get("pca"), sample_conditions, test, control)),
        ("volcano", lambda: figs.volcano_svg(degs, test, control, fdr, min_lfc)),
        ("dot_up", lambda: figs.dotplot_svg(enr_up)),
        ("dot_down", lambda: figs.dotplot_svg(enr_down)),
        (
            "heatmap",
            lambda: figs.heatmap_svg(expression.get("vst"), degs, sample_conditions, test, control),
        ),
    ):
        try:
            out[key] = fn()
        except Exception as exc:  # noqa: BLE001 — one failed figure must not sink the report
            logger.warning("Editorial report: %s figure failed: %s", key, exc)
            out[key] = None
    return out


class ReportEditorialService:
    async def _comparison_ai(
        self,
        db: AsyncSession,
        dataset_id: UUID,
        comparison_name: str,
        cached_text: Optional[str],
        generate_ai: bool,
        demo: bool,
    ) -> tuple[Optional[str], Optional[str]]:
        """English comparison interpretation: the cached row when it is English, else (on
        request, never for demo accounts) a fresh English text that is NOT stored — the
        user's cached interpretation stays as it is."""
        model = None
        if cached_text:
            model = await db.scalar(
                select(AIInterpretation.model).where(
                    AIInterpretation.dataset_id == dataset_id,
                    AIInterpretation.comparison_name == comparison_name,
                )
            )
        if cached_text and looks_english(cached_text):
            return cached_text, model
        if generate_ai and not demo:
            try:
                from app.services.ai_interpreter import generate_comparison_text

                res = await asyncio.wait_for(
                    generate_comparison_text(db, dataset_id, None, comparison_name, "en"),
                    timeout=AI_TIMEOUT_S,
                )
                if res:
                    return res
            except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
                logger.info("English AI interpretation skipped for %s: %s", comparison_name, exc)
        return cached_text, model  # non-English text: the content layer shows a notice

    async def _nova_ai(
        self,
        db: AsyncSession,
        dataset_id: UUID,
        comparison_name: str,
        nova: dict[str, Any],
        generate_ai: bool,
        demo: bool,
    ) -> tuple[Optional[str], Optional[str]]:
        row = (
            await db.execute(
                select(CosmeticInterpretation).where(
                    and_(
                        CosmeticInterpretation.dataset_id == dataset_id,
                        CosmeticInterpretation.comparison_name == comparison_name,
                    )
                )
            )
        ).scalar_one_or_none()
        if row and (row.model or "").startswith(NOVA_MODEL_TAG):
            return row.interpretation, (row.model or "").removeprefix(NOVA_MODEL_TAG)
        if generate_ai and not demo and nova.get("claims"):
            try:
                from app.services.ai_interpreter import LocalAIInterpreter

                interpreter = LocalAIInterpreter()
                text = await asyncio.wait_for(
                    interpreter.interpret_cosmomics(nova, comparison_name), timeout=AI_TIMEOUT_S
                )
                return text, interpreter.model
            except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
                logger.info("Nova AI reading skipped for %s: %s", comparison_name, exc)
        return None, None

    async def collect(
        self,
        db: AsyncSession,
        dataset_id: UUID,
        comparison_name: str,
        work_dir: str,
        requested_by=None,
        cover_info: Optional[dict] = None,
        generate_ai: bool = False,
        demo_watermark: bool = False,
        use_ai: bool = True,
        use_branding: bool = True,
    ) -> EditorialInputs:
        """The report's inputs.

        use_ai=False: no AI at all — no LLM call (whatever `generate_ai` says), no stored
        interpretation read, and the content layer prints no AI label.
        use_branding=False: the requester's saved branding and cover data are left out (a
        report requested through a service token).
        """
        data = await report_latex_service.collect_comparison_data(
            db,
            dataset_id,
            comparison_name,
            work_dir,
            generate_ai=False,
            make_figures=False,
            include_ai=use_ai,
        )
        comp = data["comparisons"][0]
        branding = (
            await report_latex_service._load_branding(db, requested_by, work_dir)
            if use_branding
            else None
        )
        cover = merge_cover_info(branding, cover_info)

        dataset = await db.get(Dataset, dataset_id)
        nova, hits = await compute_nova(db, dataset, comparison_name)
        mapping = default_claim_subgroups()
        scores = score_claims(nova)
        term_genes = term_gene_map(hits, mapping)

        enr_up, enr_down = comp.get("enr_up") or [], comp.get("enr_down") or []
        if not enr_up and not enr_down:
            enr_up, enr_down = hits_to_enrichment(hits)

        test, control = comp.get("test") or comparison_name, comp.get("control") or ""
        if not control:
            test, control = split_comparison(comparison_name)
        test, control = sanitize_condition(test), sanitize_condition(control)

        expression = data.get("expression") or {}
        qc_rows = data.get("qc_rows") or []
        sample_conditions = {
            k: sanitize_condition(v) for k, v in (expression.get("sample_conditions") or {}).items()
        }
        if not sample_conditions:
            sample_conditions = {
                str(r["sample"]): sanitize_condition(r.get("condition")) for r in qc_rows
            }
        qc_rows = [{**r, "condition": sanitize_condition(r.get("condition"))} for r in qc_rows]
        params = data.get("params") or {}

        figures = await asyncio.to_thread(
            build_figures,
            comp.get("degs") or [],
            enr_up,
            enr_down,
            expression,
            sample_conditions,
            test,
            control,
            params,
        )
        center = test if 0 < len(test) <= 8 else str(nova.get("condition") or "Active")
        nova_light = render_nova_svg(nova, "light", id_prefix="nova-light", center_label=center)
        nova_dark = render_nova_svg(nova, "dark", id_prefix="nova-dark", center_label=center)

        cmp_text = cmp_model = nova_text = nova_model = None
        if use_ai:
            cmp_text, cmp_model = await self._comparison_ai(
                db,
                dataset_id,
                comparison_name,
                comp.get("ai_interpretation"),
                generate_ai,
                demo_watermark,
            )
            nova_text, nova_model = await self._nova_ai(
                db, dataset_id, comparison_name, nova, generate_ai, demo_watermark
            )

        project = data.get("project")
        analysis = data.get("analysis")
        logo = None
        if branding and branding.get("logo_file"):
            logo = data_uri(Path(work_dir) / branding["logo_file"])
        return EditorialInputs(
            comparison_name=comparison_name,
            test=test,
            control=control,
            project_name=cover.get("project_name") or getattr(project, "name", "") or "",
            degs=comp.get("degs") or [],
            n_up=int(comp.get("n_up") or 0),
            n_down=int(comp.get("n_down") or 0),
            enr_up=enr_up,
            enr_down=enr_down,
            qc_rows=qc_rows,
            sample_conditions=sample_conditions,
            params=params,
            has_analysis=analysis is not None,
            analysis_label=(f"{analysis.name} ({str(analysis.id)[:8]})" if analysis else None),
            nova=nova,
            scores=scores,
            term_genes=term_genes,
            figures=figures,
            nova_light=nova_light,
            nova_dark=nova_dark,
            comparison_ai=cmp_text,
            comparison_ai_model=cmp_model,
            nova_ai=nova_text,
            nova_ai_model=nova_model,
            cover_info=cover,
            branding=branding,
            logo_data_uri=logo or data_uri(LOGO_PATH),
            referential_version=mapping.version,
            app_version=settings.APP_VERSION,
            generated_at=datetime.now(timezone.utc),
            demo_watermark=demo_watermark,
            use_ai=use_ai,
        )

    async def render_comparison_pdf(
        self,
        db: AsyncSession,
        dataset_id: UUID,
        comparison_name: str,
        requested_by=None,
        cover_info: Optional[dict] = None,
        generate_ai: bool = False,
        demo_watermark: bool = False,
        use_ai: bool = True,
        use_branding: bool = True,
    ) -> bytes:
        with tempfile.TemporaryDirectory(prefix="report_editorial_") as work_dir:
            inputs = await self.collect(
                db,
                dataset_id,
                comparison_name,
                work_dir,
                requested_by,
                cover_info,
                generate_ai,
                demo_watermark,
                use_ai=use_ai,
                use_branding=use_branding,
            )
            html = render_html(build_editorial_context(inputs))
            pdf, overflow = await html_to_pdf(html, work_dir)
            if overflow:
                logger.warning(
                    "Editorial report %s / %s: content clipped on page(s) %s",
                    dataset_id,
                    comparison_name,
                    overflow,
                )
            return pdf


report_editorial_service = ReportEditorialService()
