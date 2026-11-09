"""
Figures of the OmicSkin editorial report — matplotlib, exported as SVG with text kept as text
(the page's @font-face renders it in Inter). Port of docs/omicskin-report-editorial/figs.py
(light theme) driven by the comparison's real data.

Every function returns an SVG string, or None when its input is missing. Settings are applied
through `rc_context`, so other matplotlib users in the process (LaTeX report) are unaffected.
No raster: dense scatter clouds are thinned deterministically instead of being rasterized.
"""

from __future__ import annotations

import io
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from app.services.cosmomics_grouping import clean_label  # noqa: E402

logger = logging.getLogger(__name__)

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "inter"

THEME = dict(
    ink="#1B2733",
    grey="#7D8B9C",
    light="#DCE3EA",
    faint="#F1F4F7",
    up="#FF5F5F",
    dn="#33A8E3",
    green="#34B596",
    teal="#158394",
    cmap="RdBu_r",
    dotcmap="viridis",
)
# Conditions other than test / control (PCA of the whole analysis).
OTHER_CONDITION_COLORS = ["#A8DCCF", "#47BDD7", "#5D5892", "#C9ADA7", "#FEC601", "#364B44"]

# Upper bound on vector points per scatter: beyond, non-significant points are thinned.
MAX_POINTS = 8000


@lru_cache(maxsize=1)
def _register_fonts() -> None:
    for path in sorted(FONT_DIR.glob("*.ttf")):
        try:
            font_manager.fontManager.addfont(str(path))
        except Exception as exc:  # noqa: BLE001 — metrics fall back to DejaVu
            logger.warning("Could not register font %s: %s", path.name, exc)


def _rc() -> dict[str, Any]:
    _register_fonts()
    t = THEME
    return {
        "font.family": ["Inter Variable", "DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": t["grey"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": t["ink"],
        "xtick.color": t["ink"],
        "ytick.color": t["ink"],
        "text.color": t["ink"],
        "axes.facecolor": "none",
        "figure.facecolor": "none",
        "savefig.facecolor": "none",
        "legend.labelcolor": t["ink"],
        "svg.fonttype": "none",
        "svg.hashsalt": "omicskin-editorial",  # stable ids → reproducible output
    }


def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _thin(mask: np.ndarray, keep: int, seed: int = 7) -> np.ndarray:
    """Boolean mask keeping at most `keep` of the True entries (deterministic). A figure that
    thins says so in its caption (`volcano_sampling`)."""
    idx = np.flatnonzero(mask)
    if len(idx) <= keep:
        return mask
    chosen = np.random.default_rng(seed).choice(idx, size=keep, replace=False)
    out = np.zeros_like(mask)
    out[chosen] = True
    return out


def _sample_budget(n_significant: int) -> int:
    """How many of the genes that are neither up- nor down-regulated the volcano draws."""
    return max(MAX_POINTS - n_significant, 1000)


def volcano_sampling(degs: Sequence[dict]) -> Optional[tuple[int, int]]:
    """(drawn, total) of the genes that are neither up- nor down-regulated, when the volcano
    draws only a sample of them; None when it draws every gene. The up- and down-regulated genes
    are always all drawn."""
    regulation = [(g.get("regulation") or "").upper() for g in degs]
    n_significant = sum(r in ("UP", "DOWN") for r in regulation)
    n_other = len(regulation) - n_significant
    drawn = _sample_budget(n_significant)
    return (drawn, n_other) if n_other > drawn else None


def condition_colors(conditions: Sequence[str], test: str, control: str) -> dict[str, str]:
    colors: dict[str, str] = {}
    others = iter(OTHER_CONDITION_COLORS * 4)
    for cond in conditions:
        if cond == control:
            colors[cond] = THEME["grey"]
        elif cond == test:
            colors[cond] = THEME["teal"]
        else:
            colors[cond] = next(others)
    return colors


def ordered_conditions(conditions: Sequence[str], test: str, control: str) -> list[str]:
    """Control first, test second, the others alphabetically."""
    uniq = list(dict.fromkeys(conditions))
    head = [c for c in (control, test) if c in uniq]
    return head + sorted(c for c in uniq if c not in head)


def pca_svg(
    pca: Optional[dict], sample_conditions: dict[str, str], test: str, control: str
) -> Optional[str]:
    points = (pca or {}).get("data") or []
    if len(points) < 2:
        return None
    with plt.rc_context(_rc()):
        conds = [sample_conditions.get(str(p.get("sample")), "Unknown") for p in points]
        order = ordered_conditions(conds, test, control)
        colors = condition_colors(order, test, control)
        fig, ax = plt.subplots(figsize=(5.2, 4))
        for cond in order:
            pts = [p for p, c in zip(points, conds) if c == cond]
            ax.scatter(
                [p.get("x", 0.0) for p in pts],
                [p.get("y", 0.0) for p in pts],
                s=80,
                color=colors[cond],
                edgecolor="none",
                label=cond,
                zorder=3,
            )
        ax.axhline(0, color=THEME["light"], lw=0.8, zorder=1)
        ax.axvline(0, color=THEME["light"], lw=0.8, zorder=1)
        var = (pca or {}).get("explained_variance") or []
        ax.set_xlabel(f"PC1 ({var[0] * 100:.1f} % of variance)" if var else "PC1")
        ax.set_ylabel(f"PC2 ({var[1] * 100:.1f} % of variance)" if len(var) > 1 else "PC2")
        ax.legend(frameon=False, fontsize=8, loc="best")
        return _svg(fig)


def volcano_svg(
    degs: list[dict],
    test: str,
    control: str,
    fdr: Optional[float] = None,
    min_lfc: Optional[float] = None,
    n_labels: int = 3,
) -> Optional[str]:
    """Volcano coloured by each gene's `regulation`, so it agrees with the DEG counts."""
    if not degs:
        return None
    lfc = np.array([g.get("log_fc") or 0.0 for g in degs], dtype=float)
    padj = np.array(
        [g.get("padj") if g.get("padj") is not None else 1.0 for g in degs], dtype=float
    )
    y = -np.log10(np.clip(padj, 1e-300, 1.0))
    reg = np.array([(g.get("regulation") or "").upper() for g in degs])
    up, dn = reg == "UP", reg == "DOWN"
    ns = _thin(~(up | dn), _sample_budget(int(up.sum() + dn.sum())))
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        ax.scatter(lfc[ns], y[ns], s=4, color=THEME["light"], alpha=0.7, lw=0)
        ax.scatter(
            lfc[up],
            y[up],
            s=6,
            color=THEME["up"],
            alpha=0.85,
            lw=0,
            label=f"Up-regulated ({int(up.sum())})",
        )
        ax.scatter(
            lfc[dn],
            y[dn],
            s=6,
            color=THEME["dn"],
            alpha=0.85,
            lw=0,
            label=f"Down-regulated ({int(dn.sum())})",
        )
        lim = max(4.0, float(np.nanpercentile(np.abs(lfc), 99.9)) * 1.1) if len(lfc) else 4.0
        ymax = float(np.nanmax(y)) if len(y) else 1.0
        ax.set_xlim(-lim, lim)
        ax.set_ylim(0, ymax * 1.08)
        # Top genes as a callout column on each side (up right, down left): no collisions.
        for mask, side in ((up, 1), (dn, -1)):
            idx = np.flatnonzero(mask)
            idx = idx[np.lexsort((-np.abs(lfc[idx]), padj[idx]))][:n_labels]
            idx = idx[np.argsort(-y[idx])]
            for k, i in enumerate(idx):
                name = str(degs[i].get("gene_name") or degs[i].get("gene_id") or "")
                ax.annotate(
                    name,
                    (lfc[i], y[i]),
                    xytext=(side * lim * 0.97, ymax * (1.0 - 0.09 * k)),
                    fontsize=7.5,
                    color=THEME["ink"],
                    style="italic",
                    ha="right" if side > 0 else "left",
                    va="center",
                    arrowprops=dict(arrowstyle="-", color=THEME["grey"], lw=0.5),
                )
        if min_lfc:
            for v in (min_lfc, -min_lfc):
                ax.axvline(v, color=THEME["grey"], lw=0.7, ls="--")
        if fdr:
            ax.axhline(-np.log10(fdr), color=THEME["grey"], lw=0.7, ls="--")
        ax.set_xlabel(f"log2 fold-change ({test} / {control})" if control else "log2 fold-change")
        ax.set_ylabel("−log10 adjusted p-value")
        ax.legend(
            frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.14)
        )
        return _svg(fig)


def dotplot_svg(pathways: list[dict], limit: int = 12) -> Optional[str]:
    rows = sorted(
        (p for p in pathways if p.get("padj") is not None), key=lambda p: p.get("padj") or 1.0
    )[:limit]
    if not rows:
        return None
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(4.4, 0.32 * len(rows) + 0.8))
        ypos = np.arange(len(rows))[::-1]
        nl = [-np.log10(max(r.get("padj") or 1e-300, 1e-300)) for r in rows]
        counts = np.array([max(int(r.get("gene_count") or 1), 1) for r in rows], dtype=float)
        sizes = 30 + 220 * (counts / counts.max())
        sc = ax.scatter(nl, ypos, s=sizes, c=nl, cmap=THEME["dotcmap"], edgecolor="none", zorder=3)
        labels = []
        for r in rows:
            name = clean_label(r.get("pathway_name")) or str(r.get("pathway_id") or "")
            name = name if len(name) <= 40 else name[:39].rstrip() + "…"
            labels.append(f"{name}  [{r.get('category') or '—'}]")
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlabel("−log10 adjusted p-value")
        ax.grid(axis="x", color=THEME["light"], lw=0.5, zorder=0)
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("−log10 padj", fontsize=8)
        cb.outline.set_visible(False)
        return _svg(fig)


def _expression_frame(vst: bytes) -> pd.DataFrame:
    df = pd.read_parquet(io.BytesIO(vst))
    lc = {c.lower(): c for c in df.columns}
    gene_col = next((lc[c] for c in ("gene_id", "gene", "symbol", "gene_name") if c in lc), None)
    if gene_col:
        df = df.set_index(gene_col)
    return df.select_dtypes(include=[np.number])


def heatmap_genes(degs: list[dict], n: int = 30) -> list[dict]:
    """Up to n DEGs, half up / half down, most significant first."""

    def ranked(direction: str) -> list[dict]:
        sel = [g for g in degs if (g.get("regulation") or "").upper() == direction]
        return sorted(
            sel,
            key=lambda g: (
                g.get("padj") if g.get("padj") is not None else 1.0,
                -abs(g.get("log_fc") or 0),
            ),
        )

    ups, downs = ranked("UP"), ranked("DOWN")
    half = n // 2
    pick_up = ups[: max(half, n - len(downs[:half]))]
    pick_dn = downs[: n - len(pick_up)]
    return pick_up + pick_dn


def heatmap_svg(
    vst: Optional[bytes],
    degs: list[dict],
    sample_conditions: dict[str, str],
    test: str,
    control: str,
    n: int = 30,
) -> Optional[str]:
    """VST expression of the top DEGs, centred on the control mean, samples grouped by condition."""
    if not vst or not degs:
        return None
    try:
        df = _expression_frame(vst)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Heatmap: unreadable expression matrix: %s", exc)
        return None
    df.index = [str(i).split(".")[0] for i in df.index]
    df = df[~df.index.duplicated()]
    genes = [
        g for g in heatmap_genes(degs, n) if str(g.get("gene_id") or "").split(".")[0] in df.index
    ]
    if len(genes) < 2:
        return None
    ids = [str(g["gene_id"]).split(".")[0] for g in genes]
    names = [str(g.get("gene_name") or g["gene_id"]) for g in genes]

    samples = list(df.columns)
    conds = {s: sample_conditions.get(str(s)) for s in samples}
    keep = [c for c in (control, test) if c in conds.values()]
    if keep:
        samples = [s for c in keep for s in samples if conds[s] == c]
    groups = [conds.get(s) or "" for s in samples]
    mat = df.loc[ids, samples].to_numpy(dtype=float)
    ctrl_cols = [i for i, g in enumerate(groups) if g == control]
    if ctrl_cols:
        centre = mat[:, ctrl_cols].mean(axis=1, keepdims=True)
    else:
        centre = mat.mean(axis=1, keepdims=True)
    z = mat - centre
    lim = float(min(3.0, max(1.0, np.nanpercentile(np.abs(z), 98))))

    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(6.0, 0.2 * len(ids) + 0.6))
        # pcolormesh, not imshow: imshow embeds a PNG in the SVG; these are vector cells.
        im = ax.pcolormesh(z, cmap=THEME["cmap"], vmin=-lim, vmax=lim, edgecolors="none")
        ax.set_ylim(len(ids), 0)
        ax.set_yticks(np.arange(len(ids)) + 0.5)
        ax.set_yticklabels(names, fontsize=7.5, style="italic")
        bounds, centres, labels = [], [], []
        start = 0
        for i in range(1, len(groups) + 1):
            if i == len(groups) or groups[i] != groups[start]:
                centres.append((start + i) / 2)
                labels.append(groups[start] or "—")
                if i < len(groups):
                    bounds.append(i)
                start = i
        ax.set_xticks(centres)
        ax.set_xticklabels(labels, fontsize=8)
        for b in bounds:
            ax.axvline(b, color="white", lw=2)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label(
            (
                "log2 expression (VST), centred on control"
                if ctrl_cols
                else "log2 expression (VST), centred"
            ),
            fontsize=8,
        )
        cb.outline.set_visible(False)
        return _svg(fig)
