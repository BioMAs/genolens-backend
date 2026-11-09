"""
Small SVG / HTML pieces of the OmicSkin editorial report (port of
docs/omicskin-report-editorial/common.py): score ring, dose ladder, sub-group bars, gene chips,
plus the clean-up applied to every SVG inlined in the report.

Every function returns `markupsafe.Markup`; data-derived strings (sub-group and gene names) are
escaped on the way in.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Optional, Sequence

from markupsafe import Markup, escape

from app.services.cosmomics_nova import subgroup_side
from app.services.cosmomics_nova_svg import balance_color

INK = "#1B2733"
TRACK = "#EEF2F5"
UP, DOWN, AGAINST = "#34B596", "#33A8E3", "#FF5F5F"


def ring(
    value: int,
    color: str,
    size: int = 120,
    stroke: int = 10,
    track: str = TRACK,
    text_color: Optional[str] = None,
) -> Markup:
    """Circular gauge, value 0–100, the number in the middle."""
    r = (size - stroke) / 2
    circumference = 2 * math.pi * r
    v = max(0, min(100, int(value)))
    off = circumference * (1 - v / 100)
    tc = text_color or color
    fs = size * 0.3
    c = size / 2
    return Markup(
        f'<svg class="ring" width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="{track}" stroke-width="{stroke}"/>'
        f'<circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-dasharray="{circumference:.2f}" '
        f'stroke-dashoffset="{off:.2f}" transform="rotate(-90 {c} {c})"/>'
        f'<text x="{c}" y="{c + fs * 0.36}" text-anchor="middle" font-family="Inter Display" '
        f'font-weight="700" font-size="{fs}" fill="{tc}" letter-spacing="-1">{v}</text></svg>'
    )


def dose_ladder(
    points: Sequence[tuple[str, int]],
    color: str,
    w: int = 150,
    h: int = 70,
    track: str = TRACK,
    ink: str = INK,
) -> Markup:
    """One bar per dose (label, score), lightest to darkest. Room is kept above the bars for
    the value label, so a score near 100 is not clipped."""
    n = max(len(points), 1)
    gap = 8
    bw = w / n - gap
    top = 13  # value label headroom
    avail = h - 18 - top
    o = [f'<svg class="dose" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    for i, (label, value) in enumerate(points):
        v = max(0, min(100, int(value)))
        x = i * (bw + gap)
        bh = avail * v / 100
        opacity = 0.45 + (0.55 * i / (n - 1) if n > 1 else 0.55)
        o.append(
            f'<rect x="{x:.1f}" y="{h - 18 - avail:.1f}" width="{bw:.1f}" height="{avail:.1f}" '
            f'rx="3" fill="{track}"/>'
        )
        o.append(
            f'<rect x="{x:.1f}" y="{h - 18 - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" '
            f'fill="{color}" opacity="{opacity:.2f}"/>'
        )
        o.append(
            f'<text x="{x + bw / 2:.1f}" y="{h - 4}" text-anchor="middle" font-family="Inter" '
            f'font-size="9" fill="{ink}" opacity=".7">{escape(label)}</text>'
        )
        o.append(
            f'<text x="{x + bw / 2:.1f}" y="{h - 21 - bh:.1f}" text-anchor="middle" '
            f'font-family="Inter" font-weight="600" font-size="9.5" fill="{ink}">{v}</text>'
        )
    o.append("</svg>")
    return Markup("".join(o))


def _ellipsis(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max(max_chars - 1, 1)].rstrip() + "…"


def subgroup_bars(
    subgroups: Iterable[dict],
    w: int = 300,
    row: int = 22,
    maxnl: float = 12,
    ink: str = INK,
    track: str = TRACK,
    font_size: float = 10,
    limit: Optional[int] = None,
) -> Markup:
    """One row per sub-group: name (greyed only when it goes against the claim — a balanced
    sub-group is not contrary), a bar of its significance (−log10 p, unsigned Stouffer, capped at
    `maxnl`) coloured by its balance, and 10⁻ⁿ."""
    subs = sorted(subgroups, key=lambda s: -(s.get("nl") or 0))
    if limit:
        subs = subs[:limit]
    h = row * len(subs)
    lw = w * 0.5
    max_chars = int(lw / (font_size * 0.56))
    o = [f'<svg class="subgroups" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    for i, sg in enumerate(subs):
        y = i * row
        nl = float(sg.get("nl") or 0)
        bl = (w - lw - 40) * min(nl, maxnl) / maxnl
        col = balance_color(sg.get("balance") or 0)
        name = escape(_ellipsis(str(sg.get("name") or ""), max_chars))
        opacity = 0.55 if subgroup_side(sg) == "against" else 1
        o.append(
            f'<text x="0" y="{y + row * 0.68:.1f}" font-family="Inter" font-size="{font_size}" '
            f'fill="{ink}" opacity="{opacity}">{name}</text>'
        )
        o.append(
            f'<rect x="{lw}" y="{y + row * 0.3:.1f}" width="{w - lw - 40}" '
            f'height="{row * 0.4:.1f}" '
            f'rx="{row * 0.2:.1f}" fill="{track}"/>'
        )
        o.append(
            f'<rect x="{lw}" y="{y + row * 0.3:.1f}" width="{bl:.1f}" height="{row * 0.4:.1f}" '
            f'rx="{row * 0.2:.1f}" fill="{col}"/>'
        )
        o.append(
            f'<text x="{w - 36}" y="{y + row * 0.68:.1f}" font-family="Inter" '
            f'font-size="{font_size - 1}" fill="{ink}" opacity=".7">10<tspan dy="-4" '
            f'font-size="{font_size - 3}">−{nl:.0f}</tspan></text>'
        )
    o.append("</svg>")
    return Markup("".join(o))


def gene_chips(
    genes: Iterable[tuple[str, float]], up: str = UP, down: str = DOWN, ink: str = INK
) -> Markup:
    """Pill per gene: italic symbol, arrow and |log2 FC| in the direction colour."""
    out = []
    for name, lfc in genes:
        col = up if lfc > 0 else down
        arrow = "↑" if lfc > 0 else "↓"
        out.append(
            f'<span class="chip" style="border-color:{col};color:{ink}"><i>{escape(name)}</i> '
            f'<b style="color:{col}">{arrow}{abs(lfc):.1f}</b></span>'
        )
    return Markup("".join(out))


_XML_DECL = re.compile(r"<\?xml[^>]*\?>\s*|<!DOCTYPE[^>]*>\s*", re.I)
_METADATA = re.compile(r"<metadata>.*?</metadata>\s*", re.S)
_STYLE = re.compile(r"<style[^>]*>.*?</style>\s*", re.S)
_ID = re.compile(r'\bid="([^"]+)"')
_URL_REF = re.compile(r"url\(#([^)]+)\)")
_HREF = re.compile(r'((?:xlink:)?href)="#([^"]+)"')
_ROOT_SIZE = re.compile(r'\s(width|height)="[^"]*"')


def clean_inline_svg(svg: str, prefix: str, style: str = "") -> Markup:
    """Make an SVG document safe to inline next to others in one HTML page.

    Drops the XML prolog, metadata and global <style> blocks (matplotlib's `*{…}` rule would
    leak to the whole page), prefixes every id and every reference to it — `url(#…)` and
    `href="#…"` — and lets CSS size the root element.
    """
    s = _XML_DECL.sub("", svg)
    s = _METADATA.sub("", s)
    s = _STYLE.sub("", s)
    s = _ID.sub(lambda m: f'id="{prefix}-{m.group(1)}"', s)
    s = _URL_REF.sub(lambda m: f"url(#{prefix}-{m.group(1)})", s)
    s = _HREF.sub(lambda m: f'{m.group(1)}="#{prefix}-{m.group(2)}"', s)
    start = s.find("<svg")
    if start < 0:
        return Markup("")
    s = s[start:]
    head_end = s.find(">")
    head = _ROOT_SIZE.sub("", s[:head_end])
    s = head + s[head_end:]
    return Markup(s.replace("<svg", f'<svg style="display:block;{style}"', 1))
