"""
Server-side SVG of the CosmOmics Pathway Nova (report covers, PDF pages, `…/cosmomics-nova.svg`).

Port of the reference generator docs/omicskin-report-editorial/cosmomics_pathway_nova.py (`nova`,
`blocks`, `legend`) and of the dark variant derived by nova_wheel.py, driven by the payload of
`cosmomics_nova.build_nova` instead of mock data:

  * one sector per claim, width ∝ importance (Σ −log10 p of its terms), hatched part = share of
    the evidence against the claim; a grey sector for significant sub-groups linked to no claim;
  * one bubble per pathway sub-group: distance to the centre = Stouffer −log10 p (unsigned),
    colour = balance for → against (a balanced sub-group is grey, outside the hatched part), size
    = |Δ pro − anti|; small dots = the individual ontology terms;
  * the claim labels stand outside the wheel, kept apart from one another and off the wheel by the
    app's own rule; a label pushed far from its sector is tied back to it by a thin line, and the
    frame grows to hold every label.

Only what the payload carries is drawn — claim names, sub-group names and term dots — so the
claim referential never leaves the server, and no dot stands for anything but a term or a
sub-group. A demo payload (`provenance.demo`) is marked DEMO. Output is deterministic (seeded
jitter) and every `id` is prefixed, so several novas can be inlined in one HTML document.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

from app.services.cosmomics_nova import (
    ZERO_AGAINST_NOTE,
    claim_state_of,
    p_source_of,
    subgroup_side,
)

Theme = Literal["light", "dark"]

FONT = "Inter, 'Helvetica Neue', Arial, sans-serif"

# Same colours as the app (frontend/src/components/cosmomics/novaPalette.ts).
CLAIM_COLORS: dict[str, tuple[str, str]] = {
    "Firming": ("#158394", "#158394"),
    "Barrier": ("#47BDD7", "#2A6478"),
    "Regeneration": ("#FEC601", "#CA9A00"),
    "Anti-ageing": ("#5D5892", "#5D5892"),
    "Soothing": ("#C9ADA7", "#9C7B74"),
    "Antioxidant": ("#364B44", "#364B44"),
    "Energising": ("#33A8E3", "#0086BE"),
    "Anti-wrinkles": ("#5D309C", "#5D309C"),
}
_FALLBACK_COLORS: list[tuple[str, str]] = [
    ("#30415F", "#30415F"),
    ("#B78A00", "#8A6500"),
    ("#006095", "#006095"),
    ("#C6B7DC", "#5D5892"),
    ("#2E4857", "#2E4857"),
    ("#66B8E1", "#00659B"),
    ("#825C00", "#825C00"),
]

THEMES: dict[str, dict[str, Any]] = {
    "light": {
        "bg": "#FFFFFF",
        "green": "#34B596",
        "coral": "#FF5F5F",
        "muted": "#384C65",
        "dim": "#7D8B9C",
        "ink": "#3A4150",
        "halo": "#FFFFFF",
        "hatch_fill": "#EEF1F4",
        "hatch_line": "#B9C3CC",
        "ring": "#3A4150",
        "ring_label": "#3A4150",
        "core_fill": "#0B2A3E",
        "core_text": "#FFFFFF",
        "subgroup_label": "#0B2A3E",
        "wedge_opacity": 0.14,
    },
    "dark": {
        "bg": "#0F1A24",
        "green": "#42E2BA",
        "coral": "#FF7A7A",
        "muted": "#B8C4D0",
        "dim": "#8A98A8",
        "ink": "#E6EEF5",
        "halo": "#0F1A24",
        "hatch_fill": "#1C2A38",
        "hatch_line": "#4A5B6E",
        "ring": "#9FB0C2",
        "ring_label": "#C5D2DF",
        "core_fill": "#42E2BA",
        "core_text": "#0B2A3E",
        "subgroup_label": "#F2F6FA",
        "wedge_opacity": 0.22,
    },
}

AGAINST, AGAINST_TEXT = "#9AA8B6", "#7D8B9C"

W = 1000
CX, CY, R = 500, 555, 330
# Wheel only: the reference crop (nova_wheel.py), around the wheel and its claim labels.
WHEEL_VIEWBOX = "0 130 1000 850"
_CROP_X, _CROP_Y, _CROP_W, _CROP_H = (int(v) for v in WHEEL_VIEWBOX.split())

# A claim label's box: (x0, y0, x1, y1). The labels follow the app's rule (novaLayout.ts), so the
# wheel drawn here and the one drawn in the app agree.
Box = tuple[float, float, float, float]
LABEL_CLEARANCE = 4.0  # px a label keeps from the wheel (R + 4) and from the labels before it
MAX_PUSHES = 40  # pushes before a label is left where the last one put it
LEADER_MIN_SHIFT = 12.0  # a label that moved further than this (px, vertically) gets a leader line
FRAME_MARGIN = 8  # px the frame keeps around every label box


def _djb2(value: str) -> int:
    h = 5381
    for ch in value:
        h = ((h << 5) + h + ord(ch)) & 0xFFFFFFFF
    return h


def claim_color(name: str) -> tuple[str, str]:
    """(sector fill, text colour) of a claim — fixed per name, as in the app."""
    return CLAIM_COLORS.get(name) or _FALLBACK_COLORS[_djb2(name) % len(_FALLBACK_COLORS)]


def balance_color(balance: float) -> str:
    """Balance pro/anti −1..+1 → coral .. grey .. green (reference ramp and rounding)."""
    t = (max(-1.0, min(1.0, float(balance))) + 1) / 2
    a, m, z = (0xFF, 0x5F, 0x5F), (0xC9, 0xD1, 0xD9), (0x34, 0xB5, 0x96)
    lo, hi, f = (a, m, t / 0.5) if t < 0.5 else (m, z, (t - 0.5) / 0.5)
    return "#%02X%02X%02X" % tuple(round(lo[i] + (hi[i] - lo[i]) * f) for i in range(3))


def _esc(s: Any) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _text(x, y, s, size, fill, anchor="start", weight=400, extra="") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}" {extra}>{_esc(s)}</text>'
    )


def _dot(x, y, r, col, op, extra="") -> str:
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.2f}" fill="{col}" opacity="{op:.2f}" {extra}/>'
    )


def _pt(r: float, a: float) -> tuple[float, float]:
    t = math.radians(a)
    return CX + r * math.cos(t), CY + r * math.sin(t)


def _wedge(t1, t2, r, fill, op) -> str:
    if t2 - t1 >= 359.9:  # a full turn: an arc path would collapse onto itself
        return f'<circle cx="{CX}" cy="{CY}" r="{r:.1f}" fill="{fill}" opacity="{op}"/>'
    (x1, y1), (x2, y2) = _pt(r, t1), _pt(r, t2)
    large = 1 if (t2 - t1) > 180 else 0
    return (
        f'<path d="M{CX},{CY} L{x1:.1f},{y1:.1f} A{r:.1f},{r:.1f} 0 {large} 1 {x2:.1f},{y2:.1f} Z" '
        f'fill="{fill}" opacity="{op}"/>'
    )


def _arc(t1, t2, r, col, w, extra="") -> str:
    if t2 - t1 >= 359.9:
        return (
            f'<circle cx="{CX}" cy="{CY}" r="{r:.1f}" fill="none" stroke="{col}" '
            f'stroke-width="{w}" {extra}/>'
        )
    (x1, y1), (x2, y2) = _pt(r, t1), _pt(r, t2)
    large = 1 if (t2 - t1) > 180 else 0
    return (
        f'<path d="M{x1:.1f},{y1:.1f} A{r:.1f},{r:.1f} 0 {large} 1 {x2:.1f},{y2:.1f}" fill="none" '
        f'stroke="{col}" stroke-width="{w}" {extra}/>'
    )


def _bsize(d: float) -> float:
    return 3.2 + 1.9 * math.sqrt(abs(d))


AXIS_TITLE = "−log10 combined p (Stouffer, unsigned)"


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _sector_note(claim: dict[str, Any]) -> str:
    """"3 sub-groups for · 0 against": both counts are always written, a zero included."""
    n_for, n_against = int(claim.get("n_for") or 0), int(claim.get("n_against") or 0)
    n_balanced = claim.get("n_balanced")
    if n_balanced is None:  # a payload that predates the field
        n_balanced = sum(1 for s in claim.get("subgroups") or [] if subgroup_side(s) == "balanced")
    note = f"{_plural(n_for, 'sub-group')} for · {n_against} against"
    return note + (f" · {int(n_balanced)} balanced" if n_balanced else "")


@dataclass(frozen=True)
class _Label:
    """A claim label where the reference puts it, before it is kept off its neighbours and the
    wheel."""

    angle: float  # mid angle of its sector
    x: float
    y: float  # baseline of the title
    anchor: str
    title: str
    notes: list[str]  # one line, or two when the note would spill over the wheel
    title_fill: str
    note_fill: str
    box: Box  # what the text covers: 12 px per title character, 6.3 per note character


def _reference_label(
    angle: float, title: str, note: str, title_fill: str, note_fill: str
) -> _Label:
    x, y = _pt(R + 24, angle)
    ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    anchor = "middle" if abs(ca) < 0.3 else ("start" if ca > 0 else "end")
    y += 8 if sa > 0.3 else (-2 if sa < -0.3 else 6)
    title_w = len(title) * 12.0
    w = max(title_w, len(note) * 6.3)
    if anchor == "start" and x + w > 990:
        x = 990 - w
    if anchor == "end" and x - w < 10:
        x = 10 + w
    notes, note_w = [note], len(note) * 6.3
    if anchor != "middle" and " · " in note:
        edge = (CX + R + 6) if anchor == "start" else (CX - R - 6)
        if (anchor == "start" and x < edge) or (anchor == "end" and x > edge):
            # the second line would spill over the wheel: break it in two
            notes = note.split(" · ")
            note_w = max(len(line) for line in notes) * 6.3
            x = max(edge, x) if anchor == "start" else min(edge, x)
            widest = max(title_w, note_w)
            x = min(x, 990 - widest) if anchor == "start" else max(x, 10 + widest)
    width = max(title_w, note_w)
    x0 = x if anchor == "start" else x - width if anchor == "end" else x - width / 2
    box = (x0, y - 20, x0 + width, y + 18 + 16 * (len(notes) - 1) + 4)
    return _Label(angle, x, y, anchor, title, notes, title_fill, note_fill, box)


def _disc_half_height(x0: float, x1: float) -> float:
    """Half the height of the disc of radius R + 4 over the column of [x0, x1] nearest to the centre
    (0 when the span lies wholly beyond the disc's width)."""
    radius = R + LABEL_CLEARANCE
    dx = max(x0 - CX, 0, CX - x1)
    return math.sqrt(radius * radius - dx * dx) if dx < radius else 0.0


def _resolve_label_overlaps(boxes: list[Box]) -> list[Box]:
    """Where each label ends up once narrow sectors cannot stack theirs on one another, and a long
    name squeezed in by the frame's edge cannot sit over the wheel.

    A label that hits an earlier one (touching counts), or the disc of radius R + 4 around the
    centre (tangent does not), is pushed vertically, away from the centre, by the height of the
    overlap + 4 px, until it is free: 40 pushes at most, after which it stays where the last one
    left it, possibly still touching something. An earlier label is dealt with before the disc. The
    disc's overlap is measured on the label's column nearest the centre, so the label ends 4 px past
    the end of that chord. A label that hits nothing keeps its reference position.
    """
    placed: list[Box] = []
    for x0, y0, x1, y1 in boxes:
        away = 1 if (y0 + y1) / 2 >= CY else -1
        for _ in range(MAX_PUSHES):
            hit = next(
                (q for q in placed if not (x1 < q[0] or x0 > q[2] or y1 < q[1] or y0 > q[3])), None
            )
            if hit is not None:
                edge = hit[3] if away > 0 else hit[1]
            else:
                half = _disc_half_height(x0, x1)
                if half == 0 or y1 <= CY - half or y0 >= CY + half:
                    break
                edge = CY + away * half
            shift = edge - y0 + LABEL_CLEARANCE if away > 0 else edge - y1 - LABEL_CLEARANCE
            y0, y1 = y0 + shift, y1 + shift
        placed.append((x0, y0, x1, y1))
    return placed


def _leader_line(angle: float, box: Box) -> tuple[float, float, float, float]:
    """From the rim of the wheel (R + 6, on the sector's mid angle) to the nearest box point."""
    x1, y1 = _pt(R + 6, angle)
    return x1, y1, min(max(x1, box[0]), box[2]), min(max(y1, box[1]), box[3])


class _Nova:
    """One drawing of the nova. Mirrors the reference `nova()` step by step."""

    def __init__(
        self,
        payload: dict[str, Any],
        theme: str,
        prefix: str,
        center_label: str,
        alpha: float,
        subgroup_labels: bool = False,
        demo_mark: bool = True,
    ):
        self.th = THEMES[theme]
        self.subgroup_labels = subgroup_labels
        self.theme = theme
        self.p = prefix
        self.center = center_label
        # Filled by `draw()`: where each claim label (then "Not linked to a claim") ends up, and its
        # leader line (None when it stands in its reference place).
        self.label_boxes: list[Box] = []
        self.label_leaders: list[Optional[tuple[float, float, float, float]]] = []
        self.rng = np.random.default_rng(11)  # same jitter for every theme / render
        self.claims = sorted(
            (
                c
                for c in payload.get("claims") or []
                if (c.get("s_pro") or 0) + (c.get("s_anti") or 0) > 0
            ),
            key=lambda c: (-(c.get("s_pro") or 0), str(c.get("name"))),
        )
        self.other = list(payload.get("unlinked") or [])
        self.other_total = int(payload.get("unlinked_total") or len(self.other))
        self.no_enrichment = payload.get("status") == "no_enrichment"
        self.demo = demo_mark and bool((payload.get("provenance") or {}).get("demo"))
        self.sig_nl = -math.log10(alpha)
        values = [c.get("best_nl") or 0 for c in self.claims] or [
            s.get("nl") or 0 for s in self.other
        ]
        self.cap = max(max(values or [0.0]), self.sig_nl * 1.2) * 1.04

    def rad(self, v: float) -> float:
        return 30 + (R - 40) * math.sqrt(max(min(v, self.cap), 0.0) / self.cap)

    def claim_text_color(self, name: str) -> str:
        fill, text = claim_color(name)
        if self.theme == "dark":
            return "#FFD84D" if fill == "#FEC601" else "#E8EEF5"
        return text

    def halo(self) -> str:
        return (
            f'paint-order="stroke" stroke="{self.th["halo"]}" stroke-width="4" '
            f'stroke-linejoin="round"'
        )

    def defs(self) -> str:
        th = self.th
        return (
            f'<defs><pattern id="{self.p}-hatch" width="7" height="7" '
            f'patternUnits="userSpaceOnUse" '
            f'patternTransform="rotate(45)"><rect width="7" height="7" fill="{th["hatch_fill"]}"/>'
            f'<line x1="0" y1="0" x2="0" y2="7" stroke="{th["hatch_line"]}" stroke-width="2.2"/>'
            f"</pattern></defs>"
        )

    def layout(self) -> tuple[dict[str, tuple[float, ...]], dict[str, float]]:
        gap = 2.0
        band: dict[str, tuple[float, ...]] = {}
        ang: dict[str, float] = {}
        if not self.claims:  # the "not linked" sector takes the whole wheel
            other_w = 360.0
            band["other"] = (-90.0 - other_w / 2, -90.0 + other_w / 2)
            ang["other"] = -90.0
            return band, ang
        other_w = 36.0
        avail = 360 - other_w - gap * (len(self.claims) + 1)
        tot = {c["name"]: (c.get("s_pro") or 0) + (c.get("s_anti") or 0) for c in self.claims}
        k = avail / sum(tot.values())
        wid = {name: k * t for name, t in tot.items()}
        a0 = -90.0 - wid[self.claims[0]["name"]] / 2
        for c in self.claims:
            name = c["name"]
            f = wid[name] / tot[name]
            wa, wc = (c.get("s_pro") or 0) * f, (c.get("s_anti") or 0) * f
            band[name] = (a0, a0 + wa, a0 + wa + wc)
            ang[name] = a0 + (wa + wc) / 2
            a0 += wa + wc + gap
        band["other"] = (a0, a0 + other_w - gap)
        ang["other"] = a0 + (other_w - gap) / 2
        return band, ang

    def draw(self) -> list[str]:
        th, rng = self.th, self.rng
        band, ang = self.layout()
        names = [c["name"] for c in self.claims]
        by_name = {c["name"]: c for c in self.claims}
        o: list[str] = []

        # sectors: claim tint over the whole radius + hatched "against" share
        for name in names:
            s0, s1, s2 = band[name]
            o.append(_wedge(s0, s1, R, claim_color(name)[0], th["wedge_opacity"]))
            if s2 > s1:
                o.append(_wedge(s1, s2, R, f"url(#{self.p}-hatch)", 0.9))
        o.append(_wedge(*band["other"], R, AGAINST, 0.07))

        # significance rings
        rt = self.rad(self.sig_nl)
        o.append(
            f'<circle cx="{CX}" cy="{CY}" r="{rt:.1f}" fill="none" stroke="{th["ring"]}" '
            f'stroke-width=".9" stroke-dasharray="3 5" opacity=".6"/>'
        )
        ticks = [(e, self.rad(e)) for e in (5, 10, 20) if e < self.cap * 0.97]
        for _, r_ in ticks:
            o.append(
                f'<circle cx="{CX}" cy="{CY}" r="{r_:.1f}" fill="none" stroke="{th["ring"]}" '
                f'stroke-width=".5" stroke-dasharray="1 4" opacity=".45"/>'
            )
        alpha_label = f"p = {10 ** -self.sig_nl:g}"
        for lab_, r_ in [(alpha_label, rt)] + [(str(e), r_) for e, r_ in ticks]:
            xx, yy = _pt(r_, ang["other"])
            txt = (
                _esc(lab_)
                if lab_.startswith("p")
                else f'10<tspan dy="-5" font-size="8">−{lab_}</tspan>'
            )
            o.append(
                f'<text x="{xx:.1f}" y="{yy + 4:.1f}" font-family="{FONT}" font-size="10.5" '
                f'fill="{th["ring_label"]}" text-anchor="middle" {self.halo()}>{txt}</text>'
            )

        # sub-groups (bubbles) + their terms (radial trails)
        terms, bubbles, todo = [], [], []

        def place(subs, t1, t2):
            if not subs:
                return []
            pad = min(2.0, (t2 - t1) * 0.12)
            t1, t2 = t1 + pad, t2 - pad
            order = sorted(subs, key=lambda s: -(s.get("nl") or 0))
            slots = (
                list(np.linspace(t1, t2, len(order) + 2)[1:-1])
                if len(order) > 1
                else [(t1 + t2) / 2]
            )
            idx = sorted(range(len(order)), key=lambda i: (i % 2, i))  # alternate strong / weak
            return [(order[i], a + rng.normal(0, 0.25)) for i, a in zip(idx, slots)]

        for key in names + ["other"]:
            if key == "other":
                items = place(self.other, *band["other"])
            else:
                s0, s1, s2 = band[key]
                subs = by_name[key].get("subgroups") or []
                con = [s for s in subs if subgroup_side(s) == "against"]
                # A balanced sub-group is neither for nor against: it sits with the plain arc,
                # grey, never in the hatched one.
                items = place([s for s in subs if subgroup_side(s) != "against"], s0, s1)
                items += place(con, s1, s2) if s2 - s1 > 3 else place(con, s0, s2)
            best = by_name[key].get("best_subgroup") if key != "other" else None
            for s, a in items:
                r_ = self.rad(s.get("nl") or 0)
                x, y = _pt(r_, a)
                s_terms = s.get("terms") or []
                if key != "other":
                    col, sz = balance_color(s.get("balance") or 0), _bsize(s.get("delta") or 0)
                else:
                    col, sz = AGAINST, _bsize(sum(t.get("nl") or 0 for t in s_terms)) * 0.7
                for t in s_terms:
                    tnl = t.get("nl") or 0
                    tx, ty = _pt(self.rad(tnl) + rng.normal(0, 1.5), a + rng.normal(0, 0.9))
                    eff = t.get("effect") or 0
                    tc = (th["green"] if eff > 0 else th["coral"]) if eff else AGAINST
                    terms.append(
                        _dot(tx, ty, 0.9 + 0.45 * math.sqrt(tnl), tc, 0.85 if tnl >= 1.3 else 0.4)
                    )
                bubbles.append(_dot(x, y, sz, col, 0.95, f'stroke="{th["bg"]}" stroke-width="1.5"'))
                if key != "other" and (s.get("name") == best or (s.get("balance") or 0) < -0.3):
                    todo.append((x, y, sz, a, str(s.get("name") or "")))
        o += terms + bubbles

        # claim rims (outer liseré), hatched part rim in grey
        for name in names:
            s0, s1, s2 = band[name]
            o.append(_arc(s0, s1, R, claim_color(name)[0], 4))
            if s2 > s1:
                o.append(_arc(s1, s2, R, AGAINST, 4))
        o.append(_arc(*band["other"], R, AGAINST, 2, 'stroke-dasharray="2 4"'))

        # centre disc
        center = self.center if len(self.center) <= 8 else self.center[:7] + "…"
        o.append(f'<circle cx="{CX}" cy="{CY}" r="26" fill="{th["core_fill"]}"/>')
        if center:  # the disc names the condition it was given, never an invented one
            size = 13 if len(center) <= 6 else 11
            name = _text(CX, CY + 5, center, size, th["core_text"], "middle", 700)
            # the text is the cut name; the title (a tooltip) carries the whole one
            o.append(f"<g><title>{_esc(self.center)}</title>{name}</g>")
        if not self.claims:
            message = (
                "No enrichment result for this comparison"
                if self.no_enrichment
                else "No claim with significant evidence"
            )
            o.append(
                _text(CX, CY + R + 44, message, 18, th["dim"], "middle", 600, self.halo())
            )

        # claim labels outside the wheel, each where the reference puts it, then kept apart from the
        # labels before it and off the wheel; one that had to travel is tied back to its sector by a
        # thin leader line, drawn under the labels
        labels = [
            _reference_label(
                ang[name],
                name,
                _sector_note(by_name[name]),
                self.claim_text_color(name),
                th["muted"],
            )
            for name in names
        ]
        if self.other or self.claims:
            labels.append(
                _reference_label(
                    ang["other"],
                    "Not linked to a claim",
                    _plural(self.other_total, "sub-group"),
                    AGAINST_TEXT,
                    AGAINST_TEXT,
                )
            )
        self.label_boxes = _resolve_label_overlaps([label.box for label in labels])
        self.label_leaders = []
        for label, box in zip(labels, self.label_boxes):
            travelled = abs(box[1] - label.box[1])
            self.label_leaders.append(
                _leader_line(label.angle, box) if travelled > LEADER_MIN_SHIFT else None
            )
        for label, leader in zip(labels, self.label_leaders):
            if leader:
                x1, y1, x2, y2 = leader
                o.append(
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                    f'stroke="{label.title_fill}" stroke-width="0.8" opacity="0.6"/>'
                )
        halo = self.halo()
        for label, box in zip(labels, self.label_boxes):
            y = label.y + box[1] - label.box[1]  # the text moves with its box
            o.append(_text(label.x, y, label.title, 21, label.title_fill, label.anchor, 600, halo))
            for j, line in enumerate(label.notes):
                note_y = y + 18 + 16 * j
                o.append(
                    _text(label.x, note_y, line, 12.5, label.note_fill, label.anchor, 500, halo)
                )

        # key sub-group labels (best of each claim + sub-groups against), without overlap.
        # Off by default: the validated reference computes them but never draws them.
        if not self.subgroup_labels:
            return o

        def tw(s, size):
            return len(s) * size * 0.56

        boxes = list(self.label_boxes)
        for x, y, sz, _a, _n in todo:
            boxes.append((x - sz - 3, y - sz - 3, x + sz + 3, y + sz + 3))
        boxes.append((CX - 30, CY - 30, CX + 30, CY + 30))

        def free(b):
            return all(b[2] < q[0] or b[0] > q[2] or b[3] < q[1] or b[1] > q[3] for q in boxes)

        for x, y, sz, a, name in sorted(todo, key=lambda t: -t[2]):
            w = tw(name, 11.5)
            out_dir = 1 if math.cos(math.radians(a)) >= 0 else -1
            g = sz + 6
            cands = [
                (x + out_dir * g, y + 4, out_dir),
                (x - out_dir * g, y + 4, -out_dir),
                (x + out_dir * g, y - 10, out_dir),
                (x + out_dir * g, y + 17, out_dir),
                (x - out_dir * g, y - 10, -out_dir),
                (x - out_dir * g, y + 17, -out_dir),
            ]

            def bx(tx, ty, d):
                return (
                    (tx if d > 0 else tx - w) - 2,
                    ty - 10,
                    (tx + w if d > 0 else tx) + 2,
                    ty + 3,
                )

            pick = next((cd for cd in cands if free(bx(*cd))), cands[0])
            boxes.append(bx(*pick))
            o.append(
                _text(
                    pick[0],
                    pick[1],
                    name,
                    11.5,
                    th["subgroup_label"],
                    "start" if pick[2] > 0 else "end",
                    600,
                    self.halo(),
                )
            )
        return o

    def frame(self) -> tuple[int, int, int, int]:
        """The wheel's crop (x0, y0, x1, y1): the reference crop, grown to hold every label box and
        leader line with an 8 px margin. Read after `draw()`."""
        x0, y0, x1, y1 = _CROP_X, _CROP_Y, _CROP_X + _CROP_W, _CROP_Y + _CROP_H
        points = [(b[0], b[1]) for b in self.label_boxes] + [(b[2], b[3]) for b in self.label_boxes]
        for leader in self.label_leaders:
            if leader:
                points += [leader[:2], leader[2:]]
        for x, y in points:
            x0, y0 = min(x0, math.floor(x - FRAME_MARGIN)), min(y0, math.floor(y - FRAME_MARGIN))
            x1, y1 = max(x1, math.ceil(x + FRAME_MARGIN)), max(y1, math.ceil(y + FRAME_MARGIN))
        return x0, y0, x1, y1

    def annotations(self) -> list[str]:
        """What the wheel says about itself: the title of its radial axis, and the DEMO mark of a
        demo payload (drawn last, over the wheel)."""
        th = self.th
        o: list[str] = []
        if self.claims or self.other:
            o.append(_text(20, 944, f"Radius: {AXIS_TITLE}", 12.5, th["muted"], weight=500))
        if any(claim_state_of(c) == "for_only" for c in self.claims):
            first, second = ZERO_AGAINST_NOTE.split("; ")
            o.append(_text(20, 960, first + ";", 11, th["dim"]))
            o.append(_text(20, 974, second, 11, th["dim"]))
        if self.demo:
            o.append(
                f'<text x="{CX}" y="{CY + 70}" text-anchor="middle" font-family="{FONT}" '
                f'font-size="200" font-weight="700" letter-spacing="10" fill="{th["coral"]}" '
                f'fill-opacity=".16" transform="rotate(-32 {CX} {CY})">DEMO</text>'
            )
        return o

    def blocks(self, y0: float) -> list[str]:
        """Claim cards under the wheel (full poster only)."""
        th = self.th
        o: list[str] = []
        for i, c in enumerate(self.claims[:9]):
            x = 50 + (i % 3) * 315
            y = y0 + (i // 3) * 140
            coh = round(100 * (c.get("coherence") or 0))
            good = coh >= 50
            fill = claim_color(c["name"])[0]
            o += [
                f'<rect x="{x - 14}" y="{y - 18}" width="5" height="128" rx="2" fill="{fill}"/>',
                _text(x, y, c["name"], 22, self.claim_text_color(c["name"]), weight=600),
                f'<rect x="{x}" y="{y + 12}" width="120" height="5" rx="2.5" fill="#E6EEF2"/>',
                f'<rect x="{x}" y="{y + 12}" width="{1.2 * coh:.1f}" height="5" rx="2.5" '
                f'fill="{th["green"] if good else AGAINST}"/>',
                _text(x + 130, y + 18, f"coherence {coh}%", 13, th["muted"]),
                _text(x, y + 40, _sector_note(c), 12.5, th["ink"] if good else AGAINST_TEXT),
            ]
            subs = sorted(c.get("subgroups") or [], key=lambda s: -(s.get("nl") or 0))[:3]
            for j, s in enumerate(subs):
                # Only a sub-group that goes against the claim is greyed out: a balanced one is
                # neutral, in the ink of the page.
                side = subgroup_side(s)
                up = s.get("obs") == "up"
                if side == "for":
                    arrow_col = th["green"] if up else th["coral"]
                else:
                    arrow_col = th["muted"] if side == "balanced" else AGAINST_TEXT
                o += [
                    _text(x, y + 62 + 18 * j, "↑" if up else "↓", 13, arrow_col, weight=600),
                    _text(
                        x + 14,
                        y + 62 + 18 * j,
                        s.get("name") or "",
                        12,
                        AGAINST_TEXT if side == "against" else th["ink"],
                    ),
                ]
        return o

    def legend(self, y: float) -> list[str]:
        th = self.th
        ink = th["ink"]
        o = [
            _dot(56, y - 5, 6, th["green"], 1),
            _dot(70, y - 5, 6, "#C9D1D9", 1),
            _dot(84, y - 5, 6, th["coral"], 1),
            _text(96, y, "Sub-group: for → balanced → against", 13, ink),
            _dot(380, y - 5, 2.2, th["green"], 1),
            _dot(388, y - 5, 2.2, th["coral"], 1),
            _text(398, y, "Individual ontology term", 13, ink),
            f'<rect x="580" y="{y - 11}" width="18" height="12" fill="url(#{self.p}-hatch)" '
            f'stroke="{AGAINST}"/>',
            _text(604, y, "Evidence against the claim (hatched)", 13, ink),
        ]
        y2 = y + 22
        o += [
            _dot(56, y2 - 5, 3.5, AGAINST, 1),
            _dot(70, y2 - 5, 6, AGAINST, 1),
            _text(84, y2, "Size = |Δ pro − anti|", 13, ink),
            f'<path d="M262,{y2} L280,{y2 - 14} A22,22 0 0 1 284,{y2 - 1} Z" fill="#158394" '
            f'opacity=".5"/>',
            _text(288, y2, "Sector width = claim importance (Σ −log10 p of its terms)", 13, ink),
            _text(690, y2, "Coherence = share of evidence for", 13, ink),
        ]
        return o


def render_nova_svg(
    payload: dict[str, Any],
    theme: Theme = "light",
    wheel_only: bool = True,
    center_label: Optional[str] = None,
    id_prefix: Optional[str] = None,
    title: Optional[str] = None,
    logo_data_uri: Optional[str] = None,
    subgroup_labels: bool = False,
    demo_mark: bool = True,
) -> str:
    """SVG of the nova for a `build_nova` payload (the reference mock JSON is accepted too).

    wheel_only=True: the wheel and its labels on a transparent background (viewBox cropped as
    in the report, and grown to hold every label: they are kept off one another and off the wheel,
    and a label that travelled far from its sector is tied back to it by a thin line).
    wheel_only=False: the full poster — header, wheel, claim cards, legend.
    subgroup_labels=True also names the best sub-group of each claim and the ones against it.
    A demo payload (`provenance.demo`) is marked DEMO unless demo_mark is False (a page that
    carries its own mark over the picture).
    """
    if theme not in THEMES:
        raise ValueError(f"Unknown nova theme {theme!r}")
    prefix = id_prefix or f"nova-{uuid.uuid4().hex[:8]}"
    alpha = float(payload.get("alpha") or 0.05)
    label = center_label or str(payload.get("condition") or "")
    nova = _Nova(payload, theme, prefix, label, alpha, subgroup_labels, demo_mark)
    th = THEMES[theme]
    wheel = nova.draw() + nova.annotations()

    x0, y0, x1, y1 = nova.frame()
    if wheel_only:
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0} {y0} {x1 - x0} {y1 - y0}" '
            f'width="{x1 - x0}" height="{y1 - y0}">'
        )
        return "\n".join([head, nova.defs(), *wheel, "</svg>"])

    n_rows = math.ceil(min(len(nova.claims), 9) / 3)
    # Labels pushed past the crop make room: above it (the header stays, the wheel and what follows
    # move down) and below it (the cards start under the labels, not under the footnotes). The
    # poster is as much taller; it keeps its width.
    above_crop = _CROP_Y - y0
    below_crop = y1 - (_CROP_Y + _CROP_H)
    height = 1056 + 140 * n_rows + above_crop + below_crop
    wordmark = "#0B2A3E" if theme == "light" else "#F2F6FA"
    subtitle = title or f"Pathway nova · {payload.get('contrast') or ''}".rstrip(" ·")
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" '
        f'height="{height}">',
        nova.defs(),
        f'<rect width="{W}" height="{height}" fill="{th["bg"]}"/>',
        f'<text x="50" y="72" font-family="{FONT}" font-size="46" font-weight="700" '
        f'letter-spacing="-0.5"><tspan fill="{wordmark}">Cosm</tspan>'
        f'<tspan fill="{th["green"]}">Omics</tspan></text>',
        _text(50, 104, subtitle, 16, th["muted"]),
    ]
    # What the p-values are, and where significance starts: read from the kind of enrichment
    # (`provenance.enrichment_source`), the payload's own `p_source` only for a payload that
    # predates it. With no enrichment result the header says that and nothing else: there is no
    # p-value, hence no threshold, to state.
    kind = (payload.get("provenance") or {}).get("enrichment_source")
    if kind == "none" or payload.get("status") == "no_enrichment":
        o.append(_text(50, 126, "No enrichment result is stored for this comparison", 12, th["dim"]))
    else:
        p_source = p_source_of(kind, (payload.get("provenance") or {}).get("padj_columns") or [])
        p_source = p_source or payload.get("p_source")
        if p_source:
            o.append(_text(50, 126, f"{p_source} · α = {alpha:g}", 12, th["dim"]))
    if logo_data_uri:
        lw = 170
        lh = lw * 190 / 1024
        o += [
            _text(950 - lw - 8, 72, "by", 15, th["muted"], "end"),
            f'<image x="{950 - lw}" y="{72 - lh + 4:.1f}" width="{lw}" height="{lh:.1f}" '
            f'href="{logo_data_uri}"/>',
        ]
    if above_crop:
        wheel = [f'<g transform="translate(0 {above_crop})">', *wheel, "</g>"]
    o += wheel + nova.blocks(1016 + above_crop + below_crop) + nova.legend(height - 46) + ["</svg>"]
    return "\n".join(o)
