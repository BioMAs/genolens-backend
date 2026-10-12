#!/usr/bin/env python3
"""
Build app/data/claims/claim_subgroups.yaml — the CosmOmics nova claim referential.

Reads the SciLicium "Claims_Pathway_Review_*.xlsx" workbooks bundled in app/data/claims/ and
writes one line per claim × term (claim, subgroup, term_id, label, expected_direction), terms
grouped into pathway sub-groups by app/services/cosmomics_grouping.py. Deterministic: running it
twice on the same workbooks gives the same file.

The output is a starting point for curation. Once reviewed, the YAML — not the workbooks — is
what the nova reads; re-running this script overwrites manual edits.

Usage:
    python scripts/build_claim_subgroups.py
    python scripts/build_claim_subgroups.py --dir /path/to/Claims --out /tmp/claim_subgroups.yaml
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml  # noqa: E402

from app.services.claim_import_service import DEFAULT_CLAIMS_DIR, _parse_workbook  # noqa: E402
from app.services.cosmomics_nova import (  # noqa: E402
    CLAIM_SUBGROUPS_PATH,
    subgroup_rows_from_referential,
)

HEADER = """\
# CONFIDENTIAL — SciLicium know-how. Never expose this file to clients or to the frontend.
#
# CosmOmics Pathway Nova — claim referential: one line per claim x term.
#   claim               claim shown on the nova
#   subgroup            pathway sub-group the term belongs to (one bubble on the nova)
#   term_id             GO:..., HSA-... (Reactome), CL:... (STRING cluster)
#   label               term description, also used to match MSigDB-style names
#   expected_direction  UP / DOWN: the direction that SUPPORTS the claim
# A line with an empty claim defines a curated "not linked" sub-group.
#
# Pre-filled by scripts/build_claim_subgroups.py from Claims_Pathway_Review_*.xlsx, then curated.
"""


def main(claims_dir: Path, out: Path) -> None:
    workbooks = sorted(claims_dir.glob("Claims_Pathway_Review_*.xlsx"))
    if not workbooks:
        print(f"No workbooks found in {claims_dir}")
        sys.exit(1)

    pathway_rows: list[dict] = []
    for path in workbooks:
        rows, _ = _parse_workbook(path)
        pathway_rows.extend(rows)

    rows = subgroup_rows_from_referential(pathway_rows)
    lines = [HEADER, "version: 1", "terms:"]
    for row in rows:
        flow = yaml.safe_dump(row, default_flow_style=True, sort_keys=False, width=10_000)
        lines.append(f"  - {flow.strip()}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    per_claim = Counter(r["claim"] for r in rows)
    groups = Counter((r["claim"], r["subgroup"]) for r in rows)
    print(f"{len(rows)} lines, {len({r['term_id'] for r in rows})} terms -> {out}")
    for claim, n in per_claim.items():
        n_groups = sum(1 for (c, _) in groups if c == claim)
        print(f"  {claim:<14} {n:4} terms in {n_groups:3} sub-groups")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dir", type=Path, default=DEFAULT_CLAIMS_DIR)
    parser.add_argument("--out", type=Path, default=CLAIM_SUBGROUPS_PATH)
    args = parser.parse_args()
    main(args.dir, args.out)
