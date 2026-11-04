#!/usr/bin/env python3
"""
Render one comparison report to a local PDF, outside Celery — same services as the report tasks.

Usage (from backend/, with DATABASE_URL pointing at the database to read):
    python scripts/render_comparison_report.py --dataset-id <UUID> --comparison KO_vs_WT \\
        --template editorial --out report.pdf [--requested-by <user UUID>] \\
        [--cover-info '{"client_ref": "SLC-26-000", "checked_by": "J. Doe"}']

The editorial format needs Chromium (worker image); the LaTeX format needs pdflatex (r-worker
image). Nothing is written to the database or to storage.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.worker.tasks.report_task import _fresh_session  # noqa: E402


async def main(args: argparse.Namespace) -> None:
    from app.services.report_latex_service import report_latex_service

    dataset_id = UUID(args.dataset_id)
    requested_by = UUID(args.requested_by) if args.requested_by else None
    cover_info = json.loads(args.cover_info) if args.cover_info else None
    async with _fresh_session() as db:
        demo = await report_latex_service.is_demo_requester(db, requested_by)
        if args.template == "editorial":
            from app.services.report_editorial_service import report_editorial_service

            pdf = await report_editorial_service.render_comparison_pdf(
                db,
                dataset_id,
                args.comparison,
                requested_by=requested_by,
                cover_info=cover_info,
                generate_ai=args.generate_ai,
                demo_watermark=demo,
            )
        else:
            pdf = await report_latex_service.render_comparison_pdf(
                db,
                dataset_id,
                args.comparison,
                requested_by=requested_by,
                cover_info=cover_info,
                generate_ai=args.generate_ai,
                demo_watermark=demo,
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pdf)
    print(f"{out}: {len(pdf):,} bytes ({args.template})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-id", required=True, help="DEG dataset UUID")
    parser.add_argument("--comparison", required=True, help="comparison name")
    parser.add_argument("--template", choices=["latex", "editorial"], default="editorial")
    parser.add_argument("--out", required=True, help="output PDF")
    parser.add_argument("--requested-by", help="user UUID (branding, demo watermark)")
    parser.add_argument("--cover-info", help="cover / sign-off fields as JSON")
    parser.add_argument(
        "--generate-ai",
        action="store_true",
        help="call the LLM when no usable interpretation is cached",
    )
    asyncio.run(main(parser.parse_args()))
