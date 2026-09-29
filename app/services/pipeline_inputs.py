"""
Normalise the self-service pipeline inputs to real TSV before they reach R.

The upload endpoint accepts .csv/.tsv/.txt/.xlsx/.xls for the count matrix, the
sample sheet and the comparisons file, but r_scripts/run_multimethod_pipeline.R
reads all three with read_tsv. Handed a comma-separated or Excel file, R parses
it as a single column and the analysis fails.

The conversion only changes the delimiter. The header row, the gene ID column
and every cell value are carried over verbatim as text: nothing is re-typed,
renamed or re-indexed, so a blank first header (R's write.csv) stays blank.
"""

import csv
import io
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Tab first: a genuine TSV whose free-text cells contain commas must stay a TSV.
_CANDIDATE_DELIMITERS = ("\t", ",", ";")
_SNIFF_LINES = 50

_XLSX_MAGIC = b"PK\x03\x04"
_XLS_MAGIC = b"\xd0\xcf\x11\xe0"

INPUT_LABELS = {
    "counts": "count matrix",
    "samples": "sample sheet",
    "comparisons": "comparisons file",
}


class PipelineInputError(ValueError):
    """An input file could not be turned into a table R can read."""


def detect_delimiter(text: str, default: str = "\t") -> str:
    """
    Guess the field delimiter of delimited text.

    A candidate wins when it splits the first lines into the same number (> 1) of
    fields. Ragged files (a trailing blank column on some lines, a header one field
    short) fall back to the delimiter that appears most often in the header line.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()][:_SNIFF_LINES]
    if not lines:
        raise PipelineInputError("the file is empty")

    for delim in _CANDIDATE_DELIMITERS:
        widths = {len(row) for row in csv.reader(lines, delimiter=delim)}
        if len(widths) == 1 and widths.pop() > 1:
            return delim

    header_counts = {d: lines[0].count(d) for d in _CANDIDATE_DELIMITERS}
    best = max(_CANDIDATE_DELIMITERS, key=lambda d: header_counts[d])
    return best if header_counts[best] > 0 else default


def _decode(raw: bytes) -> str:
    # Excel's "CSV" export on Windows is cp1252, not UTF-8.
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _is_excel(raw: bytes) -> bool:
    # Content, not extension: a CSV renamed .xlsx is still text.
    return raw.startswith(_XLSX_MAGIC) or raw.startswith(_XLS_MAGIC)


def _read_excel_rows(raw: bytes) -> list[list[str]]:
    df = pd.read_excel(io.BytesIO(raw), sheet_name=0, header=None, dtype=str)
    return df.fillna("").values.tolist()


def _drop_empty(rows: list[list[str]]) -> list[list[str]]:
    """Drop blank lines and trailing columns that are empty on every row (stray cell formatting)."""
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    width = max((len(r) for r in rows), default=0)
    while width > 0 and all(len(r) < width or not r[width - 1].strip() for r in rows):
        width -= 1
    return [r[:width] for r in rows]


def _write_tsv(rows: list[list[str]], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh, delimiter="\t", lineterminator="\n").writerows(rows)


def normalise_to_tsv(src: Path, dest: Path) -> Path:
    """
    Return a path R can read with read_tsv.

    Tab-delimited text is returned as ``src`` untouched. Excel workbooks (first
    sheet) and comma- or semicolon-delimited text are rewritten to ``dest``. The
    delimiter is sniffed whatever the extension, since users save comma-separated
    data as .txt.
    """
    raw = src.read_bytes()

    if _is_excel(raw):
        rows = _read_excel_rows(raw)
        source_format = "Excel"
    else:
        text = _decode(raw)
        default = "," if src.suffix.lower() == ".csv" else "\t"
        delimiter = detect_delimiter(text, default=default)
        if delimiter == "\t":
            return src
        rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter))
        source_format = f"{delimiter!r}-delimited"

    rows = _drop_empty(rows)
    if not rows:
        raise PipelineInputError("the file is empty")

    _write_tsv(rows, dest)
    logger.info("[PIPELINE_INPUTS] %s (%s) -> %s", src.name, source_format, dest)
    return dest


def prepare_pipeline_inputs(inputs: dict[str, str], workdir: Path) -> dict[str, str]:
    """
    Normalise each pipeline input (keys of INPUT_LABELS) into ``workdir``.

    Returns the same keys mapped to the paths to hand to the R script. Errors name
    the offending input so the analysis error message tells the user which file
    to fix.
    """
    prepared: dict[str, str] = {}
    for key, path in inputs.items():
        src = Path(path)
        label = INPUT_LABELS.get(key, key)
        if not src.is_file():
            raise PipelineInputError(f"The {label} file ({src.name}) is missing on the server.")
        try:
            prepared[key] = str(normalise_to_tsv(src, workdir / f"{key}.tsv"))
        except Exception as exc:
            raise PipelineInputError(
                f"Could not read the {label} ({src.name}) as a table: {exc}"
            ) from exc
    return prepared
