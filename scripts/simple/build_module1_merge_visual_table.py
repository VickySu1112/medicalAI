#!/usr/bin/env python3
"""Build a readable visual workbook/HTML for Module 1 baseline merge outputs."""

from __future__ import annotations

import csv
import html
from pathlib import Path

try:
    import xlsxwriter
except Exception:
    xlsxwriter = None


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results" / "module1_baseline_ml_benchmark"
TABLES = BASE / "tables"
OUT_XLSX = TABLES / "baseline_merge_visual_1003_treatment_episodes.xlsx"
OUT_HTML = BASE / "baseline_merge_visual_1003_treatment_episodes.html"


SHEETS = [
    ("Summary", TABLES / "baseline_merge_summary.csv"),
    ("Merged_1003_NewFields", TABLES / "baseline_augmented_1003_treatment_episodes.csv"),
    ("Merge_Audit_1003", TABLES / "baseline_merge_audit.csv"),
    ("Feature_Manifest_Core", TABLES / "feature_manifest.csv"),
    ("Feature_Manifest_Aug", TABLES / "augmented_feature_manifest.csv"),
    ("Discordance_Summary", TABLES / "discordance_summary.csv"),
    ("External_Candidates", TABLES / "future_external_candidate_rows_not_used.csv"),
]


KEY_COLUMNS = [
    "Episode_Key",
    "Match_Status",
    "Match_Method",
    "Main_RAI_Date",
    "Source_File",
    "Source_Sheet",
    "RAI_Date_Diff_Days",
    "DiseaseDuration_Raw",
    "DiseaseDuration_Months",
    "DiseaseDuration_ParseFlag",
    "PreRAI_ATD_Use_Raw",
    "PreRAI_ATD_Use",
    "PreRAI_ATD_Use_ParseFlag",
    "PreRAI_ATD_Stop_Raw",
    "PreRAI_ATD_Stop_Days",
    "PreRAI_ATD_Stop_ParseFlag",
    "Comorbidity_Raw",
    "EyeSigns_Raw",
]


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def width_for(values: list[str], header: str) -> int:
    max_len = max([len(str(header))] + [len(str(v)) for v in values[:200]], default=10)
    return max(10, min(34, max_len + 2))


def write_sheet(workbook: xlsxwriter.Workbook, name: str, path: Path) -> None:
    header, rows = read_csv(path)
    ws = workbook.add_worksheet(name[:31])
    title_fmt = workbook.add_format({"bold": True, "font_size": 14, "font_color": "#17365D"})
    note_fmt = workbook.add_format({"font_color": "#666666"})
    header_fmt = workbook.add_format(
        {
            "bold": True,
            "bg_color": "#D9EAF7",
            "border": 1,
            "text_wrap": True,
            "valign": "top",
        }
    )
    body_fmt = workbook.add_format({"border": 1, "valign": "top"})
    ws.write(0, 0, name, title_fmt)
    ws.write(1, 0, f"Source: {path.name}", note_fmt)
    start_row = 3
    for col_idx, col in enumerate(header):
        ws.write(start_row, col_idx, col, header_fmt)
    for row_idx, row in enumerate(rows, start_row + 1):
        for col_idx, val in enumerate(row):
            ws.write(row_idx, col_idx, val, body_fmt)
    if header:
        ws.freeze_panes(start_row + 1, 1)
        ws.autofilter(start_row, 0, start_row + len(rows), len(header) - 1)
    for col_idx, col in enumerate(header):
        sample = [r[col_idx] for r in rows if col_idx < len(r)]
        ws.set_column(col_idx, col_idx, width_for(sample, col))


def write_readme_sheet(workbook: xlsxwriter.Workbook) -> None:
    ws = workbook.add_worksheet("README")
    title = workbook.add_format({"bold": True, "font_size": 16, "font_color": "#17365D"})
    text = workbook.add_format({"text_wrap": True, "valign": "top"})
    ws.set_column(0, 0, 28)
    ws.set_column(1, 1, 90)
    rows = [
        ("Workbook", "Module 1 baseline merge visual table"),
        ("Analysis unit", "1003 treatment episodes"),
        ("Purpose", "Human-readable view of the baseline fields merged from the two supplemental raw workbooks into the locked 1003 treatment-episode cohort."),
        ("Primary table", "Merged_1003_NewFields: one row per treatment episode with parsed disease duration, pre-RAI ATD use, pre-RAI ATD withdrawal time, comorbidity raw text, and eye-sign source fields."),
        ("Audit table", "Merge_Audit_1003: source workbook/sheet, selected join method, date difference, parse flags, and anonymized key tokens for reproducibility."),
        ("External candidates", "Rows found in the supplemental source table but not used for training because the locked analysis cohort remains the 1003 treatment episodes."),
    ]
    ws.write(0, 0, "Module 1 Baseline Merge Visual Table", title)
    for i, (k, v) in enumerate(rows, 2):
        ws.write(i, 0, k, text)
        ws.write(i, 1, v, text)


def build_xlsx() -> None:
    if xlsxwriter is None:
        return
    workbook = xlsxwriter.Workbook(str(OUT_XLSX), {"constant_memory": True})
    write_readme_sheet(workbook)
    for name, path in SHEETS:
        if path.exists():
            write_sheet(workbook, name, path)
    workbook.close()


def build_html() -> None:
    summary_header, summary_rows = read_csv(TABLES / "baseline_merge_summary.csv")
    audit_header, audit_rows = read_csv(TABLES / "baseline_merge_audit.csv")
    idx = {c: i for i, c in enumerate(audit_header)}
    selected_cols = [c for c in KEY_COLUMNS if c in idx]

    def cell(v: str) -> str:
        return html.escape("" if v is None else str(v))

    summary_html = "\n".join(
        "<tr>" + "".join(f"<td>{cell(v)}</td>" for v in row) + "</tr>" for row in summary_rows
    )
    header_html = "".join(f"<th>{cell(c)}</th>" for c in selected_cols)
    body_html = "\n".join(
        "<tr>" + "".join(f"<td>{cell(row[idx[c]])}</td>" for c in selected_cols) + "</tr>"
        for row in audit_rows
    )
    OUT_HTML.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Module 1 baseline merge visual table</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #172033; }}
h1 {{ color: #17365D; margin-bottom: 8px; }}
.note {{ color: #5f6b7a; max-width: 1080px; line-height: 1.55; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th {{ position: sticky; top: 0; background: #D9EAF7; color: #102A43; z-index: 1; }}
th, td {{ border: 1px solid #ccd6e0; padding: 6px 8px; vertical-align: top; }}
tr:nth-child(even) {{ background: #f8fbfd; }}
.panel {{ margin: 22px 0 30px 0; }}
.table-wrap {{ max-height: 78vh; overflow: auto; border: 1px solid #ccd6e0; }}
</style>
</head>
<body>
<h1>Module 1 baseline merge visual table</h1>
<p class="note">分析单位为 1003 治疗人次。此页面展示两张补充原始表贴回锁定队列后的合表审计和新增 baseline 字段；核心建模仍以已验证主表变量为准。</p>
<div class="panel">
<h2>Merge summary</h2>
<table><thead><tr>{''.join(f'<th>{cell(c)}</th>' for c in summary_header)}</tr></thead><tbody>{summary_html}</tbody></table>
</div>
<div class="panel">
<h2>Episode-level merge audit and added baseline fields</h2>
<div class="table-wrap">
<table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>
</div>
</div>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    build_xlsx()
    build_html()
    produced = [OUT_HTML]
    if OUT_XLSX.exists():
        produced.insert(0, OUT_XLSX)
    # Runtime-computed forbidden unique-patient count token; the literal must
    # not appear anywhere in this file (CLAUDE.md hard constraint).
    forbidden = str(890 - 1).encode("ascii")
    for path in produced:
        if forbidden in path.read_bytes():
            raise RuntimeError(f"Forbidden count token found in {path}")
    if OUT_XLSX.exists():
        print(OUT_XLSX)
    else:
        print("XLSX skipped: xlsxwriter unavailable in this runtime")
    print(OUT_HTML)


if __name__ == "__main__":
    main()
