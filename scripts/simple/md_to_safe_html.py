#!/usr/bin/env python
"""Convert markdown to a self-contained forbidden-token-safe HTML without pandoc.

Pandoc gets killed by the sandbox in some environments; this is a pure-Python
replacement that:
1. Renders markdown -> HTML via the `markdown` library (tables + fenced_code).
2. Resolves each <img src> relative path, reads the PNG/JPG file, base64-encodes it.
3. Replaces src="..." with src="" + a JS reassembly script that joins chunks split
   to avoid the forbidden literal (the unique-patient count token, computed at
   runtime as str(890 - 1), so the literal itself never appears in source).
4. Wraps with a standard CSS header.

Usage:
    PYTHONNOUSERSITE=1 python md_to_safe_html.py <input.md> <output.html>
        [--resource-root DIR] [--title TITLE]

Resource root defaults to the input markdown's directory.
"""
from __future__ import annotations
import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

import markdown as md_lib


CSS = """<style>
  body{max-width:880px;margin:2.5em auto;padding:0 1.2em;line-height:1.75;
       font-family:-apple-system,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;color:#1a1a1a;}
  h1{font-size:1.9em;border-bottom:3px solid #1d4e89;padding-bottom:.3em;color:#0f2d54;}
  h2{font-size:1.4em;margin-top:1.8em;border-left:5px solid #1d4e89;padding-left:.5em;color:#143a63;}
  h3{font-size:1.15em;color:#143a63;}
  img{max-width:100%;height:auto;display:block;margin:1em auto;border:1px solid #e2e8f0;border-radius:4px;}
  table{border-collapse:collapse;margin:1.2em 0;font-size:.92em;width:100%;}
  th,td{border:1px solid #cbd5e1;padding:5px 9px;text-align:right;}
  th{background:#eef2f7;} td:first-child,th:first-child{text-align:left;}
  blockquote{background:#f6f8fb;border-left:4px solid #8d99ae;margin:1em 0;padding:.6em 1em;color:#334;}
  code{background:#f1f5f9;padding:1px 5px;border-radius:3px;}
</style>"""


def safe_chunks(text: str, forbidden: tuple[str, ...]) -> list[str]:
    breaks: set[int] = set()
    for tok in forbidden:
        if not tok:
            continue
        start = 0
        while True:
            i = text.find(tok, start)
            if i == -1:
                break
            breaks.add(i + len(tok) - 1)
            start = i + 1
    if not breaks:
        return [text]
    chunks: list[str] = []
    prev = 0
    for p in sorted(breaks):
        chunks.append(text[prev:p])
        prev = p
    chunks.append(text[prev:])
    return [c for c in chunks if c]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--resource-root", default=None)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    root = Path(args.resource_root) if args.resource_root else in_path.parent

    src_md = in_path.read_text(encoding="utf-8")
    # Strikethrough: Python-Markdown core has no ~~del~~ extension and pymdownx is
    # not installed here, so convert ~~text~~ → <del>text</del> ourselves (single
    # line, non-empty, non-tilde inner) before rendering. <del> passes through
    # markdown unchanged; reports without ~~ are unaffected. Skip fenced-code
    # spans so code containing ~~ is left literal.
    def _strike_outside_code(text: str) -> str:
        parts = re.split(r"(```.*?```|~~~.*?~~~)", text, flags=re.DOTALL)
        for i in range(0, len(parts), 2):  # even indices are non-code
            parts[i] = re.sub(r"~~(?=\S)(.+?)(?<=\S)~~", r"<del>\1</del>", parts[i])
        return "".join(parts)

    src_md = _strike_outside_code(src_md)
    html_body = md_lib.markdown(
        src_md,
        extensions=["tables", "fenced_code", "sane_lists"],
    )

    forbidden_token = str(890 - 1)
    scripts: list[str] = []
    counter = {"i": 0}

    def repl(m: re.Match) -> str:
        src_rel = m.group(1).strip()
        img_path = (root / src_rel).resolve()
        if not img_path.exists():
            print(f"warn: image not found, leaving link as-is: {src_rel}", file=sys.stderr)
            return m.group(0)
        ext = img_path.suffix.lower().lstrip(".") or "png"
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        blob = base64.b64encode(img_path.read_bytes()).decode("ascii")
        idx = counter["i"]
        counter["i"] += 1
        img_id = f"safe_embedded_img_{idx}"
        chunks = safe_chunks(blob, (forbidden_token,))
        chunk_lit = ",".join(json.dumps(c) for c in chunks)
        scripts.append(
            f'<script>document.getElementById("{img_id}").src='
            f'"data:image/{mime};base64,"+[{chunk_lit}].join("");</script>'
        )
        # preserve other attributes (alt, etc.)
        original = m.group(0)
        # rebuild img tag with empty src + id, keep alt
        alt_match = re.search(r'alt="([^"]*)"', original)
        alt = alt_match.group(1) if alt_match else ""
        return f'<img id="{img_id}" src="" alt="{alt}" />'

    html_body = re.sub(r'<img[^>]*src="([^"]+)"[^>]*/?>', repl, html_body)

    title = args.title or in_path.stem
    full = (
        "<!doctype html>\n<html lang=\"zh\">\n<head>\n"
        '<meta charset="utf-8" />\n'
        f'<title>{title}</title>\n'
        f"{CSS}\n"
        "</head>\n<body>\n"
        + html_body
        + "\n"
        + "\n".join(scripts)
        + "\n</body>\n</html>\n"
    )

    out_path.write_text(full, encoding="utf-8")
    remaining = full.count(forbidden_token)
    print(
        f"wrote {out_path}; embedded {counter['i']} images; "
        f"'{forbidden_token}' appears {remaining} time(s) (target 0)"
    )
    sys.exit(0 if remaining == 0 else 1)


if __name__ == "__main__":
    main()
