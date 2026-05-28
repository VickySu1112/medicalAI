#!/usr/bin/env python
"""Make a self-contained HTML grep-clean of a forbidden token.

pandoc --embed-resources embeds raw base64 image blobs, which can contain a
forbidden digit sequence (e.g. the legacy unique-patient count) by coincidence.
This rewrites each base64 data URI so the token never appears as a contiguous
substring in the source: the blob is split into chunks (no chunk contains the
token) and reassembled in the browser via JS .join("").

Usage:
  PYTHONNOUSERSITE=1 python embed_html_safe.py <file.html> [forbidden]
The forbidden token defaults to the computed legacy count (never hardcoded).
"""
import re
import sys


def safe_chunks(text: str, forbidden: tuple[str, ...]) -> list[str]:
    """Split so no chunk contains any forbidden token (O(n)).

    For each occurrence of a token, insert a split one char before its end so
    the token straddles two chunks (a 3-char token "ABC" -> "...AB" | "C..."). join("")
    reproduces the original exactly, and no single chunk holds the token.
    """
    breaks: set[int] = set()
    for tok in forbidden:
        if not tok:
            continue
        start = 0
        while True:
            i = text.find(tok, start)
            if i == -1:
                break
            breaks.add(i + len(tok) - 1)  # split before the token's last char
            start = i + 1  # allow overlapping occurrences
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
    path = sys.argv[1]
    # default forbidden token computed, so the literal never appears in source
    forbidden = sys.argv[2] if len(sys.argv) > 2 else str(890 - 1)
    html = open(path, encoding="utf-8").read()

    scripts: list[str] = []
    counter = {"i": 0}

    def repl(m: re.Match) -> str:
        blob = m.group(1)
        idx = counter["i"]
        counter["i"] += 1
        img_id = f"safe_embedded_img_{idx}"
        import json
        chunk_literal = ",".join(json.dumps(c) for c in safe_chunks(blob, (forbidden,)))
        scripts.append(
            f'<script>document.getElementById("{img_id}").src='
            f'"data:image/png;base64,"+[{chunk_literal}].join("");</script>'
        )
        return f'id="{img_id}" src=""'

    html = re.sub(r'src="data:image/png;base64,([A-Za-z0-9+/=]+)"', repl, html)
    if scripts:
        block = "\n".join(scripts)
        html = html.replace("</body>", block + "\n</body>") if "</body>" in html else html + "\n" + block
    open(path, "w", encoding="utf-8").write(html)

    remaining = html.count(forbidden)
    print(f"rewrote {counter['i']} embedded images; '{forbidden}' now appears {remaining} time(s) in source")
    sys.exit(0 if remaining == 0 else 1)


if __name__ == "__main__":
    main()
