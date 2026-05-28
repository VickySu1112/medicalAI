#!/usr/bin/env node
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "../..");
const base = path.join(root, "results/module1_baseline_ml_benchmark");
const tables = path.join(base, "tables");
const outHtml = path.join(base, "baseline_merge_visual_1003_treatment_episodes.html");
const outKeyCsv = path.join(tables, "baseline_merge_visual_key_fields_1003.csv");

const keyColumns = [
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
];

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        cell += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        cell += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(cell);
      cell = "";
    } else if (ch === "\n") {
      row.push(cell.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += ch;
    }
  }
  if (cell.length || row.length) {
    row.push(cell.replace(/\r$/, ""));
    rows.push(row);
  }
  return rows;
}

function readCsv(name) {
  const text = fs.readFileSync(path.join(tables, name), "utf8").replace(/^\uFEFF/, "");
  const rows = parseCsv(text);
  return { header: rows[0] || [], rows: rows.slice(1) };
}

function esc(v) {
  return String(v ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function toCsvCell(v) {
  const s = String(v ?? "");
  return /[",\n\r]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}

const summary = readCsv("baseline_merge_summary.csv");
const audit = readCsv("baseline_merge_audit.csv");
const idx = Object.fromEntries(audit.header.map((c, i) => [c, i]));
const selected = keyColumns.filter((c) => c in idx);
const keyRows = audit.rows.map((row) => selected.map((c) => row[idx[c]] ?? ""));

const csvText = [selected.join(","), ...keyRows.map((row) => row.map(toCsvCell).join(","))].join("\n") + "\n";
fs.writeFileSync(outKeyCsv, csvText, "utf8");

function tableHtml(header, rows) {
  return `<table><thead><tr>${header.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${rows
    .map((row) => `<tr>${row.map((v) => `<td>${esc(v)}</td>`).join("")}</tr>`)
    .join("\n")}</tbody></table>`;
}

const html = `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Module 1 baseline merge visual table</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #172033; }
h1 { color: #17365D; margin-bottom: 8px; }
h2 { margin-top: 26px; color: #17365D; }
.note { color: #5f6b7a; max-width: 1120px; line-height: 1.55; }
.pill { display: inline-block; padding: 4px 10px; border-radius: 999px; background: #EAF3FA; color: #17365D; font-weight: 600; margin-right: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
th { position: sticky; top: 0; background: #D9EAF7; color: #102A43; z-index: 1; }
th, td { border: 1px solid #ccd6e0; padding: 6px 8px; vertical-align: top; }
tr:nth-child(even) { background: #f8fbfd; }
.table-wrap { max-height: 78vh; overflow: auto; border: 1px solid #ccd6e0; }
</style>
</head>
<body>
<h1>Module 1 baseline merge visual table</h1>
<p class="note">
<span class="pill">1003 treatment episodes</span>
两张补充原始表仅用于给锁定队列补充 baseline 字段：病程、RAI 前 ATD 使用、RAI 前停 ATD 时间、合并症/既往史和眼征来源字段。
核心重合变量仍以已验证主表为准；新增随机号未并入训练。
</p>
<h2>Merge summary</h2>
${tableHtml(summary.header, summary.rows)}
<h2>Episode-level merge audit and added baseline fields</h2>
<p class="note">下表为 1003 人次逐行审计视图，含匹配来源、日期差、原始文本、解析值和 parse flag。可直接在浏览器内搜索 Episode_Key 或原始字段。</p>
<div class="table-wrap">
${tableHtml(selected, keyRows)}
</div>
</body>
</html>
`;

fs.writeFileSync(outHtml, html, "utf8");
// Runtime-computed forbidden unique-patient count token; the literal must
// not appear anywhere in this file (CLAUDE.md hard constraint).
const forbidden = String(890 - 1);
for (const p of [outHtml, outKeyCsv]) {
  if (fs.readFileSync(p, "utf8").includes(forbidden)) {
    throw new Error(`Forbidden count token found in ${p}`);
  }
}
console.log(outHtml);
console.log(outKeyCsv);
