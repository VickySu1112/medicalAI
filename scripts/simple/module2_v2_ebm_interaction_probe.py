#!/usr/bin/env python
"""探查并可视化 EBM 真实学到的一个两两交互项 g(x_i, x_j) 的二维查表,
用于向合作者讲清「2D 交互 = 网格查表,非外积」。打印 data 结构 + 出热力图。
"""
from __future__ import annotations
import os
os.environ.setdefault("MPLBACKEND", "Agg"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
for _fp in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"):
    if os.path.exists(_fp):
        _fm.fontManager.addfont(_fp); plt.rcParams["font.family"] = "Arial Unicode MS"; break
plt.rcParams["axes.unicode_minus"] = False
from scripts.simple.module2_v2_impute_experiment import build_rows_for_method
from scripts.simple.module2_v2_b4_ebm_oof import ebm_oof_and_temporal
from scripts.simple.module2_v2_b4_ebm_axes import build_feats_at_L

DISP = {"ThyroidW": "甲状腺重量", "TRAb": "TRAb", "TGAb": "TGAb", "TPOAb": "TPOAb", "Sex": "性别",
        "FT4_0M": "FT4(0M)", "TSH_0M": "TSH(0M)", "log1p_DiseaseDuration_Months_Aug": "病程(log,月)",
        "Uptake24h": "24h 摄碘率", "HalfLife": "碘半衰期", "TSH_current": "当期 TSH", "TSH_velocity": "TSH 变化速度",
        "Hormone_load": "FT3,FT4 综合水平", "T3T4_balance": "FT3,FT4 落差",
        "Velocity_load": "FT3,FT4 综合变化速度", "Velocity_balance": "FT3,FT4 速度落差"}
def disp(t): return DISP.get(t, t)


def _resolve(term, live):
    """占位名 feature_NNNN → 真名(按 live 列序)。EBM 在 numpy array 上 fit。"""
    def one(tok):
        tok = tok.strip()
        if tok.startswith("feature_"):
            try:
                return live[int(tok.split("_")[1])]
            except (ValueError, IndexError):
                return tok
        return tok
    return " & ".join(one(p) for p in term.split(" & ")) if " & " in term else one(term)


# corrected 真值 + LOCF 口径,演示地标 6M(0M 由 M1 承担,不再做 0M EBM)。
# 选用与旧 0M 例同一特征对「TPOAb × FT3,FT4 落差」——6M 它是首位交互(importance≈0.24),
# 教学点不变(TPOAb 效应方向随落差翻转 = 2D 查表非外积),口径已更正。
L = 6
OUTPNG = ROOT / "results" / "module2_v2_vertical" / "m2v2_ebm_full_locf" / "figures" / "Interaction_demo_6M.png"

rows = build_rows_for_method("locf", (L,))
y = rows["Y_24M_NHRH"].values; lm = rows["landmark"].values
is_dev = (rows["Split"] == "Development").values
devL = is_dev & (lm == L)
# final dev-fit EBM(OOF + temporal 协议),返回 live 列序用于占位名解析
_pred, ebm, live = ebm_oof_and_temporal(rows, y, lm, is_dev, L)
feat = build_feats_at_L(rows, devL)
Xtr, ytr = feat.loc[devL, live], y[devL]
g = ebm.explain_global()
overall = dict(zip(g.data()["names"], g.data()["scores"]))

# 占位名 → 真名,优先选用 TPOAb × T3T4_balance(与旧 0M 例同对);否则退首位 cont×cont 交互
inter = [(i, n, _resolve(n, live)) for i, n in enumerate(ebm.term_names_) if " & " in n]
inter = sorted(inter, key=lambda t: -overall.get(t[1], 0))
print(f"=== {L}M 交互项(按整体重要性排序,真名) ===")
for i, n, rn in inter:
    print(f"  idx={i:2d}  {rn:42s}  importance={overall.get(n, 0):.4f}")

PREF = {"TPOAb & T3T4_balance", "T3T4_balance & TPOAb"}
chosen = next((t for t in inter if _resolve(t[1], live) in PREF), None)
idx, name, rname = chosen if chosen is not None else inter[0]
a_n, b_n = rname.split(" & ")
d = g.data(idx)
print(f"\n=== 选用交互项: {name}  ({disp(a_n)} × {disp(b_n)}) ===")
print("data keys:", sorted(d.keys()))
for k in d:
    v = d[k]
    try:
        arr = np.asarray(v, float); print(f"  {k}: shape={arr.shape}")
    except Exception:
        print(f"  {k}: {type(v).__name__} = {v}")

# 自适应取 2D scores + 两轴 edges
scores = np.asarray(d["scores"], float)               # (nA, nB)
le = np.asarray(d.get("left_names"), float)           # nA(+1) edges for A
re = np.asarray(d.get("right_names"), float)          # nB(+1) edges for B
print("\nscores 2D shape:", scores.shape, "left_names:", le.shape, "right_names:", re.shape)
print("scores 范围:", float(np.nanmin(scores)), "→", float(np.nanmax(scores)))

# ---- 左=真实 62×62 网格;右=粗化 5×5(便于读数) ----
le2 = le[np.isfinite(le)]; re2 = re[np.isfinite(re)]
nA, nB = scores.shape
ex = le2 if len(le2) == nA + 1 else np.linspace(np.nanmin(le2), np.nanmax(le2), nA + 1)
ey = re2 if len(re2) == nB + 1 else np.linspace(np.nanmin(re2), np.nanmax(re2), nB + 1)
vmax = float(np.nanmax(np.abs(scores)))
Av = Xtr[a_n].values.astype(float); Bv = Xtr[b_n].values.astype(float)

rho = float(np.corrcoef(Av, Bv)[0, 1])
H2, _, _ = np.histogram2d(Av, Bv, bins=[ex, ey])
occ = float((H2 > 0).sum()) / H2.size * 100.0
print(f"\ncorr({disp(a_n)}, {disp(b_n)}) on dev = {rho:+.3f}  |  网格 occupancy = {occ:.1f}% 格子有≥1样本（共 {H2.size} 格）")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.6))
im1 = ax1.pcolormesh(ex, ey, scores.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
ax1.scatter(Av, Bv, s=3, c="#111", alpha=0.18, linewidths=0, zorder=3)
ax1.set_xlim(np.nanpercentile(Av, 1), np.nanpercentile(Av, 99))
ax1.set_ylim(np.nanpercentile(Bv, 1), np.nanpercentile(Bv, 99))
ax1.set_xlabel(f"{disp(a_n)}（实测值）→", fontsize=9)
ax1.set_ylabel(f"{disp(b_n)}（实测值,标准化）→", fontsize=9)
ax1.set_title(f"真实 g + 训练样本(黑点) · corr={rho:+.2f}\n仅 {occ:.0f}% 格子有数据,空白处的 g 靠正则外推", fontsize=9)
fig.colorbar(im1, ax=ax1, fraction=0.046, label="log-odds 贡献")

def coarse(M, n=5):
    rs = np.array_split(np.arange(M.shape[0]), n); cs = np.array_split(np.arange(M.shape[1]), n)
    return np.array([[float(M[np.ix_(r, c)].mean()) for c in cs] for r in rs])
N = 5
C = coarse(scores, N)
im2 = ax2.imshow(C.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
for ii in range(N):
    for jj in range(N):
        ax2.text(ii, jj, f"{C[ii, jj]:+.2f}", ha="center", va="center", fontsize=10,
                 color="white" if abs(C[ii, jj]) > vmax * 0.55 else "#222")
ax2.set_xticks(range(N)); ax2.set_xticklabels(["低", "中低", "中", "中高", "高"], fontsize=8)
ax2.set_yticks(range(N)); ax2.set_yticklabels(["负", "偏负", "中", "偏正", "正"], fontsize=8)
ax2.set_xlabel(f"{disp(a_n)} 5 档 →", fontsize=9); ax2.set_ylabel(f"{disp(b_n)} 5 档 →", fontsize=9)
ax2.set_title("同表粗化成 5×5(便于读)\n两轴各分 5 档 → 25 个组合各有独立贡献", fontsize=9)
fig.colorbar(im2, ax=ax2, fraction=0.046, label="log-odds 贡献")
fig.suptitle(f"{L}M EBM 交互项 g({disp(a_n)}, {disp(b_n)}) — 2D 查表,非外积(红=升险/蓝=降险)", fontsize=11)
fig.tight_layout()
fig.savefig(OUTPNG, dpi=150, bbox_inches="tight")
print("写出:", OUTPNG.name, "| 粗化 5×5 =\n", np.round(C, 3))
