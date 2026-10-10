"""Write docs/index.html from metrics/*.json (+ one figure from DECISIONS.md).

Presentation only: no analysis happens here. The page is static HTML + inline SVG
(no JavaScript, no external requests). scripts/check_page.py re-checks every number.

Run from the repo root: python3 scripts/build_page.py
"""
import json
import math
import re
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = lambda name: json.loads((ROOT / "metrics" / f"{name}.json").read_text())

REPO = "https://github.com/suvarchaw/fraud-nn-vs-lightgbm"
DEMO = "https://drive.google.com/file/d/1pAZ-GizcqRoUWba1TiCFQ3WVbssDKLM6/view?usp=sharing"
NAMES = {"lgbm": "LightGBM", "nn": "Network"}


def k(x):  # dollars, rounded to $k (to $x.xxM from a million up)
    return f"${x / 1e6:.2f}M" if x >= 1e6 else f"${round(x / 1e3):,}k"


def a3(x):  # AUC, 3 decimals
    return f"{x:.3f}"


def extract():
    t, w, r = M("phase5_test"), M("phase6_weekly"), M("phase8_retrain")
    dec = (ROOT / "DECISIONS.md").read_text()
    gap = re.search(r"\| new \| [\d.]+ \| [\d.]+ \| (\d\.\d+) ", dec)
    starved = re.search(r"at D = 60, days 0-(\d+)", dec)
    boot = t["bootstrap"]["savings_at_c10_95"]
    by_week = {x["week"]: x for x in w["weeks"]}
    weeks = [{"week": n, **{m: [by_week[n][m]["roc_auc"], *by_week[n][m]["roc_auc_95"]] for m in NAMES}}
             for n in w["post_weeks"]]
    for m in NAMES:  # the caption says week 17 is best for both; refuse to build if untrue
        assert max(weeks, key=lambda x: x[m][0])["week"] == 17, m
    retrain = {}
    for d in ("7", "30", "60"):
        retrain[d] = {}
        for m in NAMES:
            c = r["cells"][m][d]
            retrain[d][m] = {
                "never": [c["never"]["savings"]["mean"], *c["never"]["savings_95"]],
                "four": [c["4-weekly"]["savings"]["mean"], *c["4-weekly"]["savings_95"]],
                "gain": [c["4-weekly"]["gain_vs_never"]["mean"], *c["4-weekly"]["gain_vs_never"]["range_95"]],
            }
    h = t["headline"]
    assert h["model"] == "lgbm" and h["rule"] == "calibrated" and h["c"] == 10
    return {
        "c": h["c"],
        "auc": {m: t["auc"][m]["roc_auc"]["mean"] for m in NAMES},
        "fraud_dollars": t["amounts"]["fraud_dollars"],
        "savings": {
            "lgbm": [h["savings"], *boot["lgbm/calibrated"]],
            "nn": [t["costs"]["nn"]["calibrated"]["10"]["savings"]["mean"], *boot["nn/calibrated"]],
        },
        "precision": h["precision"],
        "flags_per_day": h["flags_per_day"],
        "val_weeks": w["val_weeks"],
        "test_weeks": w["test_weeks"],
        "weekly": weeks,
        "eval_days": r["eval_days"],
        "eval_fraud_dollars": r["eval_fraud_dollars"],
        "retrain": retrain,
        "new_client_gap": float(gap.group(1)),
        "nn_d60_days": int(starved.group(1)) + 1,
    }


# ---------- charts (inline SVG, colours from CSS classes) ----------

def weekly_svg(d):
    W, H, L, R, T, B = 640, 340, 56, 70, 40, 56
    lo = math.floor(min(x[m][1] for x in d["weekly"] for m in NAMES) * 50) / 50
    hi = math.ceil(max(x[m][2] for x in d["weekly"] for m in NAMES) * 50) / 50
    wk = [x["week"] for x in d["weekly"]]
    X = lambda n: L + (n - wk[0]) / (wk[-1] - wk[0]) * (W - L - R)
    Y = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="t-weekly d-weekly">'
         '<title id="t-weekly">Weekly ROC-AUC of both frozen models, weeks 17 to 25</title>']
    # background bands: validation / gap / test
    for name, ws, cls in (("Validation", d["val_weeks"], "bandA"), ("Gap", [20, 21], "bandB"),
                          ("Test", d["test_weeks"], "bandA")):
        x0, x1 = X(ws[0]) - 14, X(ws[-1]) + 14
        o.append(f'<rect class="{cls}" x="{x0:.1f}" y="{T}" width="{x1 - x0:.1f}" height="{H - T - B}"/>'
                 f'<text class="sm mid" x="{(x0 + x1) / 2:.1f}" y="{T - 8}">{name}</text>')
    v = lo
    while v <= hi + 1e-9:
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                 f'<text class="sm end" x="{L - 6}" y="{Y(v) + 4:.1f}">{v:.2f}</text>')
        v += 0.02
    for n in wk:
        o.append(f'<text class="sm mid" x="{X(n):.1f}" y="{H - B + 18}">{n}</text>')
    o.append(f'<text class="sm mid" x="{(L + W - R) / 2}" y="{H - 8}">Week number (week 17 = first validation week)</text>'
             f'<text class="sm mid" transform="translate(14 {(T + H - B) / 2}) rotate(-90)">ROC-AUC (higher is better)</text>')
    for m in NAMES:
        band = " ".join(f"{X(x['week']):.1f},{Y(x[m][2]):.1f}" for x in d["weekly"]) + " " + \
               " ".join(f"{X(x['week']):.1f},{Y(x[m][1]):.1f}" for x in reversed(d["weekly"]))
        line = " ".join(f"{X(x['week']):.1f},{Y(x[m][0]):.1f}" for x in d["weekly"])
        o.append(f'<polygon class="band {m}f" points="{band}"/><polyline class="ln {m}" points="{line}"/>')
        for x in d["weekly"]:
            o.append(f'<circle class="dot {m}" cx="{X(x["week"]):.1f}" cy="{Y(x[m][0]):.1f}" r="4">'
                     f'<title>{NAMES[m]}, week {x["week"]}: ROC-AUC {a3(x[m][0])} '
                     f'(95% range {a3(x[m][1])} to {a3(x[m][2])})</title></circle>')
        last = d["weekly"][-1]
        o.append(f'<text class="sm lab {m}t" x="{X(last["week"]) + 12:.1f}" y="{Y(last[m][0]) + 4:.1f}">{NAMES[m]}</text>')
    o.append("</svg>")
    return "".join(o)


def weekly_table(d):
    rows = "".join(f"<tr><td>{x['week']}</td>" + "".join(
        f"<td>{a3(x[m][0])} ({a3(x[m][1])} to {a3(x[m][2])})</td>" for m in NAMES) + "</tr>" for x in d["weekly"])
    return ("<details><summary>Data table</summary><table><thead><tr><th>Week</th><th>LightGBM ROC-AUC (95% range)</th>"
            f"<th>Network ROC-AUC (95% range)</th></tr></thead><tbody>{rows}</tbody></table></details>")


def retrain_gain_svg(d):
    W, L, RT, ROW, TOP = 400, 10, 300, 40, 80e3
    ticks = (0, 20e3, 40e3, 60e3, 80e3)
    assert max(d["retrain"][dl][m]["gain"][2] for dl in d["retrain"] for m in NAMES) <= TOP, "axis would clip"
    X = lambda v: L + v / TOP * (RT - L)
    H = 3 * (24 + 2 * ROW + 10) + 40
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="t-gain d-gain">'
         '<title id="t-gain">Extra savings from retraining every 4 weeks over never retraining, by label delay</title>']
    for t in ticks:
        o.append(f'<line class="grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="0" y2="{H - 40}"/>'
                 f'<text class="sm mid" x="{X(t):.1f}" y="{H - 24}">{"$0" if t == 0 else k(t)}</text>')
    o.append(f'<text class="sm mid" x="{(L + RT) / 2}" y="{H - 6}">Extra savings over never retraining</text>')
    y = 0
    for dl in d["retrain"]:
        o.append(f'<text class="grp" x="{L}" y="{y + 16}">Labels arrive {dl} days late</text>')
        y += 24
        for m in NAMES:
            mean, lo, hi = d["retrain"][dl][m]["gain"]
            star = "*" if (dl == "60" and m == "nn") else ""
            cy = y + 24
            o.append(f'<text class="sm" x="{L}" y="{y + 8}">{NAMES[m]}{star}</text>'
                     f'<line class="whisk" x1="{X(lo):.1f}" x2="{X(hi):.1f}" y1="{cy}" y2="{cy}"/>'
                     f'<circle class="dot {m}" cx="{X(mean):.1f}" cy="{cy}" r="5"/>'
                     f'<text class="val" x="{X(hi) + 8:.1f}" y="{cy + 4}">+{k(mean)}</text>')
            y += ROW
        y += 10
    o.append("</svg>")
    return "".join(o)


def pipeline_svg(c):
    W, BW, BH, G = 320, 260, 34, 22
    steps = [("Data (IEEE-CIS payments)",), ("Time-based split",), ("LightGBM", "Network"),
             (f"Calibration and ${c} cost rule",), ("Frozen model",), ("API in Docker",)]
    H = len(steps) * (BH + G) - G + 4
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="t-pipe d-pipe">'
         '<title id="t-pipe">Pipeline from data to the API</title>']
    for i, row in enumerate(steps):
        y = 2 + i * (BH + G)
        n, w = len(row), (BW - 12 * (len(row) - 1)) / len(row)
        for j, label in enumerate(row):
            x = (W - BW) / 2 + j * (w + 12)
            o.append(f'<rect class="box" x="{x:.1f}" y="{y}" width="{w:.1f}" height="{BH}" rx="6"/>'
                     f'<text class="mid" x="{x + w / 2:.1f}" y="{y + 21}">{escape(label)}</text>')
        if i < len(steps) - 1:  # one arrow per box of whichever neighbouring row has more boxes
            m_ = max(n, len(steps[i + 1]))
            w2 = (BW - 12 * (m_ - 1)) / m_
            for j in range(m_):
                x = (W - BW) / 2 + j * (w2 + 12) + w2 / 2
                o.append(f'<line class="arr" x1="{x:.1f}" x2="{x:.1f}" y1="{y + BH}" y2="{y + BH + G - 6}"/>'
                         f'<path class="arrh" d="M{x - 4:.1f} {y + BH + G - 10} L{x:.1f} {y + BH + G - 3} L{x + 4:.1f} {y + BH + G - 10}"/>')
    o.append("</svg>")
    return "".join(o)


def retrain_text(d, dl):
    g = lambda m: d["retrain"][dl][m]["gain"]
    parts = [f"{NAMES[m]} {'+' if g(m)[0] >= 0 else '-'}{k(abs(g(m)[0]))} ({k(g(m)[1])} to {k(g(m)[2])})" for m in NAMES]
    return "Extra from retraining every 4 weeks: " + "; ".join(parts) + "."


def retrain_table(d, dl):
    rows = ""
    for m in NAMES:
        c = d["retrain"][dl][m]
        rows += (f"<tr><td>{NAMES[m]}</td><td>{k(c['never'][0])} ({k(c['never'][1])} to {k(c['never'][2])})</td>"
                 f"<td>{k(c['four'][0])} ({k(c['four'][1])} to {k(c['four'][2])})</td></tr>")
    return ("<details><summary>Data table</summary><table><thead><tr><th>Model</th><th>Never retrain</th>"
            f"<th>Retrain every 4 weeks</th></tr></thead><tbody>{rows}</tbody></table></details>")


CSS = """
:root{--bg:#fafaf9;--fg:#1c1c1a;--mut:#5f5f5a;--card:#fff;--line:#d9d9d4;--acc:#1f6f8b;--grey:#6b6b66;--band:#f0f0ec;--band2:#e6e6e0}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ececea;--mut:#a3a39d;--card:#1f1f1d;--line:#3a3a36;--acc:#5bb4d3;--grey:#a8a8a2;--band:#1d1d1b;--band2:#272725}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main,header,footer{max-width:1040px;margin:0 auto;padding:0 16px}
header{padding-top:32px}h1{font-size:1.7rem;line-height:1.25;margin:0 0 8px}h2{font-size:1.25rem;margin:40px 0 8px}
a{color:var(--acc)}p{margin:8px 0}.mut{color:var(--mut)}.tag{display:inline-block;border:1px solid var(--line);border-radius:4px;padding:0 8px;font-size:.85rem;color:var(--mut)}
.tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:24px}
@media (max-width:640px){.tiles{grid-template-columns:1fr}}
.tile,.panel,.limits{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px}
.tile.wide{grid-column:1/-1;order:-1}
.found{margin:20px 0 0;padding:0 0 0 20px}.found li{margin:4px 0}
.narrow{max-width:560px}svg .grp{font-weight:600}.box{fill:var(--card);stroke:var(--line);stroke-width:1.5}.arr{stroke:var(--mut);stroke-width:1.5}.arrh{fill:none;stroke:var(--mut);stroke-width:1.5}
.tile h3{font-size:.9rem;font-weight:600;margin:0 0 8px;color:var(--mut)}
.big{font-size:1.9rem;font-weight:700;line-height:1.2}.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.lg{color:var(--acc)}
.panel h3{margin:0 0 4px;font-size:1rem}
svg{width:100%;height:auto;display:block}
svg text{fill:var(--fg);font-size:12px}svg .sm{font-size:12px;fill:var(--mut)}svg .mid{text-anchor:middle}svg .end{text-anchor:end}
.val{font-size:12px;font-weight:600}
.grid{stroke:var(--line);stroke-width:1}.bandA{fill:var(--band)}.bandB{fill:var(--band2)}
.ln{fill:none;stroke-width:2.5}.ln.lgbm{stroke:var(--acc)}.ln.nn{stroke:var(--grey);stroke-dasharray:6 4}
.band{stroke:none;opacity:.16}.band.lgbmf{fill:var(--acc)}.band.nnf{fill:var(--grey)}
.dot.lgbm{fill:var(--acc)}.dot.nn{fill:var(--grey)}svg .lgbmt{fill:var(--acc);font-weight:600}svg .nnt{fill:var(--grey);font-weight:600}
.bar.lgbm{fill:var(--acc)}.bar.nn{fill:var(--grey)}.bar.never{opacity:.5}.whisk{stroke:var(--fg);stroke-width:1.5}
details{margin-top:8px}summary{cursor:pointer;color:var(--mut);font-size:.9rem}
table{border-collapse:collapse;font-size:.85rem;margin-top:8px;width:100%}th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left}
.note{font-size:.9rem;color:var(--mut)}.limits ul{margin:8px 0;padding-left:20px}.limits li{margin:6px 0}
.intended{font-weight:600;margin-top:12px}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
footer{padding:32px 16px 48px;font-size:.9rem;color:var(--mut)}
"""


def page(d):
    s, au = d["savings"], d["auc"]
    bnd = lambda v: f"{k(v[0])}</div><div class='note'>95% range {k(v[1])} to {k(v[2])}</div>"
    lg_best = max(d["weekly"], key=lambda x: x["lgbm"][0])["week"]
    foot = (f'<p class="note">* The network that never retrains at 60 days learned from only {d["nn_d60_days"]} days of data, '
            "so it is a weak baseline and the network's gain there is exaggerated.</p>")
    tables = "".join(f'<h3>Labels arrive {dl} days late</h3><p class="note">{escape(retrain_text(d, dl))}</p>{retrain_table(d, dl)}'
                     for dl in d["retrain"])
    sr_gain = " ".join(retrain_text(d, dl).replace("Extra from", f"At {dl} days, extra from") for dl in d["retrain"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fraud scoring study results</title>
<meta name="description" content="Results of a portfolio study comparing LightGBM and a neural network for card-fraud scoring, scored by money saved under an assumed review cost.">
<style>{CSS}</style></head><body>
<header>
<h1>LightGBM vs a neural network for card-fraud scoring</h1>
<p>A time-split study on the IEEE-CIS data, scored by money saved under an assumed ${d['c']} review cost per flagged payment; offline, not a live system.</p>
<p><a href="{REPO}">Code and write-up on GitHub</a> · <a href="{DEMO}">Demo video (about 90 seconds)</a></p>
<section aria-label="What I found"><h2>What I found</h2><ul class="found">
<li>LightGBM ranks payments better than the network, with ROC-AUC {a3(au['lgbm'])} vs {a3(au['nn'])} on the locked test block.</li>
<li>The network's savings range overlaps LightGBM's, so this project does not claim either model saves more dollars.</li>
<li>LightGBM's lead comes mostly from clients it has seen before; on new clients the gap is only about {d['new_client_gap']:.3f}.</li>
</ul></section>
</header>
<main>
<section class="tiles" aria-label="Headline numbers (locked test month)">
<div class="tile"><h3>Test ROC-AUC (mean of 5 seeds)</h3><div class="big"><span class="lg">{a3(au['lgbm'])}</span> vs {a3(au['nn'])}</div><div class="note">LightGBM vs network. Higher means the model ranks fraud above legitimate payments more often.</div></div>
<div class="tile wide"><h3>Money saved on the test month, of {k(d['fraud_dollars'])} fraud dollars (assumed ${d['c']} review cost)</h3>
<div class="two"><div><div class="note">LightGBM</div><div class="big">{bnd(s['lgbm'])}</div>
<div><div class="note">Network</div><div class="big">{bnd(s['nn'])}</div></div>
<p class="note">The ranges overlap, so these results do not show that either model saves more.</p></div>
<div class="tile"><h3>Precision of the flags</h3><div class="big">{round(d['precision'] * 100)}%</div><div class="note">About 1 in 5 flagged payments was fraud (LightGBM, ${d['c']} rule).</div></div>
<div class="tile"><h3>Flags per day</h3><div class="big">{round(d['flags_per_day'])}</div><div class="note">Payments flagged for review on an average test day (LightGBM, ${d['c']} rule).</div></div>
</section>

<h2>How it was built</h2>
<div class="panel"><div class="narrow" style="margin:0 auto">{pipeline_svg(d['c'])}</div>
<p class="sr" id="d-pipe">Flow from top to bottom: data, then a time-based split, then LightGBM and the network trained side by side, then calibration and a ${d['c']} cost rule, then a frozen model, then an API served in Docker.</p></div>

<h2>Weekly drift <span class="tag">exploratory</span></h2>
<p>Both models were frozen after training and scored week by week. Shaded bands are 95% ranges; background shows which weeks are validation, the gap between, and test.</p>
<div class="panel">{weekly_svg(d)}
<p class="sr" id="d-weekly">Line chart of weekly ROC-AUC for LightGBM and the network from week 17 to week 25. Both are highest in week 17 and lower afterwards. Weeks 17 to 19 are validation, weeks 20 and 21 are the gap, weeks 22 to 25 are test.</p>
{weekly_table(d)}</div>
<p class="note">Week 17 is the best-scoring week for both models ({a3(next(x for x in d['weekly'] if x['week'] == lg_best)['lgbm'][0])} for LightGBM, {a3(d['weekly'][0]['nn'][0])} for the network), and the fall afterwards leans on it (audit item A3 in DECISIONS.md), so the size of the fall is uncertain. Weeks 17 to 19 come from validation and weeks 22 to 25 from test.</p>

<h2>Retraining with delayed labels <span class="tag">exploratory</span> <span class="tag">retraining cost not counted</span></h2>
<p>Savings over one {d['eval_days']}-day stretch with {k(d['eval_fraud_dollars'])} of fraud, when a fraud label only arrives D days after the payment. Each dot is the mean extra savings (5 seeds) from retraining every 4 weeks instead of never retraining, with its 95% range as a line.</p>
<div class="panel"><div class="narrow">{retrain_gain_svg(d)}</div>
<p class="sr" id="d-gain">Dot chart of extra savings from retraining every 4 weeks over never retraining, for LightGBM and the network at label delays of 7, 30 and 60 days, with 95% ranges. {escape(sr_gain)}</p>
{foot}{tables}</div>

<h2>Limits</h2>
<div class="limits"><ul>
<li>The ${d['c']} review cost per flag is an assumption, not a measured figure; the savings depend on it.</li>
<li>LightGBM's lead comes mostly from returning clients: on new clients the ROC-AUC gap is about {d['new_client_gap']:.3f} (validation, checked after the fact).</li>
<li>The models are frozen; their ranking drifts down after training ends (see the weekly chart).</li>
<li>The retraining results cover one 12-week stretch, and those 12 weeks reuse months already looked at in earlier phases.</li>
<li>Retraining cost (compute, checks, release work) is not counted.</li>
<li>Everything is an offline simulation on public data; nothing was run on live payments.</li>
</ul><p class="intended">Intended use: a portfolio study of fraud scoring. Not for real payment decisions.</p></div>
</main>
<footer>
<p>Every number on this page is generated from the repository's metrics files by scripts/build_page.py and checked by scripts/check_page.py. The page holds no transaction rows, scores or models; the IEEE-CIS data is not redistributed.</p>
<p><a href="{REPO}">Repository</a></p>
</footer>
<script type="application/json" id="page-data">{json.dumps(d)}</script>
</body></html>
"""


if __name__ == "__main__":
    out = ROOT / "docs" / "index.html"
    out.write_text(page(extract()))
    print("wrote", out)
