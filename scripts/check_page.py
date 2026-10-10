"""Check docs/index.html against metrics/*.json and DECISIONS.md. Exit 1 on any mismatch.

Does not import build_page.py: it reads the sources again with its own lookups, then checks
(1) the embedded JSON block, (2) that every shown number is the right rounding of a source
number, (3) that the page shows no dollar / AUC / percent figure that is not on that list,
(4) no external loads and no banned words.

Run from the repo root: python3 scripts/check_page.py
"""
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
html = (ROOT / "docs" / "index.html").read_text()
load = lambda n: json.loads((ROOT / "metrics" / f"{n}.json").read_text())
test, weekly, retr = load("phase5_test"), load("phase6_weekly"), load("phase8_retrain")
dec = (ROOT / "DECISIONS.md").read_text()
errors = []


def need(cond, msg):
    if not cond:
        errors.append(msg)


# ---- expected values, read straight from the sources ----
flat = {}  # json-path -> source value
flat["c"] = test["headline"]["c"]
flat["auc.lgbm"] = test["auc"]["lgbm"]["roc_auc"]["mean"]
flat["auc.nn"] = test["auc"]["nn"]["roc_auc"]["mean"]
flat["fraud_dollars"] = test["amounts"]["fraud_dollars"]
flat["precision"] = test["headline"]["precision"]
flat["flags_per_day"] = test["headline"]["flags_per_day"]
for i, v in enumerate([test["headline"]["savings"], *test["bootstrap"]["savings_at_c10_95"]["lgbm/calibrated"]]):
    flat[f"savings.lgbm.{i}"] = v
for i, v in enumerate([test["costs"]["nn"]["calibrated"]["10"]["savings"]["mean"],
                       *test["bootstrap"]["savings_at_c10_95"]["nn/calibrated"]]):
    flat[f"savings.nn.{i}"] = v
need(test["headline"]["model"] == "lgbm" and test["headline"]["rule"] == "calibrated" and test["headline"]["c"] == 10,
     "tile labels assume headline = LightGBM, calibrated rule, $10")
wk = {w["week"]: w for w in weekly["weeks"]}
for j, n in enumerate(weekly["post_weeks"]):
    for m in ("lgbm", "nn"):
        for i, v in enumerate([wk[n][m]["roc_auc"], *wk[n][m]["roc_auc_95"]]):
            flat[f"weekly.{j}.{m}.{i}"] = v
    flat[f"weekly.{j}.week"] = n
flat["val_weeks"], flat["test_weeks"] = weekly["val_weeks"], weekly["test_weeks"]
flat["eval_days"], flat["eval_fraud_dollars"] = retr["eval_days"], retr["eval_fraud_dollars"]
for dl in ("7", "30", "60"):
    for m in ("lgbm", "nn"):
        c = retr["cells"][m][dl]
        for key, src in (("never", c["never"]["savings"]["mean"]), ("four", c["4-weekly"]["savings"]["mean"])):
            flat[f"retrain.{dl}.{m}.{key}.0"] = src
        for i, v in enumerate(c["never"]["savings_95"]):
            flat[f"retrain.{dl}.{m}.never.{i + 1}"] = v
        for i, v in enumerate(c["4-weekly"]["savings_95"]):
            flat[f"retrain.{dl}.{m}.four.{i + 1}"] = v
        g = c["4-weekly"]["gain_vs_never"]
        for i, v in enumerate([g["mean"], *g["range_95"]]):
            flat[f"retrain.{dl}.{m}.gain.{i}"] = v
gap = re.search(r"^\| new \| [\d.]+ \| [\d.]+ \| (\d\.\d+) ", dec, re.M)
need(gap is not None, "DECISIONS.md: new-clients gap line not found")
flat["new_client_gap"] = float(gap.group(1))
days = re.search(r"at D = 60, days 0-(\d+)", dec)
need(days is not None, "DECISIONS.md: 'days 0-37' text not found")
flat["nn_d60_days"] = int(days.group(1)) + 1
need(retr["cells"]["nn"]["60"]["never"]["retrain_days"] == [retr["r0"]], "nn D=60 never-model is not trained once at r0")
need("A3." in dec and "week 17" in dec, "DECISIONS.md: audit item A3 not found")
for m in ("lgbm", "nn"):
    best = max(weekly["post_weeks"], key=lambda n: wk[n][m]["roc_auc"])
    need(best == 17, f"week 17 is not the best week for {m}")
need(weekly["val_weeks"] == [17, 18, 19] and weekly["test_weeks"] == [22, 23, 24, 25],
     "week bands on the chart (20-21 = gap) assume val 17-19, test 22-25")
lo1, hi1 = test["bootstrap"]["savings_at_c10_95"]["lgbm/calibrated"]
lo2, hi2 = test["bootstrap"]["savings_at_c10_95"]["nn/calibrated"]
need(max(lo1, lo2) <= min(hi1, hi2), "savings ranges no longer overlap: the 'ranges overlap' line is wrong")

# ---- (1) embedded JSON block ----
m = re.search(r'<script type="application/json" id="page-data">(.*?)</script>', html, re.S)
need(m is not None, "no embedded JSON block")
data = json.loads(m.group(1)) if m else {}


def walk(path):
    cur = data
    for p in path.split("."):
        cur = cur[int(p)] if isinstance(cur, list) else cur[p]
    return cur


def resolve(path):  # weekly.J.model.I / weekly.J.week come from a list of dicts
    return walk(path)


for path, src in flat.items():
    try:
        got = resolve(path)
    except (KeyError, IndexError, ValueError, TypeError):
        errors.append(f"JSON block missing {path}")
        continue
    ok = got == src if isinstance(src, list) else math.isclose(got, src, rel_tol=1e-9, abs_tol=1e-12)
    need(ok, f"JSON {path}: page {got} != source {src}")

# ---- (2)+(3) shown numbers ----
k = lambda x: f"${round(x / 1e3):,}k"
allowed = {"$0", "$400k", "$800k", "$1.2M", f"${flat['c']}", "95%"}
for path, v in flat.items():
    if isinstance(v, list) or path in ("c", "eval_days", "nn_d60_days") or ".week" in path or path.endswith("week"):
        continue
    if path.startswith(("auc", "weekly")):
        allowed.add(f"{v:.3f}")
    elif path == "new_client_gap":
        allowed.add(f"{v:.3f}")
    elif path == "precision":
        allowed.add(f"{round(v * 100)}%")
    elif path == "flags_per_day":
        pass
    else:
        allowed.add(k(abs(v)))
text = re.sub(r"<(script|style)\b.*?</\1>", "", html, flags=re.S)
text = re.sub(r"<[^>]+>", " ", text)
shown = set(re.findall(r"\$[\d,.]+[kM]?|\b\d\.\d{3}\b|\b\d+%", text))
for tok in sorted(shown - allowed):
    errors.append(f"number on page not traced to a source: {tok}")
for tok in sorted(allowed - shown - {"$0", "$400k", "$800k", "$1.2M"}):
    errors.append(f"expected number missing from page: {tok}")
need(str(round(flat["flags_per_day"])) in text, "flags per day not shown")
need(f"{flat['new_client_gap']:.3f}" in text, "new-client gap not shown")
need(str(flat["nn_d60_days"]) + " days of data" in text, "38-day footnote missing")

# ---- (4) offline + wording ----
need(not re.search(r"<script[^>]+src=|<link\b|@import|url\(|<img|<iframe", html, re.I), "external resource tag found")
urls = set(re.findall(r"https?://[^\s\"'<>]+", re.sub(r'<script type="application/json".*?</script>', "", html, flags=re.S)))
okurls = {"https://github.com/suvarchaw/fraud-nn-vs-lightgbm",
          "https://drive.google.com/file/d/1pAZ-GizcqRoUWba1TiCFQ3WVbssDKLM6/view?usp=sharing"}
need(urls == okurls, f"unexpected URLs: {urls ^ okurls}")
for bad in (r"accurate", r"production[- ]ready", r"claude", r"anthropic", r"\bAI\b", r"\bLLM\b", r"gpt", r"copilot", r"co-authored"):
    need(not re.search(bad, html, re.I), f"banned word: {bad}")
for must in ("assumed $10 review cost", "Intended use: a portfolio study of fraud scoring. Not for real payment decisions.",
             "retraining cost not counted", "do not show that either model saves more", "reuse months already looked at"):
    need(must in html, f"missing text: {must}")

if errors:
    print("\n".join("FAIL " + e for e in errors))
    sys.exit(1)
print(f"OK: {len(flat)} source values match the JSON block; {len(shown)} distinct shown figures all traced; offline and wording checks pass")
