"""
Paper A · RQ4 信号可信度指标 (credibility metrics)
计算两件事:
  (1) 置信度校准 (ECE) — LLM 自报 confidence 应与实际命中率一致
  (2) Trigger 忠实性 (faithfulness) — trigger_phrases 必须逐字出现在原文中

零第三方依赖。E:\\anaconda\\python.exe 可直接跑。

用法:
  # 校准: 需要把每个信号是否命中 GT 的标签准备好
  python metrics.py --signals signals.jsonl --gt gt_events.json --out metrics_report.json

  # 忠实性: 需要原文 (news.jsonl)，按 signal_id/input_id 关联
  python metrics.py --signals signals.jsonl --news news.jsonl --out faithfulness_report.json

  # 合一
  python metrics.py --signals signals.jsonl --gt gt_events.json --news news.jsonl --out full_metrics.json

输出: JSON 报告 + stderr 摘要。
"""
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FOREWARD_TEMPORALITIES = {"forward_looking", "latent"}
N_BINS = 10  # 置信度分箱数


def _build_article_event_map(articles_path: Path, gt_events: list[dict],
                             post_days: int = 14) -> dict[str, str]:
    """article_id -> event_id via article-date containment in onset window.

    fetch_background.py --mode post-event fetches each event's
    [onset, onset+post_days] window separately, so every article's `date`
    falls in exactly one event's window. This is the RELIABLE attribution
    method (URL-keyword matching suffers suez/red_sea & chip-term collisions).
    """
    if not Path(articles_path).exists():
        return {}
    windows = []
    for ev in gt_events:
        try:
            onset = datetime.strptime(ev["gt_onset_date"][:10], "%Y-%m-%d")
            windows.append((onset, onset + timedelta(days=post_days), ev["event_id"]))
        except (KeyError, ValueError):
            continue
    id2event: dict[str, str] = {}
    with open(articles_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            try:
                ad = datetime.strptime(rec["date"][:10], "%Y-%m-%d")
            except (KeyError, ValueError):
                continue
            for onset, end, eid in windows:
                if onset <= ad <= end:
                    id2event[rec["id"]] = eid
                    break
    return id2event


# ---------- I/O ----------
def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ---------- 校准: ECE ----------
def compute_ece(items: list[dict], n_bins: int = N_BINS) -> dict:
    """
    items: [{"confidence": float, "correct": bool}, ...]
    返回 {"ece": float, "bins": [{bin_id, range, n, acc, avg_conf, gap}]}
    """
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    bins = []
    total = len(items)
    ece = 0.0
    for m in range(n_bins):
        lo, hi = bin_edges[m], bin_edges[m + 1]
        # 左闭右开，最后一箱右闭
        if m == n_bins - 1:
            members = [it for it in items if lo <= it["confidence"] <= hi]
        else:
            members = [it for it in items if lo <= it["confidence"] < hi]
        n_m = len(members)
        if n_m == 0:
            bins.append({"bin_id": m + 1, "range": [round(lo, 2), round(hi, 2)],
                         "n": 0, "acc": None, "avg_conf": None, "gap": None})
            continue
        acc_m = sum(1 for it in members if it["correct"]) / n_m
        conf_m = sum(it["confidence"] for it in members) / n_m
        gap = abs(acc_m - conf_m)
        ece += (n_m / total) * gap if total else 0
        bins.append({"bin_id": m + 1, "range": [round(lo, 2), round(hi, 2)],
                     "n": n_m, "acc": round(acc_m, 3), "avg_conf": round(conf_m, 3),
                     "gap": round(gap, 3)})
    return {"ece": round(ece, 4), "n_total": total, "bins": bins}


def label_correctness(signals: list[dict], gt_events: list[dict],
                       window_days: int = 180) -> list[dict]:
    """
    为每条信号打 correct 标签:
      True  — 该信号命中了至少一个 GT 事件 (与 validate.py 同一匹配规则，宽松版)
              (校准分析用宽松匹配，避免假负例低估准确率)
      False — 否则
    只对 temporality in {forward_looking, latent} 的信号打标签 (confirmation 必然不是预警)。
    """
    from validate import signal_matches_event  # 复用 validate.py 的匹配逻辑
    labeled = []
    for s in signals:
        if s.get("temporality") not in FOREWARD_TEMPORALITIES:
            continue
        correct = False
        lead_days = None
        try:
            sd = datetime.strptime(s["signal_date"][:10], "%Y-%m-%d")
        except (KeyError, ValueError):
            continue
        for ev in gt_events:
            try:
                onset = datetime.strptime(ev["gt_onset_date"][:10], "%Y-%m-%d")
            except (KeyError, ValueError):
                continue
            if not (onset - timedelta(days=window_days) <= sd < onset):
                continue
            if signal_matches_event(s, ev, strict=False):  # 宽松，避免假负例
                correct = True
                lead_days = (onset - sd).days
                break
        labeled.append({"signal_id": s.get("signal_id", ""),
                        "confidence": float(s.get("confidence", 0)),
                        "correct": correct,
                        "lead_days": lead_days})
    return labeled


# ---------- 创新 1: BSCC 背景扣除（时间污染）诊断 ----------
def compute_bscc(background_signals: list[dict], gt_events: list[dict],
                 post_days: int = 14, id2event: dict | None = None) -> dict:
    """
    Background-Subtracted (temporal) Contamination diagnostic.

    负对照 = 每个 GT 事件 onset 之后 [onset+1d, onset+post_days] 的 GDELT 新闻
    （由 pipeline/fetch_background.py --mode post-event 采集，事件专属词过滤）。
    这些文章是事后报道，LLM 应标 temporality=confirmation；
    若标为 forward_looking/latent 即时间虚构（后见之明污染 / hindsight bias）。

    β₀ = P(forward_looking ∪ latent | strictly post-onset signals)

    设计动机（为什么不是置信度减法）:
    旧版假设"背景输入诱发虚假高置信度"，对 30B+ 模型不成立——
    中性池 p₀≈0（正确拒抽），议题相邻池 p₀≈0.85（正确抽取真中断），
    没有探针能给出有用的中间 p₀。但时间判断错误是真实存在的失败模式：
    37.5% 的严格事后信号被误标为前瞻/潜在。BSCC 因此从"置信度减法"
    重构为"时间污染诊断"，β₀ 量化了需要被 FWGS 事前权重校正的污染程度。

    background_signals: LLM 从 post-event 池抽取的信号（含 temporality, signal_date, confidence）
    gt_events: GT 事件列表（含 gt_onset_date）
    post_days: 严格事后窗口上限（默认 14 天）
    """
    onsets = {}
    for ev in gt_events:
        try:
            onsets[ev["event_id"]] = datetime.strptime(ev["gt_onset_date"][:10], "%Y-%m-%d")
        except (KeyError, ValueError):
            continue

    strictly_post = []   # sd > onset 且 sd ≤ onset+post_days（明确事后）
    onset_day = []       # sd == onset（歧义：onset 公告 + 下游预测，单独报告）
    for s in background_signals:
        if not s.get("temporality"):
            continue
        try:
            sd = datetime.strptime(s["signal_date"][:10], "%Y-%m-%d")
        except (KeyError, ValueError):
            continue
        matched = False
        # 归属: 优先用 id2event（按文章日期→事件 onset 窗口，可靠）；
        # 否则回退到扫描所有事件 onset（仅聚合 β₀ 巧合正确，per-event 会错）
        cand_events = []
        if id2event is not None:
            ev_id = id2event.get(s.get("input_id", ""))
            if ev_id and ev_id in onsets:
                cand_events = [ev_id]
        if not cand_events:
            cand_events = list(onsets.keys())
        for ev_id in cand_events:
            onset = onsets[ev_id]
            if sd == onset:
                onset_day.append(s)
                matched = True
                break
            if onset < sd <= onset + timedelta(days=post_days):
                strictly_post.append(s)
                matched = True
                break
        # 信号日期不在其事件窗口内则忽略（跨事件污染，样本量太小不分析）

    if not strictly_post:
        return {"bscc": None, "reason": "no_strictly_post_onset_signals",
                "n_onset_day_ambiguous": len(onset_day)}

    from collections import Counter
    temporal_counts = Counter(s["temporality"] for s in strictly_post)
    n_mismatch = temporal_counts.get("forward_looking", 0) + temporal_counts.get("latent", 0)
    n_confirm = temporal_counts.get("confirmation", 0)
    n_total = len(strictly_post)
    beta_0 = n_mismatch / n_total

    mismatch_sigs = [s for s in strictly_post
                     if s["temporality"] in ("forward_looking", "latent")]
    confirm_sigs = [s for s in strictly_post if s["temporality"] == "confirmation"]

    def _mean_conf(lst):
        return round(sum(float(s.get("confidence", 0)) for s in lst) / len(lst), 4) if lst else None

    return {
        "beta_0_hindsight_rate": round(beta_0, 4),
        "n_strictly_post_onset": n_total,
        "n_temporal_mismatch": n_mismatch,
        "n_confirmation_correct": n_confirm,
        "n_onset_day_ambiguous": len(onset_day),
        "mean_confidence_mismatch": _mean_conf(mismatch_sigs),
        "mean_confidence_correct": _mean_conf(confirm_sigs),
        "temporal_distribution": dict(temporal_counts),
        "post_days_window": post_days,
    }


# ---------- 创新 2: TS-ECE 时间分层校准 ----------
def compute_ts_ece(items_with_lead: list[dict],
                   strata: list[tuple] = None,
                   n_bins: int = N_BINS) -> dict:
    """
    Time-Stratified ECE. 按 lead-time 分箱分别计算 ECE,
    揭示模型预测可靠性如何随前瞻窗口衰减。

    items_with_lead: [{confidence, correct, lead_days}]
    strata: 默认 (0,7], (7,30], (30,90], (90,180] 天
    """
    if strata is None:
        strata = [(0, 7), (7, 30), (30, 90), (90, 180)]
    curve = []
    for lo, hi in strata:
        members = [it for it in items_with_lead
                   if it.get("lead_days") is not None and lo < it["lead_days"] <= hi]
        if not members:
            curve.append({"stratum": f"({lo},{hi}]", "n": 0,
                          "ece": None, "mean_conf": None, "accuracy": None})
            continue
        ece_info = compute_ece(members, n_bins)
        curve.append({
            "stratum": f"({lo},{hi}]",
            "n": len(members),
            "ece": ece_info["ece"],
            "mean_conf": round(sum(it["confidence"] for it in members) / len(members), 3),
            "accuracy": round(sum(1 for it in members if it["correct"]) / len(members), 3),
        })
    return {"ts_ece_curve": curve,
            "n_with_lead": sum(c["n"] for c in curve),
            "n_no_lead": sum(1 for it in items_with_lead if it.get("lead_days") is None)}


# ---------- 创新 3: FWGS 前瞻加权忠实得分 ----------
def compute_fwgs(signals: list[dict], news_by_id: dict, gt_events: list[dict],
                 window_days: int = 180, tau: float = 30.0,
                 lambda_penalty: float = 0.5) -> dict:
    """
    Foresight-Weighted Grounded Score.
    单一综合得分，融合前瞻权重 + trigger 忠实度 + 准确性:

      FWGS = mean_i [ w_F(ℓ_i) · g_i · 1[TP_i] - λ · (1-g_i) · conf_i ]

      w_F(ℓ) = 1 - exp(-ℓ/τ)         # 前瞻权重，指数饱和
      g_i = |{trigger ∈ source_text}| / |triggers|   # IDF 可选加权
    """
    import math
    from validate import signal_matches_event
    scores = []
    n_tp = n_fp = 0
    for s in signals:
        if s.get("temporality") not in FOREWARD_TEMPORALITIES:
            continue
        try:
            sd = datetime.strptime(s["signal_date"][:10], "%Y-%m-%d")
        except (KeyError, ValueError):
            continue
        best_lead = None
        is_tp = False
        for ev in gt_events:
            try:
                onset = datetime.strptime(ev["gt_onset_date"][:10], "%Y-%m-%d")
            except (KeyError, ValueError):
                continue
            lead = (onset - sd).days
            if 0 < lead <= window_days and signal_matches_event(s, ev, strict=False):
                is_tp = True
                if best_lead is None or lead > best_lead:
                    best_lead = lead
        if is_tp:
            n_tp += 1
        else:
            n_fp += 1
        lead = best_lead if best_lead is not None else 0
        w_f = 1 - math.exp(-lead / tau)
        triggers = s.get("trigger_phrases", []) or []
        article = news_by_id.get(s.get("input_id", ""), {})
        text = ((article.get("title", "") or "") + " "
                + (article.get("text", "") or "")).lower()
        if triggers:
            hits = sum(1 for t in triggers if t.lower() in text)
            g = hits / len(triggers)
        else:
            g = 0.0
        conf = float(s.get("confidence", 0))
        score = w_f * g * (1 if is_tp else 0) - lambda_penalty * (1 - g) * conf
        scores.append(score)
    fwgs = sum(scores) / len(scores) if scores else None
    return {
        "fwgs": round(fwgs, 4) if fwgs is not None else None,
        "n_signals": len(scores),
        "n_tp": n_tp, "n_fp": n_fp,
        "tau": tau, "lambda_penalty": lambda_penalty,
    }


# ---------- 忠实性: trigger verbatim ----------
def faithfulness_audit(signals: list[dict], news_by_id: dict) -> dict:
    """
    每条信号的 trigger_phrases 必须逐字出现在原文中 (大小写不敏感)。
    返回 {"faithfulness_rate", "n_total", "n_faithful", "violations": [{signal_id, missing}]}
    """
    n_total = 0
    n_faithful = 0
    violations = []
    for s in signals:
        triggers = s.get("trigger_phrases", []) or []
        if not triggers:
            continue
        article = news_by_id.get(s.get("input_id", ""), {})
        # 原文 = title + text，都搜
        text = ((article.get("title", "") or "") + " " + (article.get("text", "") or "")).lower()
        missing = []
        for tr in triggers:
            if tr.lower() not in text:
                missing.append(tr)
        n_total += 1
        if not missing:
            n_faithful += 1
        else:
            violations.append({"signal_id": s.get("signal_id", ""),
                               "input_id": s.get("input_id", ""),
                               "missing_triggers": missing})
    rate = (n_faithful / n_total) if n_total else None
    return {"faithfulness_rate": round(rate, 4) if rate is not None else None,
            "n_total": n_total, "n_faithful": n_faithful,
            "n_violations": len(violations), "violations": violations}


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper A RQ4 信号可信度指标")
    ap.add_argument("--signals", required=True, help="signals.jsonl (extract_events.py 输出)")
    ap.add_argument("--gt", default=None, help="gt_events.json (启用校准)")
    ap.add_argument("--news", default=None, help="news.jsonl 原文 (启用忠实性/FWGS)")
    ap.add_argument("--background-signals", default=None,
                    help="黑天鹅探针信号 JSONL (启用 BSCC 背景扣除校准)")
    ap.add_argument("--fwgs-tau", type=float, default=30.0, help="FWGS 前瞻饱和时间常数 (天)")
    ap.add_argument("--fwgs-lambda", type=float, default=0.5, help="FWGS 幻觉惩罚系数 λ")
    ap.add_argument("--out", default=None, help="输出 JSON 报告路径")
    ap.add_argument("--window", type=int, default=180, help="预警窗口 (校准用)")
    args = ap.parse_args()

    signals = [s for s in load_jsonl(args.signals) if s.get("status") != "error"]
    report = {"n_signals_loaded": len(signals)}

    # 1) 校准
    if args.gt:
        with open(args.gt, encoding="utf-8") as f:
            gt_data = json.load(f)
        gt_events = gt_data.get("events", gt_data) if isinstance(gt_data, dict) else gt_data
        labeled = label_correctness(signals, gt_events, window_days=args.window)
        calibration = compute_ece(labeled)
        # 高 vs 低置信度区分力
        high = [it for it in labeled if it["confidence"] >= 0.8]
        low = [it for it in labeled if it["confidence"] < 0.5]
        high_acc = (sum(1 for it in high if it["correct"]) / len(high)) if high else None
        low_acc = (sum(1 for it in low if it["correct"]) / len(low)) if low else None
        calibration["high_conf_acc"] = round(high_acc, 3) if high_acc is not None else None
        calibration["low_conf_acc"] = round(low_acc, 3) if low_acc is not None else None
        calibration["n_high_conf"] = len(high)
        calibration["n_low_conf"] = len(low)
        report["calibration"] = calibration

    # 2) 忠实性
    news_by_id = {}
    if args.news:
        news = load_jsonl(args.news)
        news_by_id = {a.get("id", ""): a for a in news}
        report["faithfulness"] = faithfulness_audit(signals, news_by_id)

    # 3) BSCC 时间污染诊断 (创新 1) — 用 post-event 背景池测后见之明率
    if args.background_signals and args.gt:
        bg_signals = [s for s in load_jsonl(args.background_signals)
                      if s.get("status") != "error"]
        # 按文章日期→事件 onset 窗口构建 id2event（可靠归属，避免 URL 关键词冲突）
        bg_articles_path = ROOT / "data" / "by_event" / "_background.jsonl"
        id2event = _build_article_event_map(bg_articles_path, gt_events, post_days=14)
        report["bscc"] = compute_bscc(bg_signals, gt_events, post_days=14, id2event=id2event)

    # 4) TS-ECE 时间分层校准 (创新 2)
    if args.gt and labeled:
        report["ts_ece"] = compute_ts_ece(labeled)

    # 5) FWGS 前瞻加权忠实得分 (创新 3)
    if args.gt and args.news:
        report["fwgs"] = compute_fwgs(
            signals, news_by_id, gt_events, window_days=args.window,
            tau=args.fwgs_tau, lambda_penalty=args.fwgs_lambda)

    out_text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_text)
    else:
        print(out_text)

    # 摘要
    print("\n=== RQ4 CREDIBILITY METRICS ===", file=sys.stderr)
    if "calibration" in report:
        c = report["calibration"]
        print(f"  ECE = {c['ece']} (N={c['n_total']})", file=sys.stderr)
        print(f"  high-conf(c≥0.8) acc = {c['high_conf_acc']} (n={c['n_high_conf']})", file=sys.stderr)
        print(f"  low-conf (c<0.5)  acc = {c['low_conf_acc']}  (n={c['n_low_conf']})", file=sys.stderr)
    if "faithfulness" in report:
        f0 = report["faithfulness"]
        print(f"  faithfulness = {f0['faithfulness_rate']} ({f0['n_faithful']}/{f0['n_total']}, "
              f"{f0['n_violations']} violations)", file=sys.stderr)
    if "bscc" in report and report["bscc"].get("beta_0_hindsight_rate") is not None:
        b = report["bscc"]
        print(f"  BSCC: β₀={b['beta_0_hindsight_rate']} (n_post={b['n_strictly_post_onset']}, "
              f"mismatch={b['n_temporal_mismatch']}, correct={b['n_confirmation_correct']}, "
              f"onset_day_ambig={b['n_onset_day_ambiguous']})", file=sys.stderr)
        print(f"        mean_conf: mismatch={b['mean_confidence_mismatch']} "
              f"vs correct={b['mean_confidence_correct']}", file=sys.stderr)
    if "ts_ece" in report:
        print(f"  TS-ECE curve:", file=sys.stderr)
        for s in report["ts_ece"]["ts_ece_curve"]:
            if s["n"]:
                print(f"    {s['stratum']:>10}: n={s['n']:>3}  ECE={s['ece']}  "
                      f"acc={s['accuracy']}  conf={s['mean_conf']}", file=sys.stderr)
    if "fwgs" in report and report["fwgs"]["fwgs"] is not None:
        fw = report["fwgs"]
        print(f"  FWGS = {fw['fwgs']} (N={fw['n_signals']}, TP={fw['n_tp']}, FP={fw['n_fp']}, "
              f"τ={fw['tau']}, λ={fw['lambda_penalty']})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
