"""
Paper A · RQ1 前视验证 (forward-looking validation)
将 LLM 抽取出的信号 (signals.jsonl) 与 GT 事件集 (gt_events.json) 对齐，
计算命中率 / lead time / 精度 / 召回。协议见 validation_protocol.md。

零第三方依赖。E:\\anaconda\\python.exe 可直接跑。

用法:
  python validate.py --signals signals.jsonl --gt gt_events.json --report report.json
  python validate.py --signals signals.jsonl --gt gt_events.json --window 180
  python validate.py --signals signals.jsonl --gt gt_events.json --strict

输出: report.json (含逐事件明细 + 汇总指标)，同时打印摘要到 stderr。
"""
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta

# 预警信号才计入命中（confirmation 是已发生事实，不算预警）
FOREWARD_TEMPORALITIES = {"forward_looking", "latent"}


# ----- 字符串匹配工具 -----
def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _company_names(sig) -> list[str]:
    """schema 中 companies 是 [{name, role}, ...]；GT 是 [str, ...]。两边归一化。"""
    out = []
    for c in sig.get("companies", []) or []:
        if isinstance(c, dict):
            out.append(_norm(c.get("name", "")))
        else:
            out.append(_norm(str(c)))
    return [x for x in out if x]


def _match_any(sig_vals: list[str], gt_vals: list[str]) -> bool:
    """大小写不敏感的子串双向匹配（避免 'Red Sea' vs 'red sea' / 'Suez' vs 'Suez Canal'）。"""
    sv = [_norm(v) for v in (sig_vals or []) if v]
    gv = [_norm(v) for v in (gt_vals or []) if v]
    for a in sv:
        for b in gv:
            if a == b or a in b or b in a:
                return True
    return False


def signal_matches_event(sig: dict, event: dict, strict: bool) -> bool:
    """按协议 §1.4 的匹配规则。strict=True 为论文主体，False 为稳健性。"""
    # 公司匹配（最准）：信号任一公司名命中 GT 的任一公司名/别名
    if _match_any(_company_names(sig), event.get("companies", [])):
        return True
    geo_ok = _match_any(sig.get("geographies", []), event.get("geographies", []))
    com_ok = _match_any(sig.get("commodities", []), event.get("commodities", []))
    if strict:
        # 严格: 商品 + 地区 同时匹配
        return geo_ok and com_ok
    # 宽松: 地区 OR 商品
    return geo_ok or com_ok


# ----- 日期工具 -----
def _parse_date(s: str) -> datetime:
    return datetime.strptime(s[:10], "%Y-%m-%d")


# ----- 主流程 -----
def load_signals(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sig = json.loads(line)
            except json.JSONDecodeError:
                continue
            if sig.get("status") == "error":
                continue  # 跳过抽取失败的行
            out.append(sig)
    return out


def validate(signals: list[dict], gt_events: list[dict], window_days: int = 180,
             strict: bool = True) -> dict:
    """
    核心验证逻辑。返回结构:
      per_event: [{event_id, hit(bool), earliest_signal_date, lead_time_days, n_signals_in_window,
                    matched_signals:[signal_id,...]}]
      summary: {hit_rate, mean_lead_time, median_lead_time, precision, recall_note}
    命中定义 (协议 §1.3):
      1. signal_date < gt_onset_date （严格早于）
      2. temporality in {forward_looking, latent}
      3. signal_matches_event (按 strict 规则)
    窗口: [onset - window_days, onset)
    """
    # 按事件建立命中记录
    per_event = []
    matched_signal_ids = set()  # 至少命中一个事件的信号 ID（用于 precision）

    for ev in gt_events:
        try:
            onset = _parse_date(ev["gt_onset_date"])
        except (KeyError, ValueError):
            # GT onset 日期缺失或格式错的，跳过并记录
            per_event.append({
                "event_id": ev.get("event_id", "?"),
                "hit": False,
                "skipped": "invalid_or_missing gt_onset_date",
            })
            continue
        window_start = onset - timedelta(days=window_days)
        # 候选信号: 在窗口内 + temporality 为预警类
        candidates = []
        for s in signals:
            try:
                sd = _parse_date(s["signal_date"])
            except (KeyError, ValueError):
                continue
            if not (window_start <= sd < onset):
                continue
            if s.get("temporality") not in FOREWARD_TEMPORALITIES:
                continue
            if signal_matches_event(s, ev, strict=strict):
                candidates.append((sd, s))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            earliest_date, _ = candidates[0]
            lead_time = (onset - earliest_date).days
            matched_ids = [s.get("signal_id", s.get("input_id", "")) for _, s in candidates]
            matched_signal_ids.update(matched_ids)
            per_event.append({
                "event_id": ev["event_id"],
                "event_name": ev.get("event_name", ""),
                "hit": True,
                "gt_onset_date": ev["gt_onset_date"],
                "earliest_signal_date": earliest_date.strftime("%Y-%m-%d"),
                "lead_time_days": lead_time,
                "n_signals_in_window": len(candidates),
                "matched_signal_ids": matched_ids,
            })
        else:
            per_event.append({
                "event_id": ev["event_id"],
                "event_name": ev.get("event_name", ""),
                "hit": False,
                "gt_onset_date": ev["gt_onset_date"],
                "lead_time_days": None,
                "n_signals_in_window": 0,
            })

    # ----- 汇总指标 -----
    n_gt = len([e for e in per_event if "skipped" not in e])
    n_hit = sum(1 for e in per_event if e.get("hit"))
    hit_rate = n_hit / n_gt if n_gt else 0.0

    leads = [e["lead_time_days"] for e in per_event if e.get("hit")]
    mean_lead = sum(leads) / len(leads) if leads else None
    sorted_leads = sorted(leads)
    median_lead = sorted_leads[len(sorted_leads) // 2] if sorted_leads else None

    # precision: 命中 GT 的信号数 / 所有预警信号数（窗口内 forward+latent）
    # 这里 "所有预警信号数" 按 GT 窗口内的 forward+latent 信号数计；
    # 窗口外的信号（既未命中也未落入任一窗口）算 "未配对噪声" 单列。
    n_warning_total = 0
    for s in signals:
        if s.get("temporality") not in FOREWARD_TEMPORALITIES:
            continue
        try:
            sd = _parse_date(s["signal_date"])
        except (KeyError, ValueError):
            continue
        # 是否在任一 GT 窗口内
        in_any_window = False
        for ev in gt_events:
            try:
                onset = _parse_date(ev["gt_onset_date"])
            except (KeyError, ValueError):
                continue
            if onset - timedelta(days=window_days) <= sd < onset:
                in_any_window = True
                break
        if in_any_window:
            n_warning_total += 1
    precision = (len(matched_signal_ids) / n_warning_total) if n_warning_total else None

    summary = {
        "n_gt_events": n_gt,
        "n_hit": n_hit,
        "hit_rate": round(hit_rate, 3),
        "mean_lead_time_days": mean_lead,
        "median_lead_time_days": median_lead,
        "precision": round(precision, 3) if precision is not None else None,
        "n_warning_signals_in_windows": n_warning_total,
        "n_matched_signals": len(matched_signal_ids),
        "strict_matching": strict,
        "window_days": window_days,
    }
    return {"per_event": per_event, "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper A 前视验证")
    ap.add_argument("--signals", required=True, help="extract_events.py 输出的 signals.jsonl")
    ap.add_argument("--gt", required=True, help="gt_events.json 路径")
    ap.add_argument("--report", default=None, help="输出 JSON 报告路径（默认 stdout）")
    ap.add_argument("--window", type=int, default=180, help="预警窗口天数 (默认 180)")
    ap.add_argument("--strict", action="store_true", default=True, help="严格匹配 (默认开)")
    ap.add_argument("--loose", dest="strict", action="store_false", help="宽松匹配 (稳健性检验)")
    args = ap.parse_args()

    signals = load_signals(args.signals)
    with open(args.gt, encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_events = gt_data.get("events", gt_data) if isinstance(gt_data, dict) else gt_data

    result = validate(signals, gt_events, window_days=args.window, strict=args.strict)

    out_text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(out_text)
    else:
        print(out_text)

    s = result["summary"]
    print(f"\n=== VALIDATION SUMMARY ({'strict' if args.strict else 'loose'}, window={args.window}d) ===",
          file=sys.stderr)
    print(f"  GT events: {s['n_gt_events']} | hits: {s['n_hit']} | hit_rate: {s['hit_rate']}",
          file=sys.stderr)
    print(f"  lead time: mean={s['mean_lead_time_days']}d, median={s['median_lead_time_days']}d",
          file=sys.stderr)
    print(f"  precision: {s['precision']}  (matched {s['n_matched_signals']}/{s['n_warning_signals_in_windows']})",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
