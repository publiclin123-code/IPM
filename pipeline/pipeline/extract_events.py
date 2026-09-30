"""
LLM 事件抽取管道 (Paper A RQ1)
从新闻文本(JSONL)中抽取供应链风险信号，输出为 JSONL。

零第三方依赖（标准库 urllib），默认调用本地 llama.cpp llama-server（OpenAI 兼容端点）。

用法:
  python extract_events.py --input news.jsonl --output signals.jsonl
  python extract_events.py --input news.jsonl --dry-run     # 只打印拼装好的 prompt，不调用 API
  python extract_events.py --input news.jsonl --model gemma4-31b

环境变量（或 --model / --base-url / --api-key）:
  LLM_MODEL       模型别名，默认 qwen3.6-27b
  LLM_BASE_URL    llama-server 地址（如 http://localhost:8081）；留空按模型名自动定位
  LLM_API_KEY     可选，远程/带鉴权端点时用；本地 llama-server 留空

模型别名 → 端口注册表（MODEL_REGISTRY）:
  qwen3.6-27b      -> http://localhost:8081
  gemma4-31b       -> http://localhost:8082
  muse-glimmer-30b -> http://localhost:8083

注意: 思考模型需 server 以 --jinja 启动；默认开启 thinking，输出经 parse_response 剥离 <think>。

输入 JSONL 行: {"id","date","source","url","title","text"}
输出 JSONL 行: 每条信号一行（见 shared/schema/event_schema.json），错误行含 status=error。
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "event_extraction_prompt.md"

# 模型注册表：model 别名 → llama-server 端口。一台 llama-server 跑一个模型
# （或同端口 --alias 多模型）。未显式给 --base-url 时按此表定位 server。
MODEL_REGISTRY = {
    "qwen3.6-27b": "8081",
    "gemma4-31b": "8082",
    "muse-glimmer-30b": "8083",
}
DEFAULT_PORT = "8080"


def resolve_base_url(model: str, base_url: str) -> str:
    """显式 --base-url 优先；否则按模型名查注册表，找不到用默认端口。"""
    if base_url:
        return base_url
    return f"http://localhost:{MODEL_REGISTRY.get(model, DEFAULT_PORT)}"


def load_system_prompt(prompt_path: Path | None = None) -> str:
    """从 prompt 模板文件提取 System 段（## System Prompt 与 ## User Prompt 之间）。"""
    path = prompt_path if prompt_path is not None else PROMPT_PATH
    text = path.read_text(encoding="utf-8")
    start = text.index("## System Prompt") + len("## System Prompt")
    end = text.index("## User Prompt")
    return text[start:end].strip()


def build_user_prompt(article: dict) -> str:
    return (
        "请从以下新闻中抽取供应链风险信号。\n\n"
        f"新闻标题: {article.get('title', '')}\n"
        f"发布日期: {article.get('date', '')}\n"
        f"来源: {article.get('source', '')}\n"
        f"原文:\n\n{article.get('text', '')}"
    )


def call_llm(messages: list, api_key: str = "", base_url: str = "http://localhost:8080",
             model: str = "qwen3.6-27b", timeout: int = 600, thinking: bool = True) -> str:
    """调用 llama.cpp llama-server 的 OpenAI 兼容端点 /v1/chat/completions。

    llama-server（llama.cpp 的 HTTP 服务）默认监听 8080，暴露 OpenAI 兼容 API。
    对思考模型（Qwen3 / Gemma 系列等），server 需以 --jinja 启动以启用 chat
    模板中的思考块；此处用 "thinking" 字段逐请求开关。推理内容返回在
    message.reasoning_content，最终答案在 message.content；parse_response()
    仍会额外剥离任何内联 <think> 段以兼容。
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0.0,
        "max_tokens": 8192,
    }
    if thinking:
        payload["thinking"] = {"type": "enabled"}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    if not content.strip() and msg.get("reasoning_content"):
        content = msg["reasoning_content"]
    return content


def parse_response(raw: str) -> dict:
    """容错解析 LLM 输出：支持 ```json 围栏、裸 JSON、以及 qwen3 思考模式
    （<think>...</think> 前缀 + 末尾 JSON）的混合输出。"""
    raw = raw.strip()
    # 1) 剔除 qwen3 思考段: <think>...</think> 及其之前的所有内容
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1].strip()
    # 2) ```json 围栏
    if raw.startswith("```"):
        body = raw.strip("`")
        if body.startswith("json"):
            body = body[4:]
        raw = body.strip()
    # 3) 裸 JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            if "signals" in obj:
                return obj
            if not obj:  # 模型返回 {} = 无信号，等价 empty
                return {"signals": []}
    except json.JSONDecodeError:
        pass
    # 4) 最后手段: 截取第一个 { 到最后一个 } 的子串（可能含思考尾部垃圾）
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(raw[start:end + 1])
            if isinstance(obj, dict) and "signals" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    raise ValueError(f"LLM 输出无法解析为信号 JSON: {raw[:300]}")


REQUIRED_FIELDS = [
    "temporality", "event_type", "severity", "confidence",
    "description", "trigger_phrases", "commodities", "companies", "geographies",
]
V2_EXTRA_FIELDS = ["impact_uncertainty"]


def validate_signal(sig: dict, extra: list[str] | None = None) -> list:
    """返回缺失的必填字段列表；空列表表示通过。"""
    need = REQUIRED_FIELDS + (extra or [])
    return [f for f in need if f not in sig]


def load_done_ids(path: Path) -> set[str]:
    """Resume 时读取已完成的 input_id，并**原地清除 status=error 的旧行**。

    这样 error 行会被删除，重跑时能重新抽取并写入新行，避免出现
    「同一 input_id 同时有旧 error 行 + 新成功行」导致的重复计数，
    也让编排器的 error-rate 检查不会被历史 error 行卡死。
    """
    done: set[str] = set()
    if not path.exists():
        return done
    kept: list[str] = []
    had_error = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "error":
                had_error = True
                continue
            kept.append(line)
            iid = rec.get("input_id")
            if iid:
                done.add(iid)
    if had_error:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(kept) + ("\n" if kept else ""))
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM 供应链风险信号抽取")
    ap.add_argument("--input", required=True, help="输入新闻 JSONL 路径")
    ap.add_argument("--output", default=None, help="输出信号 JSONL 路径（默认 stdout 或 --dry-run）")
    ap.add_argument("--dry-run", action="store_true", help="只打印第一条拼装好的 prompt，不调用 API")
    ap.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""))
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", ""),
                    help="llama-server 地址（如 http://localhost:8081）；留空按模型名自动定位")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen3.6-27b"))
    ap.add_argument("--no-thinking", action="store_true", help="关闭思考模式（非思考模型用）")
    ap.add_argument("--prompt", default=None, help="自定义 prompt 模板路径（默认 event_extraction_prompt.md）")
    ap.add_argument("--resume", action="store_true",
                    help="追加写入并跳过 output 里已有的 input_id")
    ap.add_argument("--limit", type=int, default=0, help="最多新处理 N 篇（0=全部）")
    args = ap.parse_args()

    base_url = resolve_base_url(args.model, args.base_url)
    thinking = not args.no_thinking
    prompt_path = Path(args.prompt) if args.prompt else None
    system_prompt = load_system_prompt(prompt_path)
    extra_fields = V2_EXTRA_FIELDS if (args.prompt and "v2" in Path(args.prompt).name.lower()) else []

    done_ids: set[str] = set()
    if args.output and args.resume:
        done_ids = load_done_ids(Path(args.output))
        out = open(args.output, "a", encoding="utf-8")
    elif args.output:
        out = open(args.output, "w", encoding="utf-8")
    else:
        out = sys.stdout

    n_articles = n_signals = n_errors = n_skipped = n_empty = 0
    n_new = 0
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            article = json.loads(line)
            iid = article.get("id", "")
            if iid in done_ids:
                n_skipped += 1
                continue
            if args.limit and n_new >= args.limit:
                break
            n_articles += 1
            n_new += 1
            user_prompt = build_user_prompt(article)

            if args.dry_run:
                print(f"=== input_id={iid} ===")
                print("----- SYSTEM -----")
                print(system_prompt[:400], "...")
                print("----- USER -----")
                print(user_prompt[:400], "...")
                print("----- END -----\n")
                continue

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            result = None
            last_err = ""
            for attempt in (1, 2):  # 失败重试一次
                try:
                    raw = call_llm(messages, args.api_key, base_url, args.model, thinking=thinking)
                    result = parse_response(raw)
                    break
                except Exception as e:
                    last_err = f"attempt {attempt}: {e}"
                    if attempt == 1:
                        continue
            if result is None:
                out.write(json.dumps({
                    "input_id": iid, "status": "error", "error": last_err,
                }, ensure_ascii=False) + "\n")
                out.flush()
                n_errors += 1
                print(f"  error {iid}: {last_err[:120]}", file=sys.stderr, flush=True)
                continue

            sigs = result.get("signals") or []
            if not sigs:
                out.write(json.dumps({
                    "input_id": iid, "status": "empty",
                }, ensure_ascii=False) + "\n")
                out.flush()
                n_empty += 1
                print(f"  empty {iid}", file=sys.stderr, flush=True)
                continue

            wrote = 0
            for i, sig in enumerate(sigs, start=1):
                missing = validate_signal(sig, extra_fields)
                if missing:
                    out.write(json.dumps({
                        "input_id": iid, "status": "error",
                        "error": f"missing fields: {missing}", "partial": sig,
                    }, ensure_ascii=False) + "\n")
                    n_errors += 1
                    continue
                sig["signal_id"] = f"{iid}-{i}"
                sig["input_id"] = iid
                sig["signal_date"] = article.get("date", "")
                out.write(json.dumps(sig, ensure_ascii=False) + "\n")
                n_signals += 1
                wrote += 1
            if wrote == 0 and sigs:
                # all signals failed validation: mark article done so resume skips it
                pass
            out.flush()
            print(f"  ok {iid}: {wrote} signals", file=sys.stderr, flush=True)

    if args.output:
        out.close()
    print(
        f"done: new={n_articles} skipped={n_skipped} -> "
        f"{n_signals} signals, {n_empty} empty, {n_errors} errors",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
