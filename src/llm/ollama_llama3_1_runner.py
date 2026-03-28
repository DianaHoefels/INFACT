"""Ollama runner for Llama 3.1 instruction models.

Runs local Ollama chat inference for the INFACT benchmark using a
resumable JSONL output format.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

try:
    import requests
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "Missing dependency: requests. Install with 'pip install requests'."
    ) from exc

from src.utils.io import ensure_dir, save_json

logger = logging.getLogger(__name__)

ALLOWED_VERDICTS = [
    "True",
    "False",
    "Mixed",
    "Mostly True",
    "Mostly False",
    "Unverifiable",
]

SYSTEM_PROMPT = (
    "You are a careful fact-checking assistant. Your task is to assign a verdict "
    "to a public claim and explain your reasoning briefly and cautiously. "
    "Follow the requested label set exactly and return valid JSON only."
)

USER_TEMPLATE_V1 = """Claim:
{claim_text}

Context:
{context}

Task:
1. Choose exactly one verdict from this set:
True, False, Mixed, Mostly True, Mostly False, Unverifiable
2. Write a brief explanation in Romanian in 2-3 sentences.

Return valid JSON only in this format:
{{"verdict": "...", "explanation": "..."}}
"""

USER_TEMPLATE_V2 = """Claim:
{claim_text}

Context:
{context}

Verification scope:
{verification_scope}

Task:
1. Choose exactly one verdict from this set:
True, False, Mixed, Mostly True, Mostly False, Unverifiable
2. Write a brief explanation in Romanian in 2-3 sentences.

Return valid JSON only in this format:
{{"verdict": "...", "explanation": "..."}}
"""


@dataclass
class RunConfig:
    input_path: str
    output_jsonl: str
    output_tsv: str
    base_url: str
    model_name: str
    limit: int
    offset: int
    include_scope: bool
    prompt_version: str
    temperature: float
    max_tokens: int
    top_p: float
    seed: Optional[int]
    timeout: float
    max_retries: int


def load_dataset_subset(path: str, limit: int, offset: int, include_scope: bool) -> pd.DataFrame:
    """Load the dataset TSV and return the requested subset."""
    df = pd.read_csv(path, sep="\t", dtype={"record_id": "int64"})
    required_cols = {
        "record_id",
        "claim_text",
        "context",
    }
    if include_scope:
        required_cols.add("verification_scope")
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
    if offset < 0:
        raise ValueError("offset must be >= 0")
    df = df.sort_values("record_id", kind="mergesort").reset_index(drop=True)
    if limit <= 0:
        subset = df.iloc[offset:].copy()
    else:
        subset = df.iloc[offset : offset + limit].copy()
    logger.info("Loaded %d rows from %s (offset=%d, limit=%d)", len(subset), path, offset, limit)
    return subset


def build_messages(row: pd.Series, include_scope: bool) -> tuple[list[dict[str, str]], str]:
    """Build Ollama chat messages for a single row."""
    if include_scope:
        user_prompt = USER_TEMPLATE_V2.format(
            claim_text=str(row.get("claim_text", "")),
            context=str(row.get("context", "")),
            verification_scope=str(row.get("verification_scope", "")),
        )
        prompt_version = "v2_claim_context_scope"
    else:
        user_prompt = USER_TEMPLATE_V1.format(
            claim_text=str(row.get("claim_text", "")),
            context=str(row.get("context", "")),
        )
        prompt_version = "v1_claim_context"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return messages, prompt_version


def _is_transient_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    transient_markers = [
        "timeout",
        "timed out",
        "connection",
        "rate limit",
        "too many requests",
        "temporarily",
        "server",
        "unavailable",
    ]
    return any(marker in msg for marker in transient_markers)


def _is_retryable_status(status_code: int) -> bool:
    return status_code in {408, 429, 500, 502, 503, 504}


def _ollama_chat_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/api/chat"


class _TransientApiError(RuntimeError):
    pass


def call_model_ollama(
    base_url: str,
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    seed: Optional[int],
    timeout: float,
    max_retries: int,
) -> str:
    """Call Ollama chat endpoint with retries."""
    last_exc: Optional[Exception] = None
    url = _ollama_chat_url(base_url)
    options: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "num_predict": max_tokens,
    }
    if seed is not None:
        options["seed"] = seed

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": options,
    }

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            if response.status_code >= 400:
                detail = response.text.strip() or f"HTTP {response.status_code}"
                if _is_retryable_status(response.status_code):
                    raise _TransientApiError(detail)
                raise RuntimeError(f"Ollama API error (status {response.status_code}): {detail}")

            data = response.json()
            message = data.get("message") if isinstance(data, dict) else None
            if isinstance(message, dict) and "content" in message:
                return str(message.get("content") or "")
            if isinstance(data, dict) and "response" in data:
                return str(data.get("response") or "")
            raise RuntimeError("Ollama API response missing message content.")
        except _TransientApiError as exc:  # pragma: no cover - runtime dependency
            last_exc = exc
            if attempt >= max_retries:
                break
            backoff = min(60.0, (2 ** attempt) + random.random())
            logger.warning("Transient error: %s. Retrying in %.2fs", exc, backoff)
            time.sleep(backoff)
        except Exception as exc:  # pragma: no cover - runtime dependency
            last_exc = exc
            if attempt >= max_retries or not _is_transient_error(exc):
                break
            backoff = min(60.0, (2 ** attempt) + random.random())
            logger.warning("Transient error: %s. Retrying in %.2fs", exc, backoff)
            time.sleep(backoff)

    raise RuntimeError(f"Model call failed after {max_retries} retries: {last_exc}")


def _find_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _normalise_whitespace(text: str) -> str:
    return " ".join(text.split())


def parse_response(text: str) -> dict[str, Any]:
    """Parse model response into verdict/explanation with fallbacks."""
    result: dict[str, Any] = {
        "raw_text": text or "",
        "verdict": None,
        "explanation": None,
        "parse_ok": False,
        "error": None,
    }
    if not text:
        result["error"] = "empty_response"
        return result

    cleaned = text.strip()
    parsed_obj: Optional[dict[str, Any]] = None

    try:
        parsed_obj = json.loads(cleaned)
    except json.JSONDecodeError:
        extracted = _find_first_json_object(cleaned)
        if extracted:
            try:
                parsed_obj = json.loads(extracted)
            except json.JSONDecodeError:
                parsed_obj = None

    if parsed_obj and isinstance(parsed_obj, dict):
        verdict = parsed_obj.get("verdict")
        explanation = parsed_obj.get("explanation")
        if isinstance(verdict, str):
            verdict = _normalise_whitespace(verdict)
        if isinstance(explanation, str):
            explanation = _normalise_whitespace(explanation)
        result["verdict"] = verdict
        result["explanation"] = explanation
        if verdict in ALLOWED_VERDICTS and isinstance(explanation, str) and explanation:
            result["parse_ok"] = True
            return result
        result["error"] = "invalid_fields"
    else:
        result["error"] = "json_parse_failed"

    lowered = cleaned.lower()
    for label in sorted(ALLOWED_VERDICTS, key=len, reverse=True):
        if label.lower() in lowered:
            result["verdict"] = label
            result["parse_ok"] = False
            return result

    return result


def load_existing_record_ids(path: str) -> set[int]:
    """Load record_ids from an existing JSONL file for resumability."""
    ids: set[int] = set()
    if not Path(path).exists():
        return ids
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "record_id" in obj:
                    ids.add(int(obj["record_id"]))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    logger.info("Loaded %d processed record_ids from %s", len(ids), path)
    return ids


def save_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    """Append rows to a JSONL file."""
    ensure_dir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    if not Path(path).exists():
        return data
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def save_tsv(path: str, jsonl_path: str) -> None:
    """Convert JSONL outputs to TSV."""
    records = read_jsonl(jsonl_path)
    df = pd.DataFrame(records)
    ensure_dir(Path(path).parent)
    df.to_csv(path, sep="\t", index=False)
    logger.info("Saved TSV outputs to %s", path)


def run_ollama_inference(
    input_path: str,
    output_jsonl: str,
    output_tsv: str,
    base_url: str,
    model_name: str,
    limit: int,
    offset: int,
    include_scope: bool,
    temperature: float,
    max_tokens: int,
    top_p: float,
    seed: Optional[int],
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    df = load_dataset_subset(input_path, limit, offset, include_scope)
    processed_ids = load_existing_record_ids(output_jsonl)

    total = 0
    skipped = 0
    errors = 0
    started_at = datetime.now(timezone.utc).isoformat()

    for _, row in df.iterrows():
        record_id = int(row["record_id"])
        if record_id in processed_ids:
            skipped += 1
            continue

        messages, prompt_version = build_messages(row, include_scope)
        try:
            raw_text = call_model_ollama(
                base_url=base_url,
                model_name=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                seed=seed,
                timeout=timeout,
                max_retries=max_retries,
            )
            parsed = parse_response(raw_text)
        except Exception as exc:  # pragma: no cover - runtime dependency
            logger.exception("Model call failed for record_id=%s", record_id)
            parsed = {
                "raw_text": "",
                "verdict": None,
                "explanation": None,
                "parse_ok": False,
                "error": str(exc),
            }
            errors += 1

        row_out = {
            "record_id": record_id,
            "model": model_name,
            "prompt_version": prompt_version,
            "raw_text": parsed["raw_text"],
            "verdict": parsed["verdict"],
            "explanation": parsed["explanation"],
            "parse_ok": bool(parsed["parse_ok"]),
            "error": parsed["error"],
        }

        save_jsonl(output_jsonl, [row_out])
        total += 1

        if total % 10 == 0:
            logger.info("Processed %d records (skipped=%d, errors=%d)", total, skipped, errors)

    save_tsv(output_tsv, output_jsonl)

    completed_at = datetime.now(timezone.utc).isoformat()
    run_report = {
        "input_path": input_path,
        "output_jsonl": output_jsonl,
        "output_tsv": output_tsv,
        "base_url": base_url,
        "model_name": model_name,
        "limit": limit,
        "offset": offset,
        "include_scope": include_scope,
        "prompt_version": "v2_claim_context_scope" if include_scope else "v1_claim_context",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "seed": seed,
        "timeout": timeout,
        "max_retries": max_retries,
        "started_at": started_at,
        "completed_at": completed_at,
        "processed": total,
        "skipped": skipped,
        "errors": errors,
    }
    save_json(run_report, "results/llm_outputs/ollama_llama3_1_run_config.json")

    logger.info("Run complete. processed=%d skipped=%d errors=%d", total, skipped, errors)
    return run_report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Llama 3.1 via local Ollama.")
    parser.add_argument(
        "--input_path",
        default="data/infact_dataset_processed.tsv",
        help="Path to TSV dataset (default: data/infact_dataset_processed.tsv)",
    )
    parser.add_argument(
        "--output_jsonl",
        default="results/llm_outputs/ollama_llama3_1_outputs.jsonl",
    )
    parser.add_argument(
        "--output_tsv",
        default="results/llm_outputs/ollama_llama3_1_outputs.tsv",
    )
    parser.add_argument("--limit", type=int, default=174, help="Number of rows to process")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--base_url", default="http://localhost:11434")
    parser.add_argument("--model_name", default="llama3.1:70b-instruct")
    parser.add_argument("--include_scope", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=220)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=3)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    run_ollama_inference(
        input_path=args.input_path,
        output_jsonl=args.output_jsonl,
        output_tsv=args.output_tsv,
        base_url=args.base_url,
        model_name=args.model_name,
        limit=args.limit,
        offset=args.offset,
        include_scope=args.include_scope,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        seed=args.seed,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
