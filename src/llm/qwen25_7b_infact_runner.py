from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

# Optional few-shot examples. Replace with better in-domain examples if desired.
FEW_SHOT_EXAMPLES = [
    {
        "input": """Claim:
Guvernul a redus TVA la toate alimentele de baza.

Context:
Afirmatia a fost facuta intr-o dezbatere televizata despre masuri fiscale.

Task:
1. Choose exactly one verdict from this set:
True, False, Mixed, Mostly True, Mostly False, Unverifiable
2. Write a brief explanation in Romanian in 2-3 sentences.

Return valid JSON only in this format:
{"verdict": "...", "explanation": "..."}
""",
        "output": '{"verdict": "Mostly False", "explanation": "Afirmatia este prea generala. Unele produse au beneficiat de modificari fiscale, dar nu toate alimentele de baza au fost incluse, astfel incat formularea extinde masura dincolo de ceea ce poate fi sustinut."}',
    },
    {
        "input": """Claim:
Romania este stat membru al Uniunii Europene.

Context:
    Afirmatia apare intr-o discutie despre fonduri europene.

Task:
1. Choose exactly one verdict from this set:
True, False, Mixed, Mostly True, Mostly False, Unverifiable
2. Write a brief explanation in Romanian in 2-3 sentences.

Return valid JSON only in this format:
{"verdict": "...", "explanation": "..."}
""",
        "output": '{"verdict": "True", "explanation": "Afirmatia este corecta. Romania este stat membru al Uniunii Europene din 2007, iar acest statut este stabil si verificabil din surse institutionale publice."}',
    },
]


@dataclass
class RunConfig:
    input_path: str
    output_jsonl: str
    output_tsv: str
    run_config_json: str
    model_name: str
    limit: int
    offset: int
    include_scope: bool
    few_shot: bool
    prompt_version: str
    temperature: float
    max_new_tokens: int
    top_p: float
    seed: Optional[int]
    load_in_4bit: bool
    use_bf16: bool


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    ensure_dir(Path(path).parent)
    if path.endswith(".tsv"):
        df.to_csv(path, sep="\t", index=False)
    else:
        df.to_csv(path, index=False)


def load_dataset_subset(
    path: str,
    limit: int,
    offset: int,
    include_scope: bool,
) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype={"record_id": "int64"})
    required = {"record_id", "claim_text", "context"}
    if include_scope:
        required.add("verification_scope")
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if limit < 1 or offset < 0:
        raise ValueError("limit must be >= 1 and offset must be >= 0")
    subset = df.iloc[offset : offset + limit].copy()
    logger.info("Loaded %d rows from %s (offset=%d, limit=%d)", len(subset), path, offset, limit)
    return subset


def build_messages(row: pd.Series, include_scope: bool, few_shot: bool) -> tuple[list[dict[str, str]], str]:
    if include_scope:
        user_prompt = USER_TEMPLATE_V2.format(
            claim_text=str(row.get("claim_text", "")),
            context=str(row.get("context", "")),
            verification_scope=str(row.get("verification_scope", "")),
        )
        prompt_version = "v2_claim_context_scope_fewshot" if few_shot else "v2_claim_context_scope"
    else:
        user_prompt = USER_TEMPLATE_V1.format(
            claim_text=str(row.get("claim_text", "")),
            context=str(row.get("context", "")),
        )
        prompt_version = "v1_claim_context_fewshot" if few_shot else "v1_claim_context"

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot:
        for ex in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": ex["input"]})
            messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": user_prompt})
    return messages, prompt_version


def pick_dtype(use_bf16: bool) -> torch.dtype:
    if use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model_and_tokenizer(model_name: str, load_in_4bit: bool, use_bf16: bool):
    dtype = pick_dtype(use_bf16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["quantization_config"] = quant_config

    logger.info("Loading model %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return model, tokenizer


def call_model(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    seed: Optional[int],
) -> str:
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "temperature": max(temperature, 1e-5),
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)
    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _find_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _normalise_whitespace(text: str) -> str:
    return " ".join(str(text).split())


def parse_response(text: str) -> dict[str, Any]:
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
    ids: set[int] = set()
    file_path = Path(path)
    if not file_path.exists():
        return ids
    with open(file_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ids.add(int(obj["record_id"]))
            except Exception:
                continue
    logger.info("Loaded %d existing record_ids from %s", len(ids), path)
    return ids


def save_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not Path(path).exists():
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def save_tsv(path: str, jsonl_path: str) -> None:
    df = pd.DataFrame(read_jsonl(jsonl_path))
    save_dataframe(df, path)
    logger.info("Saved TSV outputs to %s", path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-7B-Instruct locally on GPU.")
    parser.add_argument("--input_path", default="data/infact_canonical.tsv")
    parser.add_argument("--output_jsonl", default="results/llm_outputs/qwen25_7b_outputs.jsonl")
    parser.add_argument("--output_tsv", default="results/llm_outputs/qwen25_7b_outputs.tsv")
    parser.add_argument("--run_config_json", default="results/llm_outputs/qwen25_7b_run_config.json")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--limit", type=int, default=174)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--include_scope", action="store_true")
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    prompt_version = "v2_claim_context_scope_fewshot" if args.include_scope and args.few_shot else (
        "v2_claim_context_scope" if args.include_scope else (
            "v1_claim_context_fewshot" if args.few_shot else "v1_claim_context"
        )
    )

    run_cfg = RunConfig(
        input_path=args.input_path,
        output_jsonl=args.output_jsonl,
        output_tsv=args.output_tsv,
        run_config_json=args.run_config_json,
        model_name=args.model_name,
        limit=args.limit,
        offset=args.offset,
        include_scope=args.include_scope,
        few_shot=args.few_shot,
        prompt_version=prompt_version,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        seed=args.seed,
        load_in_4bit=args.load_in_4bit,
        use_bf16=args.use_bf16,
    )

    df = load_dataset_subset(args.input_path, args.limit, args.offset, args.include_scope)
    processed_ids = load_existing_record_ids(args.output_jsonl)
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.load_in_4bit, args.use_bf16)

    processed = 0
    skipped = 0
    errors = 0
    started_at = datetime.now(timezone.utc).isoformat()

    for _, row in df.iterrows():
        record_id = int(row["record_id"])
        if record_id in processed_ids:
            skipped += 1
            continue

        messages, prompt_version = build_messages(row, args.include_scope, args.few_shot)
        try:
            raw_text = call_model(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                top_p=args.top_p,
                seed=args.seed,
            )
            parsed = parse_response(raw_text)
        except Exception as exc:
            logger.exception("Generation failed for record_id=%s", record_id)
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
            "model": args.model_name,
            "prompt_version": prompt_version,
            "raw_text": parsed["raw_text"],
            "verdict": parsed["verdict"],
            "explanation": parsed["explanation"],
            "parse_ok": bool(parsed["parse_ok"]),
            "error": parsed["error"],
        }
        save_jsonl(args.output_jsonl, [row_out])
        processed += 1
        if processed % 10 == 0:
            logger.info("Processed %d records (skipped=%d, errors=%d)", processed, skipped, errors)

    save_tsv(args.output_tsv, args.output_jsonl)
    completed_at = datetime.now(timezone.utc).isoformat()
    save_json(
        {
            **run_cfg.__dict__,
            "started_at": started_at,
            "completed_at": completed_at,
            "processed": processed,
            "skipped": skipped,
            "errors": errors,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        args.run_config_json,
    )

    logger.info("Run complete. processed=%d skipped=%d errors=%d", processed, skipped, errors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
