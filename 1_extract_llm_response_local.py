"""
Unified LLM Response Extraction Script (Local Model Version)
- Llama 3.3 70B
- Qwen3 32B
- DeepSeek R1 Distill Llama 70B
The model and alternative type (compromise / non_compromise) are passed as runtime arguments.

Usage examples:
    python 1_extract_llm_response_local.py --model qwen3-32b --option_type compromise
    python 1_extract_llm_response_local.py --model llama4-scout --option_type non_compromise
    python 1_extract_llm_response_local.py --model deepseek-r1-70b --option_type compromise --persona_type advisor --start_idx 0 --end_idx 100

  vLLM mode (B200, llama4-scout / qwen3.5-122b-a10b):
    1) Start the vLLM server externally (run_b200_staged.sh handles this automatically)
    2) Set the VLLM_BASE_URL environment variable (default: http://localhost:8000/v1)
    3) Invoke with the same commands as above — local_llm_utils dispatches automatically
"""

# pyright: basic, reportArgumentType=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false

import os
import sys
import re
import argparse
import pandas as pd
import time
import json
import ast


WORD_COUNT_RULES = {
    "compromise": {
        "option_c": (20, 30),
        "tradeoff_rule": (12, 17),
        "justification": (20, 30),
    },
    "non_compromise": {
        "option_c": (20, 30),
        "justification": (20, 30),
    },
}


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", str(text).strip())) if str(text).strip() else 0


def _check_range_violations(option_type: str, option_c: str, rule_or_type: str, justification: str):
    rules = WORD_COUNT_RULES[option_type]
    violations = []

    c_wc = _word_count(option_c)
    c_min, c_max = rules["option_c"]
    if not (c_min <= c_wc <= c_max):
        violations.append(("option_c", c_wc, c_min, c_max))

    j_wc = _word_count(justification)
    j_min, j_max = rules["justification"]
    if not (j_min <= j_wc <= j_max):
        violations.append(("justification", j_wc, j_min, j_max))

    if option_type == "compromise":
        r_wc = _word_count(rule_or_type)
        r_min, r_max = rules["tradeoff_rule"]
        if not (r_min <= r_wc <= r_max):
            violations.append(("tradeoff_rule", r_wc, r_min, r_max))

    return violations

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):
        return iterable

# ============================================================================
# Model configuration mapping
# ============================================================================
MODEL_CONFIG = {
    "llama3.3-70b": {"provider": "local", "model_id": "meta-llama/Llama-3.3-70B-Instruct"},
    "qwen3-32b": {"provider": "local", "model_id": "Qwen/Qwen3-32B"},
    "deepseek-r1-70b": {"provider": "local", "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"},
    "mistral-small-3.1-24b": {"provider": "local", "model_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503"},
    "llama4-scout": {"provider": "local", "model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct"},
    "qwen3.5-122b-a10b": {"provider": "local", "model_id": "Qwen/Qwen3.5-122B-A10B-FP8"},
}

# Prompt paths per option type
PROMPT_PATHS = {
    "compromise": "third_alternative_prompts/last_compromise_prompt.txt",
    "non_compromise": "third_alternative_prompts/last_non_compromise_prompt.txt",
}


def extract_json_from_response(response_text):
    """
    Extract JSON from the response text. Three-stage attempt:
      1) JSON inside a markdown code block
      2) Validate that the range from the first { to the last } is valid JSON
      3) Use a regex to scan all {...} candidates and return the first valid JSON
    """
    text = str(response_text).strip()

    # Stage 1: Strip markdown code blocks
    if '```json' in text:
        text = text.split('```json', 1)[1].split('```', 1)[0].strip()
    elif '```' in text:
        parts = text.split('```')
        if len(parts) >= 3:
            text = parts[1].strip()

    # Stage 2: Extract the range from the first { to the last } and validate it
    s = text.find('{')
    e = text.rfind('}') + 1
    if s != -1 and e > s:
        candidate = text[s:e]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Stage 3: Scan {...} candidates with a regex
    for m in re.finditer(r'\{[^{}]*\}', text):
        try:
            json.loads(m.group())
            return m.group()
        except json.JSONDecodeError:
            continue

    # If every attempt fails, return the original (the caller handles parse failure)
    return text


def _parse_labeled_lines(text, labels):
    result = {}
    for label in labels:
        # regex: "Label:" or "**Label:**" followed by content until next label or end
        pattern = r'(?:\*\*)?(?:' + re.escape(label) + r')(?:\*\*)?:\s*(.+?)(?=(?:\n\s*(?:\*\*)?(?:' + '|'.join(re.escape(l) for l in labels) + r')(?:\*\*)?:)|\Z)'
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            result[label] = m.group(1).strip().strip('"').strip("'").strip()
    return result


def extract_compromise_fields(response_text):
    if pd.isna(response_text) or str(response_text).strip() == '':
        return "", "", ""

    text = str(response_text).strip()

    json_text = extract_json_from_response(text)
    try:
        response_dict = json.loads(json_text)
        return (response_dict.get("option_c", ""),
                response_dict.get("tradeoff_rule", ""),
                response_dict.get("compromise_justification", ""))
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        response_dict = ast.literal_eval(json_text)
        return (response_dict.get("option_c", ""),
                response_dict.get("tradeoff_rule", ""),
                response_dict.get("compromise_justification", ""))
    except:
        pass

    labels = ["Option C", "Trade-off Rule", "Compromise Justification"]
    parsed = _parse_labeled_lines(text, labels)
    if parsed:
        return (parsed.get("Option C", ""),
                parsed.get("Trade-off Rule", ""),
                parsed.get("Compromise Justification", ""))

    return "", "", ""


def extract_non_compromise_fields(response_text):
    """Extract option_c, reframe_type, and reframe_justification from a non_compromise response."""
    if pd.isna(response_text) or str(response_text).strip() == '':
        return "", "", ""

    text = str(response_text).strip()
    json_text = extract_json_from_response(text)

    labels = ["Option C", "Reframe Type", "Reframe Justification"]
    try:
        response_dict = json.loads(json_text)
        option_c = response_dict.get("option_c", "")
        reframe_type = response_dict.get("reframe_type", "")
        reframe_justification = response_dict.get("reframe_justification", "")
        if option_c:
            return option_c, reframe_type, reframe_justification
    except json.JSONDecodeError:
        try:
            response_dict = ast.literal_eval(json_text)
            option_c = response_dict.get("option_c", "")
            reframe_type = response_dict.get("reframe_type", "")
            reframe_justification = response_dict.get("reframe_justification", "")
            if option_c:
                return option_c, reframe_type, reframe_justification
        except:
            pass

    parsed = _parse_labeled_lines(text, labels)
    if parsed:
        return (
            parsed.get("Option C", ""),
            parsed.get("Reframe Type", ""),
            parsed.get("Reframe Justification", ""),
        )

    return "", "", ""


def is_fatal_error(error_msg):
    """Check whether this is a fatal error that should terminate immediately without retrying."""
    fatal_keywords = ["credit balance is too low", "invalid_api_key", "invalid api key"]
    error_lower = error_msg.lower()
    return any(kw in error_lower for kw in fatal_keywords)


# ============================================================================
# Main execution function
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Unified LLM response extraction script (Local HuggingFace Models)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python 1_extract_llm_response_local.py --model qwen3-32b --option_type compromise
  python 1_extract_llm_response_local.py --model llama4-scout --option_type non_compromise
  python 1_extract_llm_response_local.py --model deepseek-r1-70b --option_type compromise --start_idx 0 --end_idx 100
        """,
    )

    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_CONFIG.keys()),
        help="Local LLM model to use",
    )
    parser.add_argument(
        "--option_type", type=str, required=True,
        choices=["compromise", "non_compromise"],
        help="Type of alternative (compromise or non_compromise)",
    )
    parser.add_argument(
        "--persona_type", type=str, default="agent",
        choices=["agent", "advisor"],
        help="Persona type (default: agent)",
    )
    parser.add_argument(
        "--start_idx", type=int, default=0,
        help="Start index (default: 0)",
    )

    def parse_end_idx(value):
        """Parse end_idx: convert the 'None' string or -1 to None."""
        if value is None or value == 'None' or value == 'none':
            return None
        try:
            val = int(value)
            return None if val == -1 else val
        except ValueError:
            return None

    parser.add_argument(
        "--end_idx", type=parse_end_idx, default=-1,
        help="End index (default: -1; passing -1 or 'None' processes the full dataset)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (defaults to {option_type}_outputs/{model_name}/ when omitted)",
    )

    args = parser.parse_args()

    from local_llm_utils import (
        load_local_model,
        generate_text,
        cleanup_model,
        get_max_new_tokens,
        get_model_suffix,
    )

    # ── Extract config values ──────────────────────────────────────────────
    model_name = args.model
    option_type = args.option_type
    persona_type = args.persona_type
    start_idx = args.start_idx
    end_idx = args.end_idx
    if end_idx == -1:
        end_idx = None

    config = MODEL_CONFIG[model_name]
    provider = config["provider"]
    model_id = config["model_id"]
    model_suffix = get_model_suffix(model_name)
    prompt_name = option_type

    # output_dir: automatically set to {option_type}_outputs/{model_name}/ structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output_dir is None:
        output_dir = os.path.join(script_dir, f"{option_type}_outputs", model_name)
    else:
        output_dir = args.output_dir

    # Prompt file path
    user_prompt_path = os.path.join(script_dir, PROMPT_PATHS[option_type])

    # Set CSV path (read from the inputs/ directory)
    inputs_dir = os.path.join(script_dir, "inputs")
    if persona_type == "agent":
        csv_path = os.path.join(inputs_dir, "basic_agent_151.csv")
        col_name = "agent_ab_dilemma"
    elif persona_type == "advisor":
        csv_path = os.path.join(inputs_dir, "basic_advisor_156.csv")
        col_name = "advisor_ab_dilemma"
    else:
        raise ValueError(f"Invalid persona type: {persona_type}")

    # ── Print configuration info ──────────────────────────────────────────────
    print(f"=== Configuration info ===")
    print(f"model_name (CLI):  {model_name}")
    print(f"model_id (HF):     {model_id}")
    print(f"provider:          {provider}")
    print(f"model_suffix:      {model_suffix}")
    print(f"option_type:       {option_type}")
    print(f"persona_type:      {persona_type}")
    print(f"start_idx:         {start_idx}")
    print(f"end_idx:           {end_idx if end_idx is not None else 'full dataset'}")
    print(f"user_prompt_path:  {user_prompt_path}")
    print(f"csv_path:          {csv_path}")
    print(f"output_dir:        {output_dir}")
    print(f"================\n")

    # ── Load local model ───────────────────────────────────────────────
    model = None
    tokenizer = None
    max_tokens = get_max_new_tokens(model_name, "extract")

    # ── Read prompt ───────────────────────────────────────────────
    with open(user_prompt_path, 'r', encoding='utf-8') as f:
        user_prompt_template = f.read().strip()

    # ── Read CSV file ───────────────────────────────────────────────
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"Loaded {len(df)} rows in total")

    # ── Apply index range
    if end_idx is None:
        df = df[start_idx:].reset_index(drop=True)
        print(f"Processing range: {start_idx} ~ end ({len(df)} rows total)")
    else:
        df = df[start_idx:end_idx].reset_index(drop=True)
        print(f"Processing range: {start_idx} ~ {end_idx} ({len(df)} rows total)")

    # ── Set column names ──────────────────────────────────────────────
    if option_type == "compromise":
        result_col = f"{model_name}_Compromise alternative"
    else:
        result_col = f"{model_name}_Novel and feasible"

    response_col = f'{model_suffix}_{prompt_name}_response'

    # ── Add response columns ─────────────────────────────────────────────
    df['llm_labeler_name'] = model_name
    df[response_col] = ''
    df[result_col] = ''

    if option_type == "compromise":
        df[f'{model_suffix}_{prompt_name}_tradeoff_rule'] = ''
        df[f'{model_suffix}_{prompt_name}_justification'] = ''
    else:
        df[f'{model_suffix}_{prompt_name}_reframe_type'] = ''
        df[f'{model_suffix}_{prompt_name}_justification'] = ''

    # ── Set result output path (CSV) ─────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    data_len = len(df)
    end_idx_str = "all" if end_idx is None else str(end_idx)
    output_path = os.path.join(
        output_dir,
        f"{persona_type}_{model_suffix}_{prompt_name}_s{start_idx}-e{end_idx_str}_{data_len}.csv",
    )

    def flush_csv() -> None:
        """Save current DataFrame to CSV."""
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # ── Process every row ────────────────────────────────────────
    print(f"\nExtracting responses with {model_name} ({provider}) for each row...")
    print(f"Output file (CSV): {output_path}\n")

    stop_reason = None
    try:
        model, tokenizer = load_local_model(model_name)

        for idx in tqdm(range(len(df)), desc="Processing", unit="row"):
            story = df.iloc[idx][col_name]

            if pd.isna(story) or str(story).strip() == '':
                print(f"\nRow {idx}: story is empty, skipping.")
                continue

            user_prompt = user_prompt_template.replace('{DILEMMA_TEXT}', str(story).strip())
            resp_output_text = None

            max_retries = 1 if model_name == "deepseek-r1-70b" else 5
            row_success = False
            abort_all = False

            for attempt in range(max_retries):
                try:
                    from local_llm_utils import LOCAL_MODEL_REGISTRY
                    _extract_temp = LOCAL_MODEL_REGISTRY.get(model_name, {}).get("extract_temperature", 0.0)
                    resp_output_text = generate_text(
                        model,
                        tokenizer,
                        user_prompt,
                        max_new_tokens=max_tokens,
                        temperature=_extract_temp,
                        model_name=model_name,
                    )
                    if resp_output_text is not None:
                        resp_output_text = resp_output_text.strip()

                    if not resp_output_text:
                        raise ValueError("API response is empty (None or empty string)")

                    print(f"================================ RESPONSE {idx} ================================= ")
                    print(resp_output_text)

                    df.at[idx, response_col] = resp_output_text

                    if option_type == "compromise":
                        option_c, tradeoff_rule, justification = extract_compromise_fields(resp_output_text)
                    else:
                        option_c, reframe_type, justification = extract_non_compromise_fields(resp_output_text)

                    if not option_c:
                        raise ValueError(
                            f"Failed to extract option_c (JSON parse error or missing key). "
                            f"Response: {resp_output_text[:200]}"
                        )

                    # Enforce word-count range retries (handles both JSON and labeled output formats)
                    enforce_retries = 0 if model_name == "deepseek-r1-70b" else 3
                    final_violations = []
                    for enforce_attempt in range(enforce_retries + 1):
                        if option_type == "compromise":
                            violations = _check_range_violations(option_type, option_c, tradeoff_rule, justification)
                        else:
                            violations = _check_range_violations(option_type, option_c, reframe_type, justification)
                        final_violations = violations

                        if not violations:
                            break

                        if enforce_attempt == enforce_retries:
                            print(
                                "Word-count retries exhausted; keeping latest pure model output: " +
                                "; ".join([f"{f}={wc} (target {mn}-{mx})" for f, wc, mn, mx in violations])
                            )
                            break

                        violation_msg = " ".join([
                            f'"{f}" has {wc} words; must be between {mn} and {mx} words.'
                            for f, wc, mn, mx in violations
                        ])
                        retry_prompt = (
                            f"{user_prompt}\n\n"
                            f"Your previous answer violated word limits. {violation_msg} "
                            f"Rewrite the full answer now and strictly satisfy all ranges."
                        )

                        resp_output_text = generate_text(
                            model,
                            tokenizer,
                            retry_prompt,
                            max_new_tokens=max_tokens,
                            temperature=0.3,
                            model_name=model_name,
                        ).strip()

                        print(f"----- WORD-RANGE RETRY row {idx} attempt {enforce_attempt+1}/{enforce_retries} -----")
                        print(resp_output_text)

                        if option_type == "compromise":
                            option_c, tradeoff_rule, justification = extract_compromise_fields(resp_output_text)
                        else:
                            option_c, reframe_type, justification = extract_non_compromise_fields(resp_output_text)

                    df.at[idx, response_col] = resp_output_text
                    df.at[idx, result_col] = option_c
                    if option_type == "compromise":
                        df.at[idx, f'{model_suffix}_{prompt_name}_tradeoff_rule'] = tradeoff_rule
                        df.at[idx, f'{model_suffix}_{prompt_name}_justification'] = justification
                    else:
                        df.at[idx, f'{model_suffix}_{prompt_name}_reframe_type'] = reframe_type
                        df.at[idx, f'{model_suffix}_{prompt_name}_justification'] = justification

                    row_success = True
                    try:
                        flush_csv()
                    except Exception as save_err:
                        print(f"\n❌ Intermediate CSV save failed: {save_err}", file=sys.stderr)
                    break

                except Exception as e:
                    print(f"\n===== ERROR at row {idx} (attempt {attempt+1}/{max_retries}) =====")
                    print(f"Error: {type(e).__name__}: {e}")
                    print(f"Response: {str(resp_output_text)[:200] if resp_output_text else 'None'}")

                    if is_fatal_error(str(e)):
                        print(f"\n❌ Fatal error encountered. Stopping processing (CSV will be saved in finally).")
                        stop_reason = "fatal_api_error"
                        abort_all = True
                        break

                    if attempt < max_retries - 1:
                        wait_sec = 2 * (attempt + 1)
                        print(f"  Waiting {wait_sec}s before retrying... ({attempt+1}/{max_retries})")
                        time.sleep(wait_sec)
                    else:
                        print(f"\n⚠️ Error persisted after {max_retries} retries. Skipping this row and continuing.")

            if abort_all:
                break

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("\n\n⚠️ Interrupted by user. (CSV will be saved in finally)")
        raise

    except Exception as e:
        stop_reason = f"unexpected:{type(e).__name__}"
        print(f"\n❌ Unexpected error: {type(e).__name__}: {e}")
        raise

    finally:
        try:
            flush_csv()
            if stop_reason:
                print(f"\n💾 Partial results saved: {output_path} (stop reason: {stop_reason})")
            else:
                print(f"\n💾 Results saved: {output_path}")
        except Exception as save_err:
            print(f"\n❌ CSV save failed: {save_err}", file=sys.stderr)
        finally:
            cleanup_model(model, tokenizer)

    # Count rows that received a response
    _resp = df[response_col].astype(str).str.strip()
    response_count = int(((_resp != "") & (_resp != "nan")).sum())
    if stop_reason is None:
        print(f"\nProcessing complete.")
    else:
        print(f"\nProcessing interrupted.")
    print(f"Rows with responses added out of {len(df)} total: {response_count}")


if __name__ == "__main__":
    main()
