"""
Task 8 - A/B Option Selection Experiment (Local LLMs)
5-prompt majority voting inference over 2 options (option_a, option_b)

- LOCAL model mirror of task8/five_prompts_ab/1_select_ab.py (closed-model version)
- 5 prompts × Option Shuffling (2 options: A/B) → Majority Voting
- 3 evaluation criteria (ethics, feasibility, conflict-resolution contribution)
- automatic agent / advisor detection
- vLLM (llama4-scout, qwen3.5-122b-a10b, mistral-large-123b) + transformers (qwen3-32b, llama3.3-70b, mistral-small-3.1-24b)
- row-level errors are marked as ERROR and processing continues (no FatalLLMError raised)
"""

# pyright: basic, reportArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportMissingModuleSource=false

import os
import sys
import time
import json
import random
import itertools
import re
from collections import Counter

import pandas as pd

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):
        return iterable

from local_llm_utils import (  # pyright: ignore[reportImplicitRelativeImport]
    load_local_model,
    generate_text,
    cleanup_model,
    get_model_suffix,
    get_max_new_tokens,
    LOCAL_MODEL_REGISTRY,  # pyright: ignore[reportUnusedImport]
)

for _m in os.getenv("SELECT_AB_FORCE_VLLM_MODELS", "").split(","):
    _m = _m.strip()
    if _m and _m in LOCAL_MODEL_REGISTRY:
        LOCAL_MODEL_REGISTRY[_m]["backend"] = "vllm"
        print(f"FORCED VLLM backend for {_m}")


class FatalLLMError(Exception):
    """Fatal error (e.g., model load failure). Row-level LLM failures are marked as ERROR and continue."""
    pass


class LLMRefusalError(Exception):
    """Model refused due to safety policy. Skip remaining prompts for this row."""
    pass


# ============================================================================
# Parameter Settings
# ============================================================================
SCRIPT_DIR = os.getenv("SELECT_AB_SCRIPT_DIR") or os.path.dirname(os.path.abspath(__file__))

ALL_MODELS = [
    "qwen3-32b",
    "llama3.3-70b",
    "mistral-small-3.1-24b",
    "llama4-scout",
    "qwen3.5-122b-a10b",
    "mistral-large-123b",
]

import argparse as _argparse
_parser = _argparse.ArgumentParser(
    description="A/B option selection with local LLMs (5-prompt majority voting)"
)
_parser.add_argument("--model", type=str, default=None, choices=ALL_MODELS,
                     help="Run a single model (memory-safe). When omitted, run all sequentially.")
_parser.add_argument("--start_idx", type=int, default=0,
                     help="Start row index (inclusive, default 0).")
_parser.add_argument("--end_idx", type=int, default=-1,
                     help="End row index (exclusive). -1 = to the end.")
_args = _parser.parse_args()

llm_models = [_args.model] if _args.model else ALL_MODELS
start_idx = _args.start_idx
end_idx = _args.end_idx
output_dir = os.path.join(SCRIPT_DIR, "llm_outputs_ab")

# Input file paths
csv_paths = [
    os.path.join(SCRIPT_DIR, "inputs", "basic_agent_151.csv"),
    os.path.join(SCRIPT_DIR, "inputs", "basic_advisor_156.csv"),
]

# Five-prompt directory / A/B option shuffling settings
FIVE_PROMPTS_DIR = os.path.join(SCRIPT_DIR, "five_prompts_ab")
OPTION_KEYS = ["option_a", "option_b"]
LETTERS = ["A", "B"]
ALL_PERMUTATIONS = list(itertools.permutations(OPTION_KEYS))
NUM_PROMPTS = 5
VALID_OPTIONS = ["Option A", "Option B"]


# ============================================================================
# Utility Functions
# ============================================================================
def sanitize_text(text):
    """Sanitize string to UTF-8 safe form"""
    text = str(text)
    text = text.encode("utf-8", "replace").decode("utf-8")
    return "".join(ch for ch in text if ch >= " " or ch in "\n\r\t")


def load_five_prompts(prompts_dir):
    """Load prompt1.txt ~ prompt5.txt"""
    prompts = []
    for i in range(1, NUM_PROMPTS + 1):
        path = os.path.join(prompts_dir, f"prompt{i}.txt")
        with open(path, "r", encoding="utf-8") as f:
            prompts.append(f.read().strip())
    print(f"✅ Loaded {len(prompts)} prompts ({prompts_dir})")
    return prompts


def generate_five_shuffles(row_idx):
    """
    Generate NUM_PROMPTS shuffle dictionaries.
    For 2 options (A/B) there are only 2 permutations, so 5 are drawn via with-replacement sampling.
    """
    rng = random.Random(row_idx * 7 + 2025)
    total = len(ALL_PERMUTATIONS)
    if total >= NUM_PROMPTS:
        perms = rng.sample(ALL_PERMUTATIONS, NUM_PROMPTS)
    else:
        base = list(ALL_PERMUTATIONS)
        rng.shuffle(base)
        extra = rng.choices(ALL_PERMUTATIONS, k=NUM_PROMPTS - total)
        perms = base + extra
    shuffles = []
    for perm in perms:
        shuffle_dict = {letter: key for letter, key in zip(LETTERS, perm)}
        shuffles.append(shuffle_dict)
    return shuffles


def build_options_text(row, shuffle_dict):
    """Build 'Option A: ... / Option B: ...' text according to shuffle_dict"""
    parts = []
    for letter in LETTERS:
        key = shuffle_dict[letter]
        val = row.get(key, None)
        text = sanitize_text(str(val) if val is not None and not pd.isna(val) else "").strip()
        parts.append(f"Option {letter}: {text}")
    return "\n\n".join(parts)


def majority_vote(votes):
    """Return the most-voted value among valid votes. Empty strings ignored. On tie, the first occurring value."""
    valid_votes = [v for v in votes if v and str(v).strip()]
    if not valid_votes:
        return ""
    counter = Counter(valid_votes)
    max_count = max(counter.values())
    for v in valid_votes:
        if counter[v] == max_count:
            return v
    return ""


# ============================================================================
# Automatic Dataset Detection
# ============================================================================
def detect_dataset_type(df):
    """Auto-detect agent_dilemma vs advisor_dilemma column. A/B-only mode, so option_abcd/option_shuffle are not required."""
    has_agent = "agent_dilemma" in df.columns
    has_advisor = "advisor_dilemma" in df.columns
    if has_agent:
        print("✅ Agent dataset detected: using 'agent_dilemma' + option_a/option_b")
        return "agent_dilemma"
    elif has_advisor:
        print("✅ Advisor dataset detected: using 'advisor_dilemma' + option_a/option_b")
        return "advisor_dilemma"
    else:
        raise ValueError(
            "Input data does not contain an 'agent_dilemma' or 'advisor_dilemma' column.\n"
            f"Available columns: {list(df.columns)}"
        )


# ============================================================================
# Response Parsing
# ============================================================================
def parse_selection(response_text):
    """
    Extract A or B from response → 'Option A' / 'Option B' or ''.
    Never raises (same logic as parse_selection in closed-model 1_select_ab.py).
    """
    try:
        if response_text is None:
            return ""
        if pd.isna(response_text):
            return ""
        text = str(response_text).strip()
        if not text:
            return ""

        clean = re.sub(r"\*+", "", text).strip()
        clean_upper = clean.upper()

        # 1) Entire text is a single letter
        if clean_upper in ["A", "B"]:
            return f"Option {clean_upper}"

        # 2) Collect 'Option X' patterns; unique case
        option_matches = re.findall(r"\bOPTION\s*([AB])\b", clean_upper)
        unique_options = list(dict.fromkeys(option_matches))
        if len(unique_options) == 1:
            return f"Option {unique_options[0]}"

        # 3) Analyze the last line
        last_line = clean.split("\n")[-1].strip()
        last_upper = last_line.upper()
        if last_upper in ["A", "B"]:
            return f"Option {last_upper}"
        last_opt = re.findall(r"\bOPTION\s*([AB])\b", last_upper)
        if len(last_opt) == 1:
            return f"Option {last_opt[0]}"
        last_single = re.findall(r"\b([AB])\b", last_upper)
        if len(last_single) == 1:
            return f"Option {last_single[0]}"

        # 4) Selection-verb patterns
        choice_pat = (
            r"(?:choose|select|pick|chose|selected|recommend|"
            r"answer\s+is|choice\s+is|go\s+with|would\s+be)\s*"
            r"[:\-]?\s*(?:option\s*)?([AB])\b"
        )
        choice_m = re.search(choice_pat, clean_upper)
        if choice_m:
            return f"Option {choice_m.group(1)}"

        # 5) Letter inside brackets
        bracket_m = re.search(r"[\(\[]([AB])[\)\]]", clean_upper)
        if bracket_m:
            return f"Option {bracket_m.group(1)}"

        # 6) Multiple 'Option X' → take the last one
        if len(option_matches) >= 2:
            return f"Option {option_matches[-1]}"

        # 7) First standalone letter anywhere (last resort)
        first_m = re.search(r"\b([AB])\b", clean_upper)
        if first_m:
            return f"Option {first_m.group(1)}"

        return ""
    except Exception:
        return ""


def map_selection_to_judgment(selection, shuffle_dict):
    """Map LLM selection through shuffle_dict to the actual column name."""
    try:
        if not selection or selection not in VALID_OPTIONS:
            return ""
        if isinstance(shuffle_dict, str):
            shuffle_dict = json.loads(shuffle_dict)
        if not isinstance(shuffle_dict, dict):
            return ""
        letter = selection.split()[-1]
        result = shuffle_dict.get(letter, "")
        if not result:
            print(f"  ⚠️ Key '{letter}' not found in shuffle_dict: {shuffle_dict}")
        return result
    except Exception as e:
        print(f"  ⚠️ map_selection_to_judgment error: {e}")
        return ""


# ============================================================================
# Main Execution
# ============================================================================
print(f"=== Configuration info ===")
print(f"llm_models: {llm_models}")
print(f"csv_paths: {csv_paths}")
print(f"prompts_dir: {FIVE_PROMPTS_DIR}")
print(f"output_dir: {output_dir}")
print(f"================\n")

os.makedirs(output_dir, exist_ok=True)
prompt_templates = load_five_prompts(FIVE_PROMPTS_DIR)
for i, pt in enumerate(prompt_templates):
    print(f"\n--- Prompt {i+1} (first 80 chars) ---")
    print(pt[:80] + "...")


for llm_model in llm_models:
    model_suffix = get_model_suffix(llm_model)
    print(f"\n{'='*60}")
    print(f"Loading model: {llm_model} (suffix: {model_suffix})")
    print(f"{'='*60}\n")

    model = None
    tokenizer = None
    try:
        model, tokenizer = load_local_model(llm_model)
        max_new_tokens = get_max_new_tokens(llm_model, "judgment")
        print(f"max_new_tokens: {max_new_tokens}")

        # Process each CSV
        for csv_path in csv_paths:
            print(f"\n{'='*80}")
            print(f"Starting file processing: {csv_path}")
            print(f"{'='*80}\n")

            df_original = pd.read_csv(csv_path, encoding="utf-8")
            print(f"Loaded {len(df_original)} rows in total")

            dilemma_col = detect_dataset_type(df_original)
            print(f"Dilemma column: {dilemma_col}")

            # Check required option columns
            for col in OPTION_KEYS:
                if col not in df_original.columns:
                    raise ValueError(f"Input data does not contain column '{col}'. (required: {OPTION_KEYS})")

            # Apply index range
            if end_idx == -1:
                df = df_original[start_idx:].reset_index(drop=True)
            else:
                df = df_original[start_idx:end_idx].reset_index(drop=True)
            print(f"Processing range: {start_idx} ~ {end_idx} ({len(df)} rows total)\n")

            input_filename = os.path.splitext(os.path.basename(csv_path))[0]
            actual_start = start_idx
            actual_end = len(df_original) - 1 if end_idx == -1 else end_idx - 1

            model_output_dir = os.path.join(output_dir, llm_model)
            os.makedirs(model_output_dir, exist_ok=True)

            output_prefix = f"{model_suffix}_{input_filename}_idx{actual_start}_{actual_end}"

            # Check existing finalized files (supports incomplete-row reprocessing mode)
            existing_finals = [
                f for f in os.listdir(model_output_dir)
                if f.startswith(output_prefix)
                and not f.startswith(f"INTERIM_{model_suffix}")
                and not f.startswith(f"FAILED_{model_suffix}")
                and not f.startswith(f"ERROR_LOG_")
            ]

            error_rows_index = None
            existing_path = None
            existing_df = None
            if existing_finals:
                existing_path = os.path.join(model_output_dir, existing_finals[0])
                existing_df = pd.read_csv(existing_path, encoding="utf-8-sig")
                judgment_col_check = f"{model_suffix}_judgment"
                vote_cols_check = [f"{model_suffix}_p{i+1}_vote" for i in range(NUM_PROMPTS)]
                if (judgment_col_check in existing_df.columns
                        and all(c in existing_df.columns for c in vote_cols_check)):
                    def _is_filled(s):
                        ss = s.astype(str).str.strip()
                        return s.notna() & (ss != "") & (ss.str.lower() != "nan")
                    votes_all_filled = pd.concat(
                        [_is_filled(existing_df[c]) for c in vote_cols_check], axis=1
                    ).all(axis=1)
                    is_refusal = existing_df[judgment_col_check].astype(str).str.strip() == "REFUSAL"
                    incomplete_mask = (~votes_all_filled) & (~is_refusal)
                    error_rows = existing_df[incomplete_mask]
                    if len(error_rows) == 0:
                        print(f"\n⏭️  Skipping: result file is complete → {existing_finals[0]}")
                        continue
                    else:
                        print(f"\n🔄 Found {len(error_rows)} incomplete rows → reprocessing all 5 prompts")
                        error_rows_index = list(error_rows.index)
                        df = existing_df.copy()
                else:
                    print(f"\n⏭️  Skipping: result file exists → {existing_finals[0]}")
                    continue

            # Resume work from INTERIM / FAILED file
            resume_from_idx = 0
            interim_or_failed = [
                f for f in os.listdir(model_output_dir)
                if (f.startswith(f"INTERIM_{model_suffix}_{input_filename}")
                    or f.startswith(f"FAILED_{model_suffix}_{input_filename}"))
            ]
            vote_cols = [f"{model_suffix}_p{i+1}_vote" for i in range(NUM_PROMPTS)]
            judgment_col = f"{model_suffix}_judgment"

            if error_rows_index is not None:
                resume_mode = "error_retry"
                for idx in error_rows_index:
                    for vc in vote_cols:
                        df.at[idx, vc] = ""
                    df.at[idx, judgment_col] = ""
                print(f"   Re-process targets: {error_rows_index[:10]}{'...' if len(error_rows_index)>10 else ''}")
            elif interim_or_failed:
                resume_file = sorted(interim_or_failed)[-1]
                resume_path = os.path.join(model_output_dir, resume_file)
                resumed_df = pd.read_csv(resume_path, encoding="utf-8-sig")
                if judgment_col in resumed_df.columns:
                    j_filled = resumed_df[judgment_col].astype(str).str.strip()
                    completed_mask = (j_filled != "") & (j_filled != "nan")
                    if completed_mask.any():
                        last_done_idx = completed_mask[completed_mask].index[-1]
                        resume_from_idx = last_done_idx + 1
                        df = resumed_df.copy()
                        print(f"\n🔄 Resuming from index {resume_from_idx} (last completed: {last_done_idx})")
                    else:
                        df = resumed_df.copy()
                else:
                    df = resumed_df.copy()
                resume_mode = "continue"
            else:
                resume_mode = "fresh"

            # Initialize columns only in fresh mode
            if resume_mode == "fresh":
                for vc in vote_cols:
                    df[vc] = ""
                df[judgment_col] = ""

            # Process target indices
            if resume_mode == "error_retry":
                target_indices = error_rows_index
                print(f"\nReprocessing {len(target_indices)} incomplete rows across all 5 prompts...")
            elif resume_mode == "continue":
                target_indices = list(range(resume_from_idx, len(df)))
                print(f"\nProcessing indices {resume_from_idx}~{len(df)-1}...")
            else:
                target_indices = list(range(len(df)))
                print(f"\nSelecting options with model {llm_model} for each row (5-prompt majority voting)...")

            interim_path = os.path.join(
                model_output_dir,
                f"INTERIM_{model_suffix}_{input_filename}_idx{actual_start}_{actual_end}.csv"
            )

            fatal_error = None
            error_log = []

            try:
                for idx in tqdm(target_indices, desc=f"Processing {llm_model}", unit="row"):
                    row = df.iloc[idx]
                    dilemma = row[dilemma_col]
                    if pd.isna(dilemma) or str(dilemma).strip() == "":
                        print(f"\nRow {idx}: dilemma is empty, skipping.")
                        continue

                    shuffles = generate_five_shuffles(idx)
                    votes = []
                    row_refused = False
                    row_had_error = False

                    for p_idx in range(NUM_PROMPTS):
                        prompt_tpl = prompt_templates[p_idx]
                        shuffle_dict = shuffles[p_idx]
                        options_text = build_options_text(row, shuffle_dict)
                        user_message = prompt_tpl.replace(
                            "{DILEMMA_TEXT}", sanitize_text(str(dilemma)).strip()
                        ).replace(
                            "{OPTIONS_TEXT}", options_text
                        )

                        vote = ""
                        max_retries = 3
                        last_exception = None
                        last_resp_text = None
                        success = False

                        for attempt in range(max_retries):
                            try:
                                resp_text = generate_text(
                                    model, tokenizer, user_message,
                                    max_new_tokens=max_new_tokens,
                                    temperature=0.0,
                                    model_name=llm_model,
                                )
                                last_resp_text = resp_text
                                selection = parse_selection(resp_text)
                                if selection in VALID_OPTIONS:
                                    vote = map_selection_to_judgment(selection, shuffle_dict)
                                    success = True
                                    break
                                # Parse failed
                                last_exception = ValueError(
                                    f"Parse failed (selection: {selection!r})"
                                )
                                print(f"\n  ⚠️ Row {idx} prompt {p_idx+1}: parse failed "
                                      f"(attempt {attempt+1}/{max_retries})")
                                print(f"  Response: {str(resp_text)[:200]}")
                                if attempt < max_retries - 1:
                                    time.sleep(1)
                            except LLMRefusalError:
                                if not row_refused:
                                    print(f"\n  🚫 Row {idx} prompt {p_idx+1}: model refused → skipping this row.")
                                row_refused = True
                                vote = ""
                                success = True
                                break
                            except Exception as e:
                                last_exception = e
                                print(f"\n  ⚠️ Row {idx} prompt {p_idx+1}: API error "
                                      f"(attempt {attempt+1}/{max_retries}) → {type(e).__name__}: {e}")
                                if attempt < max_retries - 1:
                                    err_str = str(e)
                                    if "503" in err_str or "UNAVAILABLE" in err_str or "overloaded" in err_str.lower():
                                        wait = min(10 * (2 ** attempt), 120)
                                    else:
                                        wait = 2 * (attempt + 1)
                                    print(f"  → Waiting {wait}s before retry...")
                                    time.sleep(wait)

                        if not success:
                            error_log.append({
                                "row_idx": idx,
                                "prompt_idx": p_idx + 1,
                                "error_type": type(last_exception).__name__ if last_exception else "Unknown",
                                "error_message": str(last_exception) if last_exception else "",
                                "last_llm_response": str(last_resp_text) if last_resp_text else "",
                            })
                            print(f"\n  ❌ Row {idx} prompt {p_idx+1}: 3 failures → marking ERROR and moving to next row")
                            row_had_error = True
                            break

                        if row_refused:
                            for remaining_vc in vote_cols[p_idx:]:
                                df.at[idx, remaining_vc] = ""
                            break

                        votes.append(vote)
                        df.at[idx, vote_cols[p_idx]] = vote

                    if row_had_error:
                        df.at[idx, judgment_col] = "ERROR"
                        continue
                    if row_refused:
                        df.at[idx, judgment_col] = "REFUSAL"
                        continue

                    # Majority Voting
                    final_judgment = majority_vote(votes)
                    df.at[idx, judgment_col] = final_judgment

                    # Interim save (every 50 rows)
                    if (idx + 1) % 50 == 0:
                        df.to_csv(interim_path, index=False, encoding="utf-8-sig")
                        print(f"\n  💾 Interim save ({idx+1}/{len(df)} rows): {interim_path}")

            except FatalLLMError as fe:
                fatal_error = fe
                print(f"\n❌ Fatal error encountered: {fe}")

            # Stats
            j_series = df[judgment_col].astype(str).str.strip()
            refusal_count = int((j_series == "REFUSAL").sum())
            error_count = int((j_series == "ERROR").sum())
            valid_count = int(((j_series != "") & (j_series != "REFUSAL") & (j_series != "ERROR")).sum())

            print(f"\n=== {llm_model} judgment stats (Majority Voting) ===")
            print(df[judgment_col].value_counts())
            print(f"Total valid: {valid_count} / {len(df)}")
            if refusal_count > 0:
                print(f"🚫 Refused: {refusal_count}")
            if error_count > 0:
                print(f"❌ Errors: {error_count}")
            for i, vc in enumerate(vote_cols):
                vc_valid = int((df[vc].astype(str).str.strip() != "").sum())
                print(f"  Prompt {i+1} valid: {vc_valid} / {len(df)}")

            # Output path
            if fatal_error is not None:
                final_name = (
                    f"FAILED_{model_suffix}_{input_filename}"
                    f"_idx{actual_start}_{actual_end}_valid{valid_count}.csv"
                )
            else:
                final_name = (
                    f"{model_suffix}_{input_filename}"
                    f"_idx{actual_start}_{actual_end}_valid{valid_count}.csv"
                )
            final_path = os.path.join(model_output_dir, final_name)

            # Clean up existing files
            if resume_mode == "error_retry" and existing_path is not None:
                old_file = os.path.join(model_output_dir, existing_finals[0])
                if os.path.exists(old_file) and os.path.abspath(old_file) != os.path.abspath(final_path):
                    os.remove(old_file)
                    print(f"  🗑️ Deleted previous result file: {existing_finals[0]}")
            if resume_mode == "continue" and interim_or_failed:
                for old_f in interim_or_failed:
                    old_path = os.path.join(model_output_dir, old_f)
                    if os.path.exists(old_path):
                        os.remove(old_path)
                        print(f"  🗑️ Deleted previous INTERIM/FAILED: {old_f}")

            df.to_csv(final_path, index=False, encoding="utf-8-sig")
            if fatal_error is not None:
                print(f"\n⚠️ {llm_model} Interrupted result saved: {final_path}")
            else:
                print(f"\n✅ {llm_model} Result saved: {final_path}")

            # Save error log
            if error_log:
                error_df = pd.DataFrame(error_log)
                error_filename = f"ERROR_LOG_{model_suffix}_{input_filename}_idx{actual_start}_{actual_end}.csv"
                error_path = os.path.join(model_output_dir, error_filename)
                error_df.to_csv(error_path, index=False, encoding="utf-8-sig")
                print(f"  📋 ERROR_LOG: {error_path} ({len(error_log)} entries)")
            else:
                old_error_log = os.path.join(
                    model_output_dir,
                    f"ERROR_LOG_{model_suffix}_{input_filename}_idx{actual_start}_{actual_end}.csv"
                )
                if os.path.exists(old_error_log):
                    os.remove(old_error_log)
                    print(f"  🗑️ Deleted previous error log (no errors)")

            # Clean up interim file
            if os.path.exists(interim_path):
                os.remove(interim_path)
                print(f"  🗑️ Deleted interim file")

            # Abort everything on fatal error
            if fatal_error is not None:
                print(f"\nAborting the entire script.")
                sys.exit(1)

            print(f"\n{'='*80}")
            print(f"File processing complete: {csv_path}")
            print(f"{'='*80}\n")

    finally:
        cleanup_model(model, tokenizer)

print(f"\n{'='*80}")
print(f"All files and models processed!")
print(f"{'='*80}")
