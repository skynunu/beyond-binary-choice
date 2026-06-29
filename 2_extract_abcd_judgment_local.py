"""
Task 2 - Option Selection Experiment (4 options: A, B, C, D)
Experiment to see what LLM models choose when given 4 options
- 5 prompts x 5 Option Shufflings -> Majority Voting
- Judgment based on 3 evaluation criteria (ethics, feasibility, conflict resolution contribution)
- Automatic detection of agent/advisor datasets
- Mapping to actual selection values via the option_shuffle dictionary
"""

# pyright: basic, reportArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportMissingModuleSource=false

import os
import sys
import time
import json
import random
import itertools
import pandas as pd
import re
from collections import Counter

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
)


class FatalLLMError(Exception):
    """Exception raised when an LLM call fails the maximum number of retries.
    When this exception is raised, abort the main loop and save the results gathered so far."""
    pass


# ============================================================================
# Parameter Settings
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ALL_MODELS = [
    "qwen3-32b",
    "deepseek-r1-70b",
    "llama3.3-70b",
    "mistral-small-3.1-24b",
    "llama4-scout",
    "qwen3.5-122b-a10b",
]

import argparse as _argparse
_parser = _argparse.ArgumentParser(description="Judgment extraction with local models")
_parser.add_argument("--model", type=str, default=None, choices=ALL_MODELS,
                     help="Run a single model (recommended for memory). Omit to run all sequentially.")
_parser.add_argument("--start_idx", type=int, default=0,
                     help="Start row index, inclusive (default: 0).")
_parser.add_argument("--end_idx", type=int, default=-1,
                     help="End row index, exclusive. -1 (default) means run to end of dataset. "
                          "Used by smoke/sample modes and 5-row previews.")
_args = _parser.parse_args()

llm_models = [_args.model] if _args.model else ALL_MODELS
start_idx = _args.start_idx
end_idx = _args.end_idx
output_dir = os.path.join(SCRIPT_DIR, "judgment_outputs")

# Input file paths (a list allows processing multiple files)
csv_paths = [
    os.path.join(SCRIPT_DIR, "inputs", "basic_agent_151.csv"),
    os.path.join(SCRIPT_DIR, "inputs", "basic_advisor_156.csv"),
]

# 5-prompt directory / option shuffling settings
FIVE_PROMPTS_DIR = os.path.join(SCRIPT_DIR, "five_prompts")
OPTION_KEYS = ["option_a", "option_b", "Compromise alternative", "Novel and feasible alternative"]
LETTERS = ["A", "B", "C", "D"]
ALL_PERMUTATIONS = list(itertools.permutations(OPTION_KEYS))
NUM_PROMPTS = 5


# ============================================================================
# Utility Functions
# ============================================================================
def sanitize_text(text):
    """Sanitize string to UTF-8 safe form before sending to API"""
    text = str(text)
    text = text.encode("utf-8", "replace").decode("utf-8")
    return "".join(ch for ch in text if ch >= " " or ch in "\n\r\t")


# ============================================================================
# Load 5 Prompts & Option Shuffling & Majority Voting
# ============================================================================
def load_five_prompts(prompts_dir):
    """Load prompt1.txt ~ prompt5.txt from the five_prompts directory"""
    prompts = []
    for i in range(1, NUM_PROMPTS + 1):
        path = os.path.join(prompts_dir, f"prompt{i}.txt")
        with open(path, 'r', encoding='utf-8') as f:
            prompts.append(f.read().strip())
    print(f"✅ Loaded {len(prompts)} prompts ({prompts_dir})")
    return prompts


def generate_five_shuffles(row_idx):
    """
    Generate 5 different Option Shuffle dictionaries based on the row index.
    Select 5 out of 24 possible permutations without duplicates and return reproducibly.
    """
    rng = random.Random(row_idx * 7 + 2025)
    perms = rng.sample(ALL_PERMUTATIONS, NUM_PROMPTS)
    shuffles = []
    for perm in perms:
        shuffle_dict = {letter: key for letter, key in zip(LETTERS, perm)}
        shuffles.append(shuffle_dict)
    return shuffles


def build_options_text(row, shuffle_dict):
    """Build option text according to shuffle_dict (format: Option A: ... / Option B: ...)"""
    parts = []
    for letter in LETTERS:
        key = shuffle_dict[letter]
        val = row.get(key, None)
        text = sanitize_text(str(val) if val is not None and not pd.isna(val) else "").strip()
        parts.append(f"Option {letter}: {text}")
    return "\n\n".join(parts)


def majority_vote(votes):
    """
    Majority voting: return the most frequently selected value among valid votes.
    Ignore empty strings. On ties, return the value that appeared first.
    """
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
    """
    Automatic dataset type detection (agent / advisor)
    - agent dataset: has 'agent_dilemma' and 'option_abcd' columns
    - advisor dataset: has 'advisor_dilemma' and 'option_abcd' columns

    Returns:
        dilemma_col (str): name of the dilemma column
    """
    has_agent = "agent_dilemma" in df.columns
    has_advisor = "advisor_dilemma" in df.columns
    has_option_abcd = "option_abcd" in df.columns
    has_option_shuffle = "option_shuffle" in df.columns

    if not has_option_abcd:
        raise ValueError("Input data does not contain an 'option_abcd' column.")
    if not has_option_shuffle:
        raise ValueError("Input data does not contain an 'option_shuffle' column.")

    if has_agent:
        print("✅ Agent dataset detected: using 'agent_dilemma' + 'option_abcd' columns")
        return "agent_dilemma"
    elif has_advisor:
        print("✅ Advisor dataset detected: using 'advisor_dilemma' + 'option_abcd' columns")
        return "advisor_dilemma"
    else:
        raise ValueError(
            "Input data does not contain an 'agent_dilemma' or 'advisor_dilemma' column.\n"
            f"Available columns: {list(df.columns)}"
        )


# ============================================================================
# Response Parsing Function
# ============================================================================
VALID_OPTIONS = ["Option A", "Option B", "Option C", "Option D"]


def parse_selection(response_text):
    """
    Extract one of A, B, C, D from the response and return it in the form 'Option X'.
    Never raises an exception; returns an empty string ("") on parsing failure.

    Processing priority:
      1. Entire response is a single letter (A/B/C/D)
      2. Only one unique 'Option X' pattern exists
      3. Look for a standalone letter on the last line
      4. Selection verb patterns (choose/select/pick/answer is)
      5. Letter inside brackets (A), [A]
      6. Multiple 'Option X' patterns -> the last one (conclusion pattern)
      7. First standalone letter in the entire response
    """
    try:
        if response_text is None:
            return ""
        if pd.isna(response_text):
            return ""
        text = str(response_text).strip()
        if not text:
            return ""

        clean = re.sub(r'\*+', '', text).strip()
        clean_upper = clean.upper()

        # 1) Entire response is a single letter
        if clean_upper in ['A', 'B', 'C', 'D']:
            return f"Option {clean_upper}"

        # 2) Collect 'Option X' patterns
        option_matches = re.findall(r'\bOPTION\s*([ABCD])\b', clean_upper)
        unique_options = list(dict.fromkeys(option_matches))
        if len(unique_options) == 1:
            return f"Option {unique_options[0]}"

        # 3) Analyze the last line (pattern where the LLM writes the answer at the end)
        last_line = clean.split('\n')[-1].strip()
        last_upper = last_line.upper()
        if last_upper in ['A', 'B', 'C', 'D']:
            return f"Option {last_upper}"
        last_opt = re.findall(r'\bOPTION\s*([ABCD])\b', last_upper)
        if len(last_opt) == 1:
            return f"Option {last_opt[0]}"
        last_single = re.findall(r'\b([ABCD])\b', last_upper)
        if len(last_single) == 1:
            return f"Option {last_single[0]}"

        # 4) Selection verb patterns
        choice_pat = (
            r'(?:choose|select|pick|chose|selected|recommend|'
            r'answer\s+is|choice\s+is|go\s+with|would\s+be)\s*'
            r'[:\-]?\s*(?:option\s*)?([ABCD])\b'
        )
        choice_m = re.search(choice_pat, clean_upper)
        if choice_m:
            return f"Option {choice_m.group(1)}"

        # 5) Letter inside brackets
        bracket_m = re.search(r'[\(\[]([ABCD])[\)\]]', clean_upper)
        if bracket_m:
            return f"Option {bracket_m.group(1)}"

        # 6) Multiple 'Option X' -> the last one
        if len(option_matches) >= 2:
            return f"Option {option_matches[-1]}"

        # 7) First standalone letter in the entire response (last resort)
        first_m = re.search(r'\b([ABCD])\b', clean_upper)
        if first_m:
            return f"Option {first_m.group(1)}"

        return ""
    except Exception:
        return ""


def map_selection_to_judgment(selection, shuffle_dict):
    """
    Map the LLM selection (Option A/B/C/D) to the actual selection value
    (e.g., 'Compromise alternative', 'option_a', etc.) via shuffle_dict.
    Never raises an exception; returns an empty string on mapping failure.

    Args:
        selection: parsed selection (e.g., "Option A")
        shuffle_dict: dict or JSON string
            e.g., {"A": "Compromise alternative", "B": "option_b", ...}
    """
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
            print(f"  ⚠️ '{letter}' key not found in shuffle_dict: {shuffle_dict}")
        return result
    except Exception as e:
        print(f"  ⚠️ map_selection_to_judgment error: {e}")
        return ""


# ============================================================================
# Main Execution
# ============================================================================
print(f"=== Settings ===")
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
    model_output_dir = os.path.join(output_dir, llm_model)
    os.makedirs(model_output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Loading model: {llm_model} (local, suffix: {model_suffix})")
    print(f"{'='*60}")

    model = None
    tokenizer = None
    try:
        model, tokenizer = load_local_model(llm_model)
        max_tokens = get_max_new_tokens(llm_model, "judgment")

        for csv_path in csv_paths:
            print(f"\n{'='*80}")
            print(f"Starting file processing: {csv_path}")
            print(f"{'='*80}\n")

            print(f"Reading CSV file: {csv_path}")
            df_original = pd.read_csv(csv_path, encoding='utf-8')
            print(f"Loaded a total of {len(df_original)} rows")

            dilemma_col = detect_dataset_type(df_original)
            print(f"Dilemma column: {dilemma_col}")

            for col in OPTION_KEYS:
                if col not in df_original.columns:
                    raise ValueError(f"Input data does not contain a '{col}' column. (required: {OPTION_KEYS})")

            if end_idx == -1:
                df = df_original[start_idx:].reset_index(drop=True)
            else:
                df = df_original[start_idx:end_idx].reset_index(drop=True)
            print(f"Processing range: {start_idx} ~ {end_idx} (total {len(df)} rows)\n")

            input_filename = os.path.splitext(os.path.basename(csv_path))[0]

            actual_start = start_idx
            actual_end = len(df_original) - 1 if end_idx == -1 else end_idx - 1

            output_prefix = f"{model_suffix}_{input_filename}_idx{actual_start}_{actual_end}"
            existing_finals = [
                f for f in os.listdir(model_output_dir)
                if f.startswith(output_prefix) and not f.startswith(f"INTERIM_{model_suffix}") and not f.startswith(f"FAILED_{model_suffix}")
            ]
            if existing_finals:
                print(f"\n⏭️  Skipping: result file already exists -> {existing_finals[0]}")
                continue

            print(f"\nProcessing model: {llm_model} (local, suffix: {model_suffix})")
            print(f"Method: {NUM_PROMPTS} prompts x Option Shuffling -> Majority Voting")

            vote_cols = [f'{model_suffix}_p{i+1}_vote' for i in range(NUM_PROMPTS)]
            judgment_col = f'{model_suffix}_judgment'
            for vc in vote_cols:
                df[vc] = ''
            df[judgment_col] = ''

            print(f"\nSelecting options with the {llm_model} model for each row (5-prompt majority voting)...")

            interim_path = os.path.join(
                model_output_dir,
                f"INTERIM_{model_suffix}_{input_filename}_idx{actual_start}_{actual_end}.csv"
            )

            fatal_error = None
            try:
                for idx in tqdm(range(len(df)), desc=f"Processing {llm_model}", unit="row"):
                    row = df.iloc[idx]

                    dilemma = row[dilemma_col]
                    if pd.isna(dilemma) or str(dilemma).strip() == '':
                        print(f"\nRow {idx}: dilemma is empty, skipping.")
                        continue

                    shuffles = generate_five_shuffles(idx)
                    votes = []

                    row_refused = False
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
                                    model,
                                    tokenizer,
                                    user_message,
                                    max_new_tokens=max_tokens,
                                    temperature=0.0,
                                    model_name=llm_model,
                                )
                                last_resp_text = resp_text
                                selection = parse_selection(resp_text)
                                if selection in VALID_OPTIONS:
                                    vote = map_selection_to_judgment(selection, shuffle_dict)
                                    success = True
                                    break
                                last_exception = ValueError(
                                    f"Parsing failed (selection: {selection!r})"
                                )
                                print(f"\n  ⚠️ Row {idx} prompt{p_idx+1}: parsing failed "
                                      f"(attempt {attempt+1}/{max_retries})")
                                print(f"  ┌─ Full LLM response ─────────────────────────")
                                print(f"  │ {str(resp_text)}")
                                print(f"  └───────────────────────────────────────────")
                                if attempt < max_retries - 1:
                                    print(f"  -> Waiting before retry...")
                                    time.sleep(1 * (attempt + 1))
                            except Exception as e:
                                last_exception = e
                                print(f"\n  ⚠️ Row {idx} prompt{p_idx+1}: generation error "
                                      f"(attempt {attempt+1}/{max_retries}) -> "
                                      f"{type(e).__name__}: {e}")
                                if last_resp_text is not None:
                                    print(f"  ┌─ Previous LLM response ─────────────────────────")
                                    print(f"  │ {str(last_resp_text)}")
                                    print(f"  └───────────────────────────────────────────")
                                else:
                                    print(f"  (exception occurred before receiving any LLM response in this attempt)")
                                if attempt < max_retries - 1:
                                    print(f"  -> Waiting before retry...")
                                    time.sleep(2 * (attempt + 1))

                        if not success:
                            raise FatalLLMError(
                                f"All {max_retries} attempts failed at row {idx} prompt{p_idx+1}.\n"
                                f"  Last exception: "
                                f"{type(last_exception).__name__ if last_exception else 'None'}: "
                                f"{last_exception}\n"
                                f"  Full last LLM response:\n"
                                f"  ─────────────────────────────────────────────\n"
                                f"  {str(last_resp_text)}\n"
                                f"  ─────────────────────────────────────────────"
                            )

                        if row_refused:
                            for remaining_vc in vote_cols[p_idx:]:
                                df.at[idx, remaining_vc] = ""
                            break

                        votes.append(vote)
                        df.at[idx, vote_cols[p_idx]] = vote

                    if row_refused:
                        df.at[idx, judgment_col] = "REFUSAL"
                        continue

                    final_judgment = majority_vote(votes)
                    df.at[idx, judgment_col] = final_judgment

                    if (idx + 1) % 50 == 0:
                        df.to_csv(interim_path, index=False, encoding='utf-8-sig')
                        print(f"\n  💾 Interim save ({idx+1}/{len(df)} rows): {interim_path}")

            except FatalLLMError as fe:
                fatal_error = fe
                print(f"\n\n{'❌'*20}")
                print(f"❌ Fatal error encountered -> aborting code execution.")
                print(f"❌ {fe}")
                print(f"{'❌'*20}\n")

            j_series = df[judgment_col].astype(str).str.strip()
            refusal_count = int((j_series == 'REFUSAL').sum())
            valid_count = int(((j_series != '') & (j_series != 'REFUSAL')).sum())

            print(f"\n=== {llm_model} judgment statistics (Majority Voting) ===")
            judgment_counts = df[judgment_col].value_counts()
            print(judgment_counts)
            print(f"Total Majority Voting success rows: {valid_count} / {len(df)}")
            if refusal_count > 0:
                print(f"🚫 Safety policy refusal rows: {refusal_count} / {len(df)}")

            for i, vc in enumerate(vote_cols):
                vc_valid = int((df[vc].astype(str).str.strip() != '').sum())
                print(f"  Prompt{i+1} valid responses: {vc_valid} / {len(df)}")

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

            df.to_csv(final_path, index=False, encoding='utf-8-sig')
            if fatal_error is not None:
                print(f"\n⚠️ {llm_model} aborted results saved: {final_path}")
            else:
                print(f"\n✅ {llm_model} results saved: {final_path}")

            if os.path.exists(interim_path):
                os.remove(interim_path)
                print(f"  🗑️ Interim save file deleted: {interim_path}")

            if fatal_error is not None:
                print(f"\nAborting the entire script.")
                sys.exit(1)

            print(f"\n{'='*80}")
            print(f"Finished processing file: {csv_path}")
            print(f"{'='*80}\n")

    finally:
        cleanup_model(model, tokenizer)

print(f"\n{'='*80}")
print(f"All files and models processed!")
print(f"{'='*80}")
