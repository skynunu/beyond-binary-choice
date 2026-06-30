# Can LLMs Imagine Moral Alternatives Beyond Binary Dilemmas?

This repository contains the released data and local-model experiment scripts for parts of the paper **"Can LLMs Imagine Moral Alternatives Beyond Binary Dilemmas?"**

The paper studies whether moral reasoning by LLMs changes when a dilemma is no longer forced into two options. Each item starts with an original A/B moral dilemma and adds two kinds of alternatives:

- **Compromise alternative**: preserves core aims from both A and B through a concrete trade-off.
- **Reframed alternative**: changes the frame of the conflict by adding a new principle, decision logic, stakeholder, or scope.

The released MoralAltDataset contains 307 dilemmas: 156 narrative **Advisor** dilemmas and 151 AI-facing **Agent** dilemmas. The scripts here cover alternative generation and choice judgment experiments with local or vLLM-served models.

## Repository Layout

```text
1_extract_llm_alternative_response_local.py  # Generate compromise or reframed alternatives
2_extract_abcd_judgment_local.py             # Choose among A/B/compromise/reframed options
3_extract_ab_judgment_local.py               # Choose between the original A/B options
local_llm_utils.py                           # Shared model loading and generation helpers

inputs/                                      # Runnable input CSVs used by the scripts
third_alternative_prompts/                   # Prompts for alternative generation
five_prompts/                                # Five prompts for A/B/C/D judgment
five_prompts_ab/                             # Five prompts for A/B judgment

MoralAlt_dataset/                            # Released dataset and collected outputs
compromise_outputs/                          # Example/generated compromise outputs
non_compromise_outputs/                      # Example/generated reframed outputs
judgment_outputs/                            # Example/generated A/B/C/D judgment outputs
```

In the code, `non_compromise` refers to the paper's reframed or "novel and feasible" alternative.

## Setup

Clone the repository and install the main Python dependencies:

```bash
git clone https://github.com/skynunu/beyond-binary-choice.git
cd beyond-binary-choice

python -m venv .venv
source .venv/bin/activate
pip install pandas tqdm torch transformers accelerate bitsandbytes sentencepiece protobuf openai
```

Use a PyTorch/CUDA build that matches your GPU environment. Some Hugging Face models are gated and require login before loading.

Model names, Hugging Face IDs, backends, and generation limits are defined in `local_llm_utils.py`. Transformer-backed models are loaded with 4-bit NF4 quantization. vLLM-backed models use an OpenAI-compatible server; start that server separately and set:

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
```

## Quick Start

Run a small 5-row smoke test first.

Generate compromise alternatives:

```bash
python 1_extract_llm_alternative_response_local.py \
  --model qwen3-32b \
  --option_type compromise \
  --persona_type agent \
  --start_idx 0 \
  --end_idx 5
```

Generate reframed alternatives:

```bash
python 1_extract_llm_alternative_response_local.py \
  --model qwen3-32b \
  --option_type non_compromise \
  --persona_type advisor \
  --start_idx 0 \
  --end_idx 5
```

Run four-option judgment over A, B, compromise, and reframed alternatives:

```bash
python 2_extract_abcd_judgment_local.py \
  --model qwen3-32b \
  --start_idx 0 \
  --end_idx 5
```

Run the original binary A/B judgment baseline:

```bash
python 3_extract_ab_judgment_local.py \
  --model qwen3-32b \
  --start_idx 0 \
  --end_idx 5
```

## What Each Script Does

### 1. Alternative generation

`1_extract_llm_alternative_response_local.py` reads `inputs/basic_agent_151.csv` or `inputs/basic_advisor_156.csv` and asks a model to generate one new Option C.

- `--option_type compromise` uses `third_alternative_prompts/last_compromise_prompt.txt`.
- `--option_type non_compromise` uses `third_alternative_prompts/last_non_compromise_prompt.txt`.
- `--persona_type agent` uses the Agent dilemmas.
- `--persona_type advisor` uses the Advisor dilemmas.

Outputs are saved under:

```text
compromise_outputs/{model}/
non_compromise_outputs/{model}/
```

The output CSV keeps the raw model response plus parsed fields such as generated alternative, trade-off rule, reframe type, and justification.

### 2. Four-option judgment

`2_extract_abcd_judgment_local.py` evaluates whether a model selects one of four options:

- original Option A
- original Option B
- compromise alternative
- reframed alternative

For each row, the script uses five prompt templates from `five_prompts/`, shuffles option order to reduce position bias, parses the selected letter, maps it back to the true option type, and stores the majority-vote result.

Outputs are saved under:

```text
judgment_outputs/{model}/
```

### 3. Binary A/B judgment

`3_extract_ab_judgment_local.py` runs the same five-prompt majority-vote procedure, but only between the original A and B options. This provides the binary-choice baseline before alternatives are introduced.

Outputs are saved under:

```text
llm_outputs_ab/{model}/
```

## Data Notes

- `inputs/basic_agent_151.csv` and `inputs/basic_advisor_156.csv` are the runnable CSVs used by the local scripts.
- `MoralAlt_dataset/` contains the released dataset files, including alternative datasets and LLM/human judgment outputs.
- Existing output folders include previously generated CSVs that can be inspected without rerunning large models.

This repository focuses on the alternative-generation and judgment portions of the experiments. Plotting and full paper-level statistical analysis may require additional analysis code beyond these local inference scripts.


### Source datasets

| Subset in this repository | Source dataset | Original license | How it is used here | Main modifications |
|---|---|---|---|---|
| **Agent** dilemmas | **AIRiskDilemmas / LitmusValues** | **CC BY 4.0** | Used as the source for AI-facing A/B moral dilemmas. Source: https://huggingface.co/datasets/kellycyy/AIRiskDilemmas | Subset selection, reformatting, prompt-based generation of compromise and reframed alternatives, and LLM/human judgment collection. |
| **Advisor** dilemmas | **MPST: Movie Plot Synopses with Tags** | **CC BY-SA 4.0** | Used as a source of narrative movie information for constructing Advisor dilemmas. Source: https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags | Subset selection, transformation into Advisor-style dilemma items, reformatting, prompt-based generation of compromise and reframed alternatives, and LLM/human judgment collection. |

## License

The dataset files are released under CC BY-SA 4.0 because they include adapted material from MPST, which is licensed under CC BY-SA 4.0. The source code in this repository is released under the **Apache License 2.0**.


### Paper citation

```bibtex
@article{chiu2025litmusvalues,
  title={Will AI Tell Lies to Save Sick Children? Litmus-Testing AI Values Prioritization with AIRiskDilemmas},
  author={Chiu, Yu Ying and Wang, Zhilin and Maiya, Sharan and Choi, Yejin and Fish, Kyle and Levine, Sydney and Hubinger, Evan},
  journal={arXiv preprint arXiv:2505.14633},
  year={2025}
}
```

