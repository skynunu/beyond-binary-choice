"""
Local LLM Utilities for Ethics Research Task8

Hybrid loader supporting two backends:
  1) HuggingFace Transformers + bitsandbytes 4-bit NF4 (legacy models)
       - llama3.3-70b, qwen3-32b, deepseek-r1-70b, mistral-small-3.1-24b
  2) vLLM OpenAI-compatible HTTP server (B200 1-GPU FP8 deployment)
       - llama4-scout      → meta-llama/Llama-4-Scout-17B-16E-Instruct (on-the-fly FP8)
       - qwen3.5-122b-a10b → Qwen/Qwen3.5-122B-A10B-FP8 (official FP8 release)

For (2), an external vLLM server must be running on http://localhost:8000/v1
(see run_b200_staged.sh for automated start/stop). The Python code only acts
as an OpenAI client; it does NOT load model weights into the same process.

Usage:
    from local_llm_utils import load_local_model, generate_text, cleanup_model
    model, tokenizer = load_local_model("qwen3.5-122b-a10b")
    response = generate_text(model, tokenizer, "Hello", model_name="qwen3.5-122b-a10b")
    cleanup_model(model, tokenizer)
"""

import gc
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Mistral3ForConditionalGeneration


# ============================================================================
# Model Registry
# ============================================================================
# Notes on model IDs:
#   - meta-llama/Llama-4-Scout-17B-16E-Instruct  (109B total, 17B activated, 16 experts)
#     — `meta-llama/Llama-4-109B-Instruct` does NOT exist on HuggingFace.
#   - Qwen/Qwen3.5-122B-A10B                     (122B total, 10B activated, 256 experts)
#     — `Qwen/Qwen3.5-122B-A10B-Instruct` does NOT exist; the model is shipped without
#       the `-Instruct` suffix.
#   - Qwen/Qwen3.5-122B-A10B-FP8                 (official FP8 release for B200)

LOCAL_MODEL_REGISTRY = {
    "llama3.3-70b": {
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "max_new_tokens_judgment": 128,
        "max_new_tokens_extract": 512,
        "requires_hf_login": True,
        "thinking_model": False,
        "system_prompt": "",
        "extract_temperature": 0.7,
        "backend": "transformers",
    },
    "qwen3-32b": {
        "model_id": "Qwen/Qwen3-32B",
        "max_new_tokens_judgment": 128,
        "max_new_tokens_extract": 512,
        "requires_hf_login": False,
        "thinking_model": True,
        "backend": "transformers",
    },
    "deepseek-r1-70b": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "max_new_tokens_judgment": 4096,
        "max_new_tokens_extract": 1024,
        "requires_hf_login": False,
        "thinking_model": True,
        "system_prompt": "Return only the final answer in the requested output format. Do not output chain-of-thought or <think> content.",
        "backend": "transformers",
    },
    "mistral-small-3.1-24b": {
        "model_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "max_new_tokens_judgment": 128,
        "max_new_tokens_extract": 512,
        "requires_hf_login": False,
        "thinking_model": False,
        "system_prompt": "",
        "backend": "transformers",
    },
    # ── B200 deployment via vLLM ─────────────────────────────────────────────
    "llama4-scout": {
        # 109B total, 17B activated, 16 experts. Multimodal; we use text-only.
        "model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "max_new_tokens_judgment": 128,
        "max_new_tokens_extract": 512,
        "requires_hf_login": True,
        "thinking_model": False,
        "system_prompt": "",
        "extract_temperature": 0.7,
        "backend": "vllm",
        "is_multimodal": True,
    },
    "qwen3.5-122b-a10b": {
        # Official FP8 release for B200. 122B total, 10B activated, 256 experts.
        # Multimodal; --language-model-only is used at vLLM startup.
        "model_id": "Qwen/Qwen3.5-122B-A10B-FP8",
        "max_new_tokens_judgment": 128,
        "max_new_tokens_extract": 512,
        "requires_hf_login": False,
        "thinking_model": True,
        "backend": "vllm",
        "is_multimodal": True,
    },
}


# Backward-compatibility alias: some older scripts may still reference these
# legacy keys. They are NOT defined here on purpose to surface the inconsistency
# loudly via KeyError. To migrate older callers, replace `llama4-109b` with
# `llama4-scout` in their imports.


# ============================================================================
# Helpers
# ============================================================================
def get_model_suffix(model_name: str) -> str:
    return model_name.replace("-", "").replace(".", "")


def list_available_models():
    return list(LOCAL_MODEL_REGISTRY.keys())


def get_backend(model_name: str) -> str:
    """Return 'transformers' or 'vllm' for a registered model."""
    cfg = LOCAL_MODEL_REGISTRY.get(model_name, {})
    return cfg.get("backend", "transformers")


def is_vllm_model(model_name: str) -> bool:
    return get_backend(model_name) == "vllm"


# ============================================================================
# vLLM client wrapper
# ============================================================================
class _VLLMClient:
    """Lightweight wrapper around an OpenAI-compatible vLLM server.

    The vLLM server is expected to be launched out-of-band (typically by
    run_b200_staged.sh) and to serve EXACTLY ONE model at /v1.
    """

    def __init__(self, model_name: str, model_id: str, base_url: str):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "Package 'openai' is required for vLLM-backed models. "
                "Install with: pip install openai"
            ) from e

        self.model_name = model_name
        self.model_id = model_id
        self.base_url = base_url
        self._client = OpenAI(base_url=base_url, api_key="EMPTY")

    def chat(self, messages, max_tokens, temperature, extra_body=None):
        kwargs = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        # vLLM accepts temperature=0 for greedy. OpenAI SDK passes it through.
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def __repr__(self):
        return f"_VLLMClient(model={self.model_id}, base_url={self.base_url})"


def _get_vllm_base_url() -> str:
    """Return vLLM server base URL from env or fall back to localhost:8000."""
    return os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")


# ============================================================================
# Loading
# ============================================================================
def load_local_model(model_name: str):
    """Load a model. Returns (model, tokenizer) for legacy callers.

    For vLLM-backed models, `model` is a _VLLMClient and `tokenizer` is None.
    For transformers-backed models, the previous BNB 4-bit NF4 path is preserved.
    """
    if model_name not in LOCAL_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(LOCAL_MODEL_REGISTRY.keys())}"
        )

    config = LOCAL_MODEL_REGISTRY[model_name]
    model_id = config["model_id"]
    backend = config.get("backend", "transformers")

    print(f"\n{'='*60}")
    print(f"Loading model: {model_name} ({model_id}) [backend={backend}]")
    print(f"{'='*60}")

    # ── vLLM backend: server must already be running ─────────────────────────
    if backend == "vllm":
        base_url = _get_vllm_base_url()
        client = _VLLMClient(model_name=model_name, model_id=model_id, base_url=base_url)

        # Probe server: list models and warn if expected model is not loaded.
        try:
            import urllib.request, json as _json
            with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as r:
                served = _json.load(r)
            served_ids = [m.get("id") for m in served.get("data", [])]
            if model_id not in served_ids:
                print(
                    f"  WARNING: vLLM server at {base_url} reports models {served_ids}, "
                    f"but Task8 expects '{model_id}'. Continuing — calls may fail."
                )
            else:
                print(f"  vLLM server OK: serving {model_id}")
        except Exception as e:
            print(f"  WARNING: could not probe vLLM server at {base_url}: {e}")
            print(f"  Make sure 'vllm serve {model_id} --port 8000 ...' is running.")

        return client, None

    # ── transformers + bnb 4-bit NF4 (legacy path, unchanged) ────────────────
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU VRAM: {free/1024**3:.1f}GB free / {total/1024**3:.1f}GB total")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading tokenizer...")
    tokenizer_kwargs = {"trust_remote_code": True}
    if model_name == "mistral-small-3.1-24b":
        tokenizer_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model weights (may take several minutes)...")
    if model_name == "mistral-small-3.1-24b":
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
    model.eval()

    if torch.cuda.is_available():
        free_after, _ = torch.cuda.mem_get_info(0)
        used = (free - free_after) / 1024**3
        print(f"Model loaded. Used {used:.1f}GB VRAM. "
              f"Remaining: {free_after/1024**3:.1f}GB free")

    return model, tokenizer


# ============================================================================
# Generation
# ============================================================================
def strip_thinking_tags(text: str) -> str:
    if not text:
        return text
    # regex: <think>...</think> blocks including multiline; also unclosed <think> from truncation
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _build_messages(model_name: str, user_message: str):
    config = LOCAL_MODEL_REGISTRY.get(model_name, {})
    sys_msg = config.get("system_prompt", "")
    messages = []
    if sys_msg:
        messages.append({"role": "system", "content": sys_msg})
    messages.append({"role": "user", "content": user_message})
    return messages


def _generate_via_vllm(
    client: "_VLLMClient",
    user_message: str,
    max_new_tokens: int,
    temperature: float,
    model_name: str,
) -> str:
    config = LOCAL_MODEL_REGISTRY.get(model_name, {})
    is_thinking_model = config.get("thinking_model", False)

    messages = _build_messages(model_name, user_message)

    extra_body = {}
    # Disable Qwen-style "thinking mode" so we get a direct answer matching
    # the original transformers code path that passed enable_thinking=False.
    if model_name in {"qwen3-32b", "qwen3.5-122b-a10b", "deepseek-r1-70b"}:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    response = client.chat(
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=temperature if (temperature and temperature > 0) else 0.0,
        extra_body=extra_body or None,
    )

    if is_thinking_model:
        response = strip_thinking_tags(response)

    return (response or "").strip()


def _generate_via_transformers(
    model,
    tokenizer,
    user_message: str,
    max_new_tokens: int,
    temperature: float,
    model_name: str,
) -> str:
    config = LOCAL_MODEL_REGISTRY.get(model_name, {})
    is_thinking_model = config.get("thinking_model", False)

    messages = _build_messages(model_name, user_message)

    try:
        if model_name in {"qwen3-32b", "qwen3.5-122b-a10b", "deepseek-r1-70b"}:
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    except Exception:
        # Some models/tokenizers (e.g. Mistral3) may not define chat_template.
        # Fall back to plain user prompt for text-only generation.
        input_text = user_message

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    if temperature and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(temperature)
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    input_length = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # fix byte-level BPE artifacts (Ġ=space, Ċ=newline) from some tokenizers
    response = response.replace('\u0120', ' ').replace('\u010a', '\n')

    if is_thinking_model:
        response = strip_thinking_tags(response)

    return response.strip()


def generate_text(
    model,
    tokenizer,
    user_message: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    model_name: str = "",
) -> str:
    """Generate text. Dispatches to vLLM client or transformers based on model type."""
    if isinstance(model, _VLLMClient):
        return _generate_via_vllm(
            model, user_message, max_new_tokens, temperature, model_name
        )
    return _generate_via_transformers(
        model, tokenizer, user_message, max_new_tokens, temperature, model_name
    )


# ============================================================================
# Cleanup
# ============================================================================
def cleanup_model(model=None, tokenizer=None):
    # vLLM server lives in a separate process — no GPU memory to free here.
    if isinstance(model, _VLLMClient):
        print("vLLM client mode: server lifecycle is managed externally. "
              "No in-process GPU memory to free.")
        return

    print(f"\nCleaning up GPU memory...")
    free_before = 0
    total = 0
    if torch.cuda.is_available():
        free_before, total = torch.cuda.mem_get_info(0)

    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_after, _ = torch.cuda.mem_get_info(0)
        freed = (free_after - free_before) / 1024**3
        print(f"Freed {freed:.1f}GB. Now: {free_after/1024**3:.1f}GB free / {total/1024**3:.1f}GB total")
    print("Cleanup complete.\n")


# ============================================================================
# Word-count enforcement (unchanged from original)
# ============================================================================
WORD_COUNT_RULES = {
    "compromise": {
        "option_c": (20, 30),
        "tradeoff_rule": (12, 17),
        "compromise_justification": (20, 30),
    },
    "non_compromise": {
        "option_c": (20, 30),
        "reframe_type": (3, 5),
        "reframe_justification": (20, 30),
    },
}


def validate_and_retry_word_count(
    model, tokenizer, original_prompt: str, json_text: str,
    option_type: str, model_name: str, max_new_tokens: int,
    max_retries: int = 2,
) -> str:
    import json as _json

    rules = WORD_COUNT_RULES.get(option_type, {})
    if not rules:
        return json_text

    try:
        s = json_text.find("{")
        e = json_text.rfind("}") + 1
        if s == -1 or e <= s:
            return json_text
        data = _json.loads(json_text[s:e])
    except _json.JSONDecodeError:
        return json_text

    for retry in range(max_retries):
        violations = []
        for field, (min_w, max_w) in rules.items():
            val = str(data.get(field, ""))
            wc = len(val.split())
            if wc < min_w or wc > max_w:
                violations.append((field, wc, min_w, max_w))

        if not violations:
            return _json.dumps(data, ensure_ascii=False)

        feedback_parts = []
        for field, wc, min_w, max_w in violations:
            if wc < min_w:
                feedback_parts.append(
                    f'"{field}" has only {wc} words but requires {min_w}–{max_w} words.'
                )
            else:
                feedback_parts.append(
                    f'"{field}" has {wc} words but must be reduced to {min_w}–{max_w} words.'
                )
        feedback = " ".join(feedback_parts)

        retry_prompt = (
            f"{original_prompt}\n\n"
            f"Your previous answer was too short. {feedback} "
            f"Rewrite the entire JSON. Each field must be WITHIN its word count range — not below the minimum and not above the maximum. "
            f"option_c: 20–30 words (not more). tradeoff_rule: 12–17 words (not more). compromise_justification/reframe_justification: 20–30 words (not more)."
        )

        print(f"  Word count retry {retry+1}/{max_retries}: {feedback}")
        new_resp = generate_text(
            model, tokenizer, retry_prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            model_name=model_name,
        )

        try:
            s2 = new_resp.find("{")
            e2 = new_resp.rfind("}") + 1
            if s2 != -1 and e2 > s2:
                data = _json.loads(new_resp[s2:e2])
                json_text = new_resp
        except _json.JSONDecodeError:
            pass

    return json_text


def get_max_new_tokens(model_name: str, task_type: str = "judgment") -> int:
    config = LOCAL_MODEL_REGISTRY.get(model_name, {})
    if task_type == "judgment":
        return config.get("max_new_tokens_judgment", 128)
    return config.get("max_new_tokens_extract", 512)


if __name__ == "__main__":
    print("Available Models:")
    for name, cfg in LOCAL_MODEL_REGISTRY.items():
        print(f"  {name}: {cfg['model_id']} "
              f"(backend={cfg.get('backend', 'transformers')}, "
              f"login={cfg.get('requires_hf_login', False)}, "
              f"thinking={cfg.get('thinking_model', False)})")
