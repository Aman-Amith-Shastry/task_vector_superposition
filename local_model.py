import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

MODEL_ID  = os.environ.get("TASK_VECTOR_MODEL",    "meta-llama/Llama-3.2-3B-Instruct")
# Set TASK_VECTOR_QUANTIZE=int8 or int4 to load a quantized model (recommended for 8B on MPS)
_QUANTIZE = os.environ.get("TASK_VECTOR_QUANTIZE", "").lower()

_device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

if _QUANTIZE in ("int8", "int4"):
    from transformers import QuantoConfig
    _qcfg = QuantoConfig(weights=_QUANTIZE)
    # quanto quantizes on CPU then moves to device — don't use device_map here
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=_qcfg,
    )
    model.to(_device)
    print(f"Loaded {MODEL_ID} with {_QUANTIZE} quantization")
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"": _device},
    )

print(f"Model device: {model.device}")
model.eval()
n_layers = len(model.model.layers)

_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    return_full_text=False,
)

hf_pipeline = HuggingFacePipeline(pipeline=_pipe)
chat_model = ChatHuggingFace(llm=hf_pipeline, verbose=False)


def get_activation(
    messages: list[dict],
    layer: int = 13,
    add_generation_prompt: bool = True,
) -> torch.Tensor:
    """Residual stream at `layer` at the last token of the formatted sequence.

    Pass add_generation_prompt=True when messages end with a user turn (test
    question) — the hook then fires at the assistant-header token, which
    reflects all ICL context equally rather than being dominated by the last
    ICL answer token.
    The hook is always removed via finally, even if the forward pass errors.
    """
    _buffer: dict[str, torch.Tensor] = {}

    def _hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # hidden: [batch, seq, hidden] or [seq, hidden] depending on transformers version
        _buffer["act"] = (hidden[0, -1, :] if hidden.dim() == 3 else hidden[-1, :]).detach().cpu()

    handle = model.model.layers[layer].register_forward_hook(_hook)
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        inputs = tokenizer(text, return_tensors="pt").to(_device)
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    return _buffer["act"]


def get_activations_all_layers(
    messages: list[dict],
    layers: list[int],
    add_generation_prompt: bool = True,
) -> dict[int, torch.Tensor]:
    """Residual stream at the last token for every layer in `layers`, one forward pass."""
    buffer: dict[int, torch.Tensor] = {}
    handles = []

    for layer in layers:
        def _hook(module, input, output, _l=layer):
            hidden = output[0] if isinstance(output, tuple) else output
            buffer[_l] = (hidden[0, -1, :] if hidden.dim() == 3 else hidden[-1, :]).detach().cpu()
        handles.append(model.model.layers[layer].register_forward_hook(_hook))

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        inputs = tokenizer(text, return_tensors="pt").to(_device)
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    return buffer


def predict_mcq_with_injection(
    messages: list[dict],
    task_vector: torch.Tensor,
    layer: int,
    alpha: float = 2.0,
) -> tuple[str, dict[str, float]]:
    """predict_mcq with `task_vector` additively injected at `layer`.

    The hook fires at `layer`, adds alpha * task_vector to the last-token
    hidden state, and the modified representation propagates through all
    subsequent layers. Equivalent to ELICIT's h̃_l = h_l + α·θ_l.
    """
    tv = task_vector.to(_device)

    def _hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.dim() == 3:
            hidden[:, -1, :] = hidden[:, -1, :] + alpha * tv
        else:
            hidden[-1, :] = hidden[-1, :] + alpha * tv
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    handle = model.model.layers[layer].register_forward_hook(_hook)
    try:
        result = predict_mcq(messages)
    finally:
        handle.remove()
    return result


def predict_constrained(
    messages: list[dict],
    candidates: list[str],
) -> str:
    """Return the highest-probability candidate at the next-token position.

    Each candidate is checked both bare ("yes") and space-prefixed (" yes").
    Only single-token candidates are scored; multi-token strings are skipped.
    """
    text   = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(_device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    probs  = torch.softmax(logits, dim=-1)

    scores: dict[str, float] = {}
    for cand in candidates:
        best = 0.0
        for variant in (cand, f" {cand}"):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                best = max(best, probs[ids[0]].item())
        scores[cand] = best
    return max(scores, key=scores.get)


def predict_constrained_with_injection(
    messages: list[dict],
    candidates: list[str],
    task_vector: torch.Tensor,
    layer: int,
    alpha: float = 2.0,
) -> str:
    """predict_constrained with `task_vector` injected at `layer`."""
    tv = task_vector.to(_device)

    def _hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.dim() == 3:
            hidden[:, -1, :] = hidden[:, -1, :] + alpha * tv
        else:
            hidden[-1, :] = hidden[-1, :] + alpha * tv
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    handle = model.model.layers[layer].register_forward_hook(_hook)
    try:
        result = predict_constrained(messages, candidates)
    finally:
        handle.remove()
    return result


def predict_mcq(
    messages: list[dict],
) -> tuple[str, dict[str, float]]:
    """Predict an MCQ answer in a single forward pass.

    Reads the logits at the last token position and finds the probability of
    each answer letter. Both the bare token ("A") and the space-prefixed token
    (" A") are checked; the max is taken so the function is robust to whether
    the chat template leaves a trailing space before the answer slot.

    Returns (predicted_letter, {letter: normalised_probability}).
    Probabilities are normalised over {A, B, C, D} so they sum to 1 and can
    be interpreted as the model's relative confidence across answer choices.
    """
    text   = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(_device)

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]   # [vocab_size]

    raw_probs = torch.softmax(logits, dim=-1)

    letter_scores: dict[str, float] = {}
    for letter in "ABCD":
        best = 0.0
        for candidate in (letter, f" {letter}"):
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) == 1:
                best = max(best, raw_probs[ids[0]].item())
        letter_scores[letter] = best

    total = sum(letter_scores.values()) or 1.0
    probs = {letter: v / total for letter, v in letter_scores.items()}
    return max(probs, key=probs.get), probs


def classify_output_format(
    messages: list[dict],
    categories: dict[str, list[str]],
) -> str:
    """Classify which output format the model would use via first-token logits.

    One forward pass; no generation needed. Each category is scored by the
    maximum softmax probability among its candidate tokens (both bare and
    space-prefixed forms are checked). Returns the category label whose
    candidates collectively have the highest max probability.
    """
    text   = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(_device)

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]   # [vocab_size]

    probs = torch.softmax(logits, dim=-1)

    def _max_prob(words: list[str]) -> float:
        best = 0.0
        for word in words:
            for candidate in (word, f" {word}"):
                ids = tokenizer.encode(candidate, add_special_tokens=False)
                if len(ids) == 1:
                    best = max(best, probs[ids[0]].item())
        return best

    scores = {label: _max_prob(words) for label, words in categories.items()}
    return max(scores, key=scores.get)


def generate_response(messages: list[dict], max_new_tokens: int = 20) -> str:
    """Greedy decode up to max_new_tokens after the given messages.

    Deterministic (do_sample=False) so outputs are reproducible across
    conditions. Short generation is enough to capture the format token(s)
    needed for behavioral superposition classification.
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(_device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def generate_response_with_injection(
    messages: list[dict],
    task_vector: torch.Tensor,
    layer: int,
    alpha: float = 2.0,
    max_new_tokens: int = 10,
) -> str:
    """generate_response with `task_vector` injected at `layer` on the first pass only.

    The hook fires once — at the initial prefill step where the last token is
    the assistant header. Subsequent autoregressive steps are unmodified so the
    injection steers the first generated token without corrupting the rest of
    the decoding loop.
    """
    tv    = task_vector.to(_device)
    fired = [False]

    def _hook(module, input, output):
        if not fired[0]:
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.dim() == 3:
                hidden[:, -1, :] = hidden[:, -1, :] + alpha * tv
            else:
                hidden[-1, :] = hidden[-1, :] + alpha * tv
            fired[0] = True
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    handle = model.model.layers[layer].register_forward_hook(_hook)
    try:
        result = generate_response(messages, max_new_tokens=max_new_tokens)
    finally:
        handle.remove()
    return result


def score_continuation(messages: list[dict], continuation: str) -> float:
    """Log-probability of generating `continuation` as the next tokens after `messages`.

    Uses teacher-forcing (no sampling) so scores are deterministic and comparable
    across agents. Call math.exp() on the result to get a raw probability.
    """
    context_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    context_ids = tokenizer.encode(context_text, add_special_tokens=False, return_tensors="pt")
    # Leading space matters for BPE: " arrhythmia" ≠ "arrhythmia" as continuation
    cont_ids = tokenizer.encode(" " + continuation, add_special_tokens=False, return_tensors="pt")
    full_ids = torch.cat([context_ids, cont_ids], dim=1).to(_device)

    with torch.no_grad():
        outputs = model(full_ids)

    log_probs = torch.log_softmax(outputs.logits[0], dim=-1)  # [seq_len, vocab]
    n_ctx = context_ids.shape[1]

    total = 0.0
    for i, tid in enumerate(cont_ids[0]):
        total += log_probs[n_ctx - 1 + i, tid].item()

    return total
