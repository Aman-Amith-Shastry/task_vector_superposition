import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

_device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map={"": _device},  # load directly onto MPS, no intermediate CPU copy
)

print(f"Model device: {model.device}")
model.eval()

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
