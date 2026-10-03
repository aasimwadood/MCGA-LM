"""The paper's language model: LLaMA-3-8B-Instruct, 4-bit NF4, LoRA (Table 3).

Sec. 3.5 / Table 3: r = 16, alpha = 32 on the attention matrices and feed-forward
gates, quantised to 4-bit NF4 for on-device inference.

Requires a CUDA GPU -- bitsandbytes NF4 has no Apple Silicon or CPU backend -- and
gated weights from Hugging Face. See "Running with LLaMA-3-8B" in README.md.

``classify_function`` has no counterpart in the paper. The prompt of Sec. 3.5
asks for a bare utterance, not a labelled intent, but both the simulated user's
acceptance rule and IHR@K compare against the pragmatic function, so it has to be
recovered from the generated text. It is a measurement instrument, not part of
the architecture.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..config import LLMConfig
from .base import GeneratedUtterance
from .prompt import StructuredPrompt


class HFLanguageBackend:
    """4-bit quantised causal LM with LoRA adapters (Table 3: r = 16, alpha = 32)."""

    def __init__(self, cfg: LLMConfig, device: str = "cuda", attach_lora: bool = True) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "the 'hf' backend needs transformers/peft/accelerate: pip install -e '.[llm]'"
            ) from exc

        self.cfg = cfg
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = None
        if cfg.quantisation == "nf4":  # Table 3
            # bitsandbytes NF4 is CUDA-only. On Apple Silicon or CPU the import
            # succeeds and the failure surfaces much later as an opaque kernel
            # error, so check here and say what is actually wrong.
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "quantisation='nf4' (Table 3) needs a CUDA GPU: bitsandbytes has no "
                    "Apple Silicon or CPU backend. Either run on CUDA, or set "
                    "llm.quantisation='none' to load in bf16 -- which needs ~16 GB and is "
                    "no longer the paper's deployed configuration, so say so in any result."
                )
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif cfg.quantisation not in ("none", "", None):
            raise ValueError(
                f"unknown quantisation {cfg.quantisation!r}; expected 'nf4' (Table 3) or 'none'"
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, quantization_config=quant_config, device_map="auto"
        )
        if attach_lora:
            self.model = attach_lora_adapters(self.model, cfg)
        self.model.eval()
        self.embedding_dim = int(self.model.config.hidden_size)

    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: StructuredPrompt,
        n: int = 3,
        temperature: float = 1.0,
        top_p: float = 0.9,
        rng: Optional[np.random.Generator] = None,
    ) -> List[GeneratedUtterance]:
        import torch

        if rng is not None:
            torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
        text = prompt.render()
        # NOTE: prompt.render() emits the [INST] ... [/INST] wrapper that Sec. 3.5
        # specifies verbatim. That is Llama-2/Mistral syntax; Llama-3-Instruct
        # expects its own header-token chat template and will treat [INST] as
        # ordinary text. Following the paper here rather than the model, since
        # the prompt format is a stated part of the method -- but expect this to
        # cost output quality, and see prompt.py if you want the native template.
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                num_return_sequences=n,
                max_new_tokens=self.cfg.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        generated = out.sequences[:, inputs["input_ids"].shape[1] :]
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        # Rank by mean token log-probability rather than by the order the
        # sampler happened to emit. num_return_sequences draws independently, so
        # generation order carries no information and IHR@1 vs IHR@5 would be
        # meaningless without a real score.
        scores = self._sequence_logprobs(out, generated)

        node_types = {name.lower(): ntype for name, ntype, _ in prompt.active_nodes}
        results: List[GeneratedUtterance] = []
        for i, raw in enumerate(decoded):
            utterance = raw.strip().split("\n")[0].strip()
            # Entities are the graph-linked nodes the utterance names -- NOT every
            # token in it. Passing all tokens makes the simulated user accept on
            # any incidental word match and makes every unlisted word look like a
            # hallucinated entity.
            entities = tuple(t for t in _words(utterance) if t in node_types)
            results.append(
                GeneratedUtterance(
                    text=utterance,
                    function=classify_function(utterance, [node_types[e] for e in entities]),
                    source_nodes=entities,
                    entities=entities,
                    score=float(scores[i]) if i < len(scores) else 0.0,
                    grounded=bool(entities),
                )
            )
        return results

    @staticmethod
    def _sequence_logprobs(out, generated) -> List[float]:
        """Mean log-probability of each sampled continuation."""
        import torch

        transition = getattr(out, "scores", None)
        if not transition:
            return [0.0] * generated.shape[0]
        # (steps, batch, vocab) -> per-step log-probs of the tokens actually taken
        stacked = torch.stack(transition, dim=0).log_softmax(dim=-1)
        steps = min(stacked.shape[0], generated.shape[1])
        taken = generated[:, :steps].T.unsqueeze(-1)
        chosen = stacked[:steps].gather(-1, taken).squeeze(-1)  # (steps, batch)
        return chosen.mean(dim=0).float().cpu().tolist()

    def embed_tokens(self, text: str, max_len: int = 24) -> np.ndarray:
        import torch

        ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        ids = {k: v.to(self.model.device) for k, v in ids.items()}
        with torch.no_grad():
            embeddings = self.model.get_input_embeddings()(ids["input_ids"])
        return embeddings[0].float().cpu().numpy()


_STOPWORDS = frozenset(
    "a an the i me my you your it is am are was were be been do does did to of for "
    "and or but if then some any please could would can will shall may might just "
    "now with on in at from that this these those".split()
)


def _words(text: str) -> List[str]:
    return [w.strip(".,!?;:\"'").lower() for w in text.split() if w.strip(".,!?;:\"'")]


def classify_function(utterance: str, node_types: Sequence[str] = ()) -> str:
    """Recover the pragmatic function (Sec. 4.1 taxonomy) from generated text.

    The paper's prompt (Sec. 3.5) asks the LLM for a bare utterance, not a
    labelled intent, so the function has to be recovered afterwards. Both the
    simulated user's acceptance rule and IHR@K compare against it, so leaving it
    empty -- as this backend previously did -- forces IHR to exactly 0% and
    disables the exact-match acceptance path, regardless of how good the model
    is. That is a measurement artefact, not a property of the language model.

    Scored by content-word overlap against each function's surface forms in
    ``TEMPLATES``, restricted where possible to functions whose slot type matches
    a node the utterance actually named. Returns ``""`` when nothing overlaps,
    so an unrecognisable utterance is not silently assigned a function.

    ASSUMPTION: this classifier has no counterpart in the paper. It is a
    measurement instrument, not part of the architecture, and it is deliberately
    conservative -- a wrong label costs an acceptance, an empty one costs a hit.
    """
    from ..data.taxonomy import PRAGMATIC_FUNCTIONS, SLOT_TYPE
    from .template_backend import TEMPLATES

    tokens = set(_words(utterance)) - _STOPWORDS
    if not tokens:
        return ""

    allowed = set(PRAGMATIC_FUNCTIONS)
    if node_types:
        typed = {fn for fn in PRAGMATIC_FUNCTIONS if SLOT_TYPE[fn] in set(node_types)}
        if typed:
            allowed = typed

    best, best_score = "", 0.0
    for fn in PRAGMATIC_FUNCTIONS:
        if fn not in allowed:
            continue
        for form in TEMPLATES[fn]:
            cue = set(_words(form.replace("{slot}", ""))) - _STOPWORDS
            if not cue:
                continue
            overlap = len(cue & tokens) / len(cue)
            if overlap > best_score:
                best, best_score = fn, overlap
    return best if best_score > 0.0 else ""


def attach_lora_adapters(model, cfg: LLMConfig):
    """LoRA per Sec. 3.5 / Table 3: r = 16, alpha = 32, attention + FF gates.

    "Only these adapters (~0.1% of parameters) are updated."
    """
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_targets),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


def build_backend(cfg: LLMConfig, device: str = "cpu", embedding_dim: Optional[int] = None):
    """Backend factory honouring ``LLMConfig.backend``.

    ``embedding_dim`` is ``d_w`` from Sec. 3.2 -- the dialogue history is
    "embedded using the same tokeniser as the downstream LLM", so the backend's
    token embeddings and ``InputDims.ling_dim`` must be the same width. Callers
    pass ``cfg.inputs.ling_dim``; :func:`check_embedding_dim` enforces it.
    """
    if cfg.backend == "template":
        from .template_backend import TemplateLanguageBackend

        return TemplateLanguageBackend(
            embedding_dim=embedding_dim if embedding_dim is not None else 256,
            global_prior=cfg.template_global_prior,
            grounded_gain=cfg.template_grounded_gain,
        )
    if cfg.backend == "hf":
        backend = HFLanguageBackend(cfg, device=device)
        if embedding_dim is not None and backend.embedding_dim != embedding_dim:
            raise ValueError(
                f"inputs.ling_dim is {embedding_dim} but {cfg.model_name} has hidden size "
                f"{backend.embedding_dim}; set inputs.ling_dim to match (configs/llm_hf.yaml)"
            )
        return backend
    raise ValueError(f"unknown LLM backend {cfg.backend!r}; expected 'template' or 'hf'")


def check_embedding_dim(backend, expected: int) -> None:
    """Fail loudly rather than deep inside a matmul if d_w does not line up."""
    actual = getattr(backend, "embedding_dim", None)
    if actual is not None and actual != expected:
        raise ValueError(
            f"language backend emits {actual}-d token embeddings but the intent-scoring head "
            f"expects inputs.ling_dim = {expected} (Sec. 3.2, d_w)"
        )
