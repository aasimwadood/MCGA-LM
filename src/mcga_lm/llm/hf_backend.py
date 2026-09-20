
from __future__ import annotations

from typing import List, Optional

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
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
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
            )
        generated = out[:, inputs["input_ids"].shape[1] :]
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        known = {name.lower() for name, _, _ in prompt.active_nodes}
        results: List[GeneratedUtterance] = []
        for i, raw in enumerate(decoded):
            utterance = raw.strip().split("\n")[0].strip()
            entities = tuple(t for t in utterance.lower().split() if t in known)
            results.append(
                GeneratedUtterance(
                    text=utterance,
                    function="",  # not recoverable without a classifier
                    source_nodes=entities,
                    entities=tuple(utterance.lower().split()),
                    score=-float(i),
                    grounded=bool(entities),
                )
            )
        return results

    def embed_tokens(self, text: str, max_len: int = 24) -> np.ndarray:
        import torch

        ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        ids = {k: v.to(self.model.device) for k, v in ids.items()}
        with torch.no_grad():
            embeddings = self.model.get_input_embeddings()(ids["input_ids"])
        return embeddings[0].float().cpu().numpy()


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
