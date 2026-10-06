"""LoRA personalisation, LLM-side embeddings and two-stage pre-training.

Paper Sec. 3.5 (per-user LoRA on accepted utterances), Sec. 3.6 (the 24-hour
on-device queue; Eq. 8 through the LoRA modules), Sec. 3.2 / Eq. 11 (x_ling and
z_utt from the LLM), Sec. 3.7 (two pre-training stages), Sec. 4.8 (3 epochs per
user).

The HF tests build a tiny randomly initialised LLaMA and a word-level tokeniser
in memory, so they exercise the real peft/transformers code path without
downloading anything.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mcga_lm.training.lora import (
    LoRAUpdateQueue,
    instruction_tune,
    load_instruction_pairs,
    personalise,
)


# ------------------------------------------------------- 24-hour queue -- #
def test_queue_releases_only_pairs_that_have_waited_24_hours() -> None:
    """Sec. 3.6: "after a 24-hour on-device queue, fine-tune the LoRA adapters"."""
    q = LoRAUpdateQueue(hours=24.0)
    q.add("p1", "u1", now_h=0.0)
    q.add("p2", "u2", now_h=10.0)
    assert q.release(now_h=23.9) == []
    assert q.release(now_h=24.0) == [("p1", "u1")]
    assert len(q) == 1
    assert q.release(now_h=34.0) == [("p2", "u2")]
    assert len(q) == 0


def test_template_backend_reports_lora_as_skipped() -> None:
    from mcga_lm.llm.template_backend import TemplateLanguageBackend

    backend = TemplateLanguageBackend(embedding_dim=16)
    out = personalise(backend, [("p", "u")], adapter="P00", epochs=3, lr=1e-4)
    assert "skipped" in out and out["n_pairs"] == 1
    assert "skipped" in instruction_tune(backend, [("p", "u")], epochs=3, lr=1e-4)


def test_instruction_pairs_are_read_from_json_lines(tmp_path) -> None:
    path = tmp_path / "aac.jsonl"
    path.write_text(json.dumps({"prompt": "[INST] hi [/INST]", "response": "hello"}) + "\n\n")
    assert load_instruction_pairs(path) == [("[INST] hi [/INST]", "hello")]
    path.write_text(json.dumps({"prompt": "x"}) + "\n")
    with pytest.raises(ValueError):
        load_instruction_pairs(path)


# --------------------------------------------- x_ling / z_utt via backend -- #
def test_template_backend_reproduces_the_lexeme_encoding_exactly(torch_mod, small_cfg, persona) -> None:
    """Routing x_ling and z_utt through the template backend changes nothing."""
    from mcga_lm.data.dataset import TurnEncoder
    from mcga_lm.llm.template_backend import TemplateLanguageBackend

    turns = persona.simulate_session(seed=0, n_turns=4)
    plain = TurnEncoder(small_cfg.inputs).encode_session(persona, turns, seed=0)
    backend = TemplateLanguageBackend(embedding_dim=small_cfg.inputs.ling_dim)
    routed = TurnEncoder(small_cfg.inputs, backend=backend).encode_session(persona, turns, seed=0)
    assert torch_mod.equal(plain.batch.ling, routed.batch.ling)
    assert torch_mod.allclose(plain.utterance_embeddings, routed.utterance_embeddings)


# ------------------------------------------- Algorithm 1 + the LoRA queue -- #
def test_accepted_utterances_enter_the_lora_queue(torch_mod, small_cfg, persona, monkeypatch) -> None:
    from mcga_lm.data.dataset import TurnEncoder
    from mcga_lm.inference import AdaptiveCommunicator
    from mcga_lm.llm import build_backend
    from mcga_lm.memory.graph import IntentMemoryGraph
    from mcga_lm.models.gat import graph_tensors
    from mcga_lm.pipeline import MCGALM

    torch = torch_mod
    torch.manual_seed(0)
    model = MCGALM(small_cfg).eval()
    backend = build_backend(small_cfg.llm, embedding_dim=small_cfg.inputs.ling_dim)
    graph = IntentMemoryGraph.from_dict(persona.graph.to_dict())
    turns = persona.simulate_session(seed=0, n_turns=2)
    session = TurnEncoder(small_cfg.inputs).encode_session(persona, turns, graph, seed=0)
    with torch.no_grad():
        z, _ = model.encode_context(session.batch)
        query = model.build_query(z, model.cognitive_state(z))
        f, ei, ew = graph_tensors(graph)
        nf, ns = model.attend_graph(f, ei, query, ew)
    monkeypatch.setattr(persona, "accepts", lambda *a, **k: True)
    queue = LoRAUpdateQueue(hours=24.0)
    comm = AdaptiveCommunicator(model, backend, small_cfg, lora_queue=queue)
    comm.gate.tau = 1.0  # present immediately
    res = comm.run_turn(persona, graph, turns[0], query[0:1], ns[0:1], nf[0:1], np.random.default_rng(0), now_h=5.0)
    assert res.accepted and len(queue) == 1
    (prompt, utterance), = queue.release(now_h=29.0)
    assert utterance == res.utterance and prompt.startswith("[INST]")


def test_lora_site_mc_dropout_needs_lora_adapters(torch_mod, small_cfg) -> None:
    from mcga_lm.inference import AdaptiveCommunicator
    from mcga_lm.llm import build_backend
    from mcga_lm.pipeline import MCGALM

    small_cfg.safety.mc_dropout_site = "lora"
    backend = build_backend(small_cfg.llm, embedding_dim=small_cfg.inputs.ling_dim)
    with pytest.raises(ValueError, match="hf"):
        AdaptiveCommunicator(MCGALM(small_cfg), backend, small_cfg)


def test_training_reports_lora_skipped_on_the_template_backend(torch_mod, small_cfg, personas) -> None:
    from mcga_lm.llm import build_backend
    from mcga_lm.training.personalise import train

    small_cfg.training.device = "cpu"
    backend = build_backend(small_cfg.llm, embedding_dim=small_cfg.inputs.ling_dim)
    _, report = train(small_cfg, personas, backend, epochs=1, head_epochs=1)
    assert set(report.lora_by_persona) == {p.spec.persona_id for p in personas}
    assert all("skipped" in r for r in report.lora_by_persona.values())


# ------------------------------------------------- two-stage pre-training -- #
def test_pretraining_runs_two_stages_and_stage_two_leaves_the_encoder_alone(torch_mod, small_cfg, monkeypatch) -> None:
    """Sec. 3.7: stage 1 trains Perceiver IO + TFT jointly, stage 2 the TFT."""
    import importlib

    P = importlib.import_module("mcga_lm.training.pretrain")  # the package re-exports the function
    torch = torch_mod
    small_cfg.training.device = "cpu"
    small_cfg.training.batch_size = 16
    snapshots = {}
    original = P._run_stage

    def spy(name, model, *args, **kwargs):
        before = {k: v.clone() for k, v in model.context_encoder.state_dict().items()}
        out = original(name, model, *args, **kwargs)
        after = model.context_encoder.state_dict()
        snapshots[name] = all(torch.equal(before[k], after[k]) for k in before)
        return out

    monkeypatch.setattr(P, "_run_stage", spy)
    report = P.pretrain(small_cfg, data_root="/nonexistent", epochs=1)
    assert report["epochs"] == 1 and report["tft_epochs"] == 1
    assert snapshots == {"stage 1": False, "stage 2": True}
    assert report["contrastive_active"] is False  # no corpus carries transcripts


def test_stage_one_contrastive_term_uses_transcripts(torch_mod, small_cfg, monkeypatch) -> None:
    from dataclasses import replace

    import importlib

    from mcga_lm.llm.template_backend import TemplateLanguageBackend

    P = importlib.import_module("mcga_lm.training.pretrain")
    small_cfg.training.device = "cpu"
    small_cfg.training.batch_size = 16
    records = P.synthetic_pretraining_corpus(small_cfg.inputs, seed=0)[:32]
    records = [replace(r, transcript=f"i would like some {w}") for r, w in zip(records, ["tea", "water"] * 16)]
    monkeypatch.setattr(P, "load_public_corpora", lambda root: records)
    report = P.pretrain(
        small_cfg, epochs=1,
        backend_factory=lambda: TemplateLanguageBackend(embedding_dim=small_cfg.inputs.ling_dim),
    )
    assert report["contrastive_active"] is True
    assert report["history"][0]["contrastive"] > 0.0
    assert report["stage2_history"][0]["contrastive"] == 0.0


# ------------------------------------------------ HF backend, tiny model -- #
WORDS = "inst user profile fatigue state graph history generate a single appropriate next utterance " \
        "that the would say please pass me water tea i need some help with pain nurse".split()


@pytest.fixture
def tiny_hf(torch_mod):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer, models, pre_tokenizers

    from mcga_lm.config import LLMConfig
    from mcga_lm.llm.hf_backend import HFLanguageBackend

    specials = ["[UNK]", "[PAD]", "<s>", "</s>"]
    vocab = {tok: i for i, tok in enumerate(specials + sorted(set(WORDS)))}
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=core, unk_token="[UNK]", pad_token="[PAD]", bos_token="<s>", eos_token="</s>"
    )
    torch_mod.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
        pad_token_id=1, bos_token_id=2, eos_token_id=3,
    )
    model = transformers.LlamaForCausalLM(config)
    cfg = LLMConfig(quantisation="none", lora_dropout=0.3, max_new_tokens=6)
    return HFLanguageBackend(cfg, device="cpu", model=model, tokenizer=tokenizer)


PAIRS = [("[INST] user profile fatigue state [/INST]", "please pass me water")] * 4


def test_per_user_adapters_start_as_copies_of_the_shared_one(tiny_hf) -> None:
    import torch

    tiny_hf.use_adapter("P00", create=True)
    assert tiny_hf.active_adapter == "P00"
    params = dict(tiny_hf.model.named_parameters())
    for name, p in params.items():
        if ".lora_A.P00." in name:
            assert torch.equal(p, params[name.replace(".P00.", ".default.")])
    # Unknown users fall back to the shared adapter when not asked to create.
    assert tiny_hf.use_adapter("P99", create=False) == "default"


def test_fine_tune_trains_only_the_active_adapter_and_lowers_its_loss(tiny_hf) -> None:
    import torch

    tiny_hf.use_adapter("P00", create=True)
    base = {n: p.detach().clone() for n, p in tiny_hf.model.named_parameters() if "lora_" not in n}
    shared = {n: p.detach().clone() for n, p in tiny_hf.model.named_parameters() if ".default." in n}
    history = tiny_hf.fine_tune(PAIRS, epochs=3, lr=1e-2)
    assert len(history) == 3 and history[-1]["loss"] < history[0]["loss"]
    after = dict(tiny_hf.model.named_parameters())
    assert all(torch.equal(base[n], after[n]) for n in base), "the base model must stay frozen"
    assert all(torch.equal(shared[n], after[n]) for n in shared), "only the user's adapter moves"


def test_eq8_lora_variance_is_zero_until_the_adapter_is_trained(tiny_hf) -> None:
    """Fresh adapters have B = 0, so LoRA dropout cannot perturb the output."""
    prompt, utterance = PAIRS[0]
    tiny_hf.use_adapter("P00", create=True, reset=True)
    v0, c0 = tiny_hf.mc_dropout(prompt, utterance, n_passes=8)
    assert v0 == pytest.approx(0.0, abs=1e-12) and 0.0 < c0 < 1.0
    tiny_hf.fine_tune(PAIRS, epochs=3, lr=1e-2)
    v1, _ = tiny_hf.mc_dropout(prompt, utterance, n_passes=8)
    assert v1 > 0.0


def test_llm_embeddings_have_the_models_width(tiny_hf) -> None:
    short = tiny_hf.embed_history([("User", "please")], n_tokens=10)  # 3 tokens: User : please
    assert short.shape == (10, 32)
    assert np.allclose(short[:7], 0.0) and not np.allclose(short[7:], 0.0)  # left-padded
    long = tiny_hf.embed_history([("Nurse", "i need some tea")] * 3, n_tokens=10)
    assert not np.allclose(long, 0.0, atol=0.0) and not np.any(np.all(long == 0.0, axis=1))
    assert tiny_hf.utterance_embedding("please pass me water").shape == (32,)


def test_hf_generate_runs_end_to_end(tiny_hf) -> None:
    from mcga_lm.config import LLMConfig
    from mcga_lm.llm.prompt import UserProfile, build_prompt

    prompt = build_prompt(
        profile=UserProfile(persona_id="P00", diagnosis="ALS", age=60), state_level="low", graph_text="",
        history=[], cfg=LLMConfig(),
    )
    out = tiny_hf.generate(prompt, n=2, temperature=1.0, top_p=0.9, rng=np.random.default_rng(0))
    assert len(out) == 2 and all(isinstance(c.score, float) for c in out)
