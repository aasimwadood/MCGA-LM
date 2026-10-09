# MCGA-LM

**Official implementation** of *MCGA-LM: Multimodal Context Graph-Augmented
Language Model for Adaptive Intent Reconstruction in Assistive Communication* —
Al-Nefaie, Wadood, Aldhyani, Uddin & Saeed.

This is the source repository referenced in the paper's Data and Code
Availability statement. It contains the preprocessing, persona generation,
training and evaluation pipelines, and the scripts that produce the tables and
figures.

### Reproduction status

**The paper's reported figures have been generated from this
repository, and results are committed here.** Running `scripts/evaluate.py`
with the default configuration will not give you Table 6. Two reasons, both
concrete:

* The default language backend is a weight-free template realiser, as the
  paper's Data and Code Availability statement says. It reproduces the
  interfaces the architecture needs but none of a language model's fluency or
  world knowledge, so every generation-dependent metric differs. The paper's
  LLaMA-3-8B-Instruct configuration is `configs/llm_hf.yaml`, and it has run end to end. It needs a CUDA GPU and gated weights; see **Running
  with LLaMA-3-8B** below.
* The retrieval stage has not been shown to train to the accuracy the paper
  reports. `scripts/evaluate.py --quick` is a **wiring check, not a result** —
  it shrinks the encoder eightfold and trains the retrieval head for a single
  epoch, which drives IHR, SACT, WPM and FAR to chance. Quick output is written
  to `runs/evaluate/quick/` and stamped accordingly so it cannot be mistaken for
  a measurement.


### Defaults follow the paper

Where the paper states a method, that method is the default, including where
the printed version has a known defect. Each alternative is a config switch and
is not in the paper:

| Setting | Default (the paper) | Alternative |
|---|---|---|
| `graph.attention_form` | `paper`: Eq. 7 as printed | `query_gated` |
| `graph.node_scoring` | `incoming`: what Eq. 13 supervises | `query` |
| `safety.decision_rule` | `variance_only`: Sec. 3.6 | `guarded` (adds a confidence floor) |
| `safety.tau_rule` | `smallest`: Sec. 3.6 (ERRATA E-5) | `largest` |
| `safety.mc_dropout_site` | `scoring_head`: Sec. 3.6, Table 5, Sec. 4.10 | `lora`: Sec. 3.6 before v3 (ERRATA E-12) |
| encoder fitting | shared: Sec. 3.7 | `--per-persona` |
| `evaluate.py` statistics | Sec. 4.8: RM-ANOVA, Holm-corrected paired tests, ART + Wilcoxon, KL and bootstrapped ECE, Table 7's markers, Cohen's d_s | `--descriptive-only` |

The language backend is set by the paper in two places. Its Data and Code
Availability statement makes the weight-free template realiser the repository's
default, and Table 2's LLaMA-3-8B-Instruct configuration is `configs/llm_hf.yaml`.

The printed rules have consequences you will see in a run. Eq. 7 as printed
makes attention almost independent of the query. The smallest-τ rule calibrates
every persona to τ = 0.02. A variance-only gate cannot bound FAR when the scoring
head is confidently wrong. 

Three things the paper relies on are not published, so the code can only take
them as inputs: the "generic AAC prompt dataset" for instruction tuning
(`--instruction-data`), the fourth pre-training corpus with transcripts that
stage 1's contrastive term needs (ERRATA E-9), and any aggregation step for the
"federated" LoRA updates, which the paper does not describe. LoRA updates stay
on the machine that holds the model.

---

## Running with LLaMA-3-8B

The paper's LLaMA-3-8B-Instruct configuration (Table 2: 4-bit NF4, LoRA
r=16/α=32) is `configs/llm_hf.yaml`. It is `configs/default.yaml` with the
language backend and `inputs.ling_dim` changed.

```bash
pip install -e ".[llm]"
huggingface-cli login                  # Meta-Llama-3-8B-Instruct is gated
python scripts/pretrain_encoder.py --config configs/llm_hf.yaml --out runs/
python scripts/evaluate.py --config configs/llm_hf.yaml --out runs/ \
       --pretrained runs/pretrain/encoder_pretrained.pt [--instruction-data aac_prompts.jsonl]
```

**LoRA personalisation (Sec. 3.5, 3.6, 4.9).** After τ is calibrated, stage 3
fine-tunes one LoRA adapter per persona on the utterances that persona accepted,
for 3 epochs. Each adapter starts as a copy of the shared one, which
`--instruction-data` (JSON lines of `{"prompt", "response"}`) instruction-tunes
first. Adapters are saved under `runs/train/<variant>/lora_adapters/`. The
cold-start study treats each session as a day, so accepted utterances reach the
adapter through the 24-hour queue of Sec. 3.6. LLM-Only is not personalised,
since Sec. 4.4 describes it as prompted only. These paths are tested on a tiny,
randomly initialised LLaMA, not on LLaMA-3-8B.

**Requirements this path does not negotiate:**

* **A CUDA GPU.** `bitsandbytes` NF4 has no Apple Silicon or CPU backend. The
  backend now fails immediately with that message rather than deep inside a
  kernel. Without CUDA, set `llm.quantisation=none` to load in bf16 (~16 GB of
  VRAM) — but that is no longer the deployed configuration of Table 2, and any
  result from it should say so.
* **Gated weights.** `meta-llama/Meta-Llama-3-8B-Instruct` requires an accepted
  licence on Hugging Face.
* **`inputs.ling_dim = 4096`**, set in `configs/llm_hf.yaml`, to match the model's hidden size.
  Sec. 3.2 embeds dialogue history "using the same tokeniser as the downstream
  LLM", and with this backend `x_ling` and `z_utt` both come from LLaMA itself.
  `check_embedding_dim` enforces the width rather than letting it fail in a matmul.

**Known caveat.** Sec. 3.5 specifies the prompt wrapper `[INST] … [/INST]`
verbatim. That is Llama-2/Mistral syntax; Llama-3-Instruct expects its own
header-token chat template and will read `[INST]` as ordinary text. The code
follows the paper rather than the model, because the prompt format is a stated
part of the method — expect some quality cost, and see `llm/prompt.py` to switch.

---

## What the system does

One binary switch press, one whole utterance. The pipeline, per communicative
turn :

```
 EEG · HRV · EDA          ┐
 gaze · pupil · blink     ├─► Perceiver IO ──► z_ctx  ─┐
 location · noise · time  │    (Eqs. 2–4)              ├─► q_t ─► GAT over the
 last 10 dialogue tokens  ┘                            │         intent graph
                               TFT ──► s_cog ──────────┘         (Eqs. 6–7)
                            (Eq. 5, W = 32)                            │
                                                                       ▼
                                       structured prompt ◄── active sub-graph
                                       [INST] U S T_graph H … [/INST]
                                                   │
                                                   ▼
                                       decode once, then N = 20
                                       MC-Dropout passes re-score it (Eq. 8)
                                                   │
                                    Var < τ ───────┴─────── Var ≥ τ
                                       │                       │
                                 present candidate       Intent Bubbles,
                                 (1 switch press)        one binary scan
```

Switch activations per turn are bounded **architecturally** at
`C_max + J_max = 5` (Algorithm 1 line 24); `tests/test_algorithm1.py` asserts it
holds even when the gate is made degenerate.

## Install

```bash
git clone <this repo> && cd MCGA-LM
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + pytest
pip install -e ".[llm]"          # the paper's LLaMA-3-8B path
pytest                           # 294 tests, ~15 s, no downloads
```

Python ≥ 3.9 and PyTorch ≥ 2.1. The LoRA tests need the `[llm]` extras and skip
without them. The commands below use the default configuration, which needs no
GPU, no model weights and no dataset downloads; every evaluation runs on personas
generated at run time. Add `--config configs/llm_hf.yaml` to run the
LLaMA-3-8B-Instruct configuration on a CUDA GPU.

## Run it

```bash
# 1. the persona suite : 20 personas, ~380-node graphs
python scripts/make_personas.py --out runs/ --save-graphs

# 2. encoder pre-training, stages 1 and 2  (Sec. 4.9: 100 epochs each; --epochs to shorten)
python scripts/pretrain_encoder.py --out runs/

# 3. train MCGA-LM, calibrate τ per persona, fit per-persona LoRA (LLaMA only)
python scripts/train.py --out runs/ --pretrained runs/pretrain/encoder_pretrained.pt

# 4. see a turn happen
python scripts/demo.py --checkpoint runs/train/MCGA-LM/mcga_lm.pt --turns 5 --show-prompt

# 5. the main experiment  (--descriptive-only skips the Sec. 4.8 tests)
python scripts/evaluate.py --out runs/ --pretrained runs/pretrain/encoder_pretrained.pt

# 6. the rest
python scripts/run_ablations.py     --out runs/
python scripts/run_fatigue_study.py --out runs/
python scripts/run_cold_start.py    --out runs/
python scripts/benchmark_latency.py --out runs/
python scripts/run_graph_growth.py  --out runs/
python scripts/sensitivity_grounding.py
```


Every experiment script also takes `--per-persona`, which fits a separate
encoder, TFT and GAT for each persona. The paper does not describe this: Sec. 3.7
gives the TFT "no per-person fine-tuning for synthetic evaluation", and τ and the
LoRA adapters are per persona either way. It is still the first thing to try if
retrieval looks weak. It costs 20× the training time.


Configuration is YAML plus dotted overrides:

```bash
python scripts/evaluate.py --config configs/reduced_sensors.yaml \
  --set simulation.seeds='[0,1,2]' safety.tau_rule=largest
```

## Repository layout

```
MCGA-LM/
├── README.md         
├── pyproject.toml / requirements.txt / requirements-llm.txt
├── configs/
│   ├── default.yaml             # Table 2, weight-free template backend (CPU)
│   ├── llm_hf.yaml              # the same, with LLaMA-3-8B-Instruct (CUDA)
│   ├── quick.yaml               # smoke-test sizes, CPU
│   ├── reduced_sensors.yaml     #  EDA + PPG + gaze, no EEG
│   ├── ablations/*.yaml         # the five ablations 
│   └── baselines/*.yaml         # LLM-Only, M-LLM, RAG-LLM
├── data/README.md               # DOIs for MAMEM/CLAS/WESAD; nothing bundled
├── runs/                        # outputs; no trained weights or results committed
├── scripts/                     # 12 CLI entry points
├── src/mcga_lm/
│   ├── config.py                # every hyperparameter, cited to the paper
│   ├── pipeline.py              # Phases I–V assembled (Eq. 1)
│   ├── inference.py             # Algorithm 1, line by line
│   ├── losses.py                # Eqs. 9–13
│   ├── states.py                # low/moderate/high fatigue discretisation
│   ├── privacy.py               # sensor toggles, crypto-erase, disclaimer
│   ├── models/                  # perceiver_io, tft, gat, intent_head, layers
│   ├── memory/                  # graph, frames, linearise
│   ├── llm/                     # prompt, template backend, HF backend
│   ├── safety/                  # Bayesian gate + τ calibration
│   ├── training/                # pre-training stages 1-2, stage 3, per-user LoRA
│   ├── data/                    # personas, physiology, corpus, preprocessing
│   ├── baselines/               # 7 baselines + ablation variants
│   ├── eval/                    # metrics, hallucination, stats, cost, runner
│   └── viz/                     # Figs. 2–5 and the Table 5 latency breakdown
└── tests/                       # 294 tests
```


The default backend is a template realiser that fills (pragmatic function, slot)
pairs from surface forms. It is not a language model, and it is not pretending to
be one. What it does reproduce is every *interface* the architecture depends on:
fatigue-modulated temperature, nucleus sampling, conditioning on the retrieved
sub-graph, and token embeddings for the scoring head.

Retrieved
candidates score high, but global-lexicon candidates stay in the pool at a low
prior, so a hot decode can still wander off-graph — retrieval biases generation
without constraining it, as RAG does. That makes the hallucination rate a
*consequence* of the temperature schedule rather than a constant we set. It is
still conditioned on the score ratio we chose (assumption A-30), so
`scripts/sensitivity_grounding.py` prints the entire surface instead of one
flattering point.


## Citation

If you use this code, please cite the paper:

```bibtex
@article{alnefaie2026mcgalm,
  title   = {{MCGA-LM}: Multimodal Context Graph-Augmented Language Model for
             Adaptive Intent Reconstruction in Assistive Communication},
  author  = {Al-Nefaie, Abdullah H. and Wadood, Asim and
             Aldhyani, M. Theyazn H.H. and Uddin, M. Irfan and Saeed, Imran},
  year    = {2026},
  note    = {Code: https://github.com/aasimwadood/MCGA-LM}
}
```

