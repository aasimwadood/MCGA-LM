# MCGA-LM

**Official implementation** of *MCGA-LM: Multimodal Context Graph-Augmented
Language Model for Adaptive Intent Reconstruction in Assistive Communication* —
Al-Nefaie, Wadood, Aldhyani, Uddin & Saeed.

This is the source repository referenced in the paper's Data and Code
Availability statement. It contains the preprocessing, persona generation,
training and evaluation pipelines, and the scripts that produce the tables and
figures.

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
pip install -e ".[llm]"          # optional: the paper's LLaMA-3-8B path
pytest                           # 228 tests, ~4 s, no downloads
```

Python ≥ 3.9 and PyTorch ≥ 2.1. No GPU, no model weights and no dataset
downloads are needed for anything below — the default language backend is
weight-free and every evaluation runs on personas generated at run time.

## Run it

```bash
# 1. the persona suite : 20 personas, ~380-node graphs
python scripts/make_personas.py --out runs/ --save-graphs

# 2. encoder pre-training 
python scripts/pretrain_encoder.py --out runs/ --epochs 20

# 3. train MCGA-LM  and calibrate τ per persona 
python scripts/train.py --out runs/ --pretrained runs/pretrain/encoder_pretrained.pt

# 4. see a turn happen
python scripts/demo.py --checkpoint runs/train/MCGA-LM/mcga_lm.pt --turns 5 --show-prompt

# 5. the main experiment
python scripts/evaluate.py --out runs/ --pretrained runs/pretrain/encoder_pretrained.pt

# 6. the rest
python scripts/run_ablations.py     --out runs/   
python scripts/run_fatigue_study.py --out runs/  
python scripts/run_cold_start.py    --out runs/   
python scripts/benchmark_latency.py --out runs/  
python scripts/run_graph_growth.py  --out runs/  
python scripts/sensitivity_grounding.py         
```


Every experiment script also takes `--per-persona`, which fits one model per
persona instead of one shared across all twenty.  MCGA-LM is a personal, on-device system — and it is
the first thing to try if retrieval looks weak. It costs 20× the training time,
which is why the shared model is the default.


Configuration is YAML plus dotted overrides:

```bash
python scripts/evaluate.py --config configs/reduced_sensors.yaml \
  --set simulation.seeds='[0,1,2]' llm.template_grounded_gain=8.0
```

## Repository layout

```
MCGA-LM/
├── README.md
├── pyproject.toml / requirements.txt / requirements-llm.txt
├── configs/
│   ├── default.yaml             # the paper's hyperparameters (Table 3)
│   ├── quick.yaml               # smoke-test sizes
│   ├── reduced_sensors.yaml     #  EDA + PPG + gaze, no EEG
│   ├── llm_hf.yaml              # LLaMA-3-8B path (UNVERIFIED)
│   ├── ablations/*.yaml         # the five ablations 
│   └── baselines/*.yaml         # LLM-Only, M-LLM, RAG-LLM
├── data/README.md               # DOIs for MAMEM/CLAS/WESAD; nothing bundled
├── docs/
│   ├── ASSUMPTIONS.md           # every gap we filled, and the discrepancies
│   ├── PAPER_MAPPING.md         # paper section → file → status
│   ├── RESULTS.md               # measurements taken during development
│   └── STATUS.md                # what was actually executed
├── scripts/                     # 11 CLI entry points
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
│   ├── data/                    # personas, physiology, corpus, preprocessing
│   ├── baselines/               # 7 baselines + ablation variants
│   ├── eval/                    # metrics, hallucination, stats, runner
│   └── viz/                     # Figs. 2–6
└── tests/                       # 228 tests
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

