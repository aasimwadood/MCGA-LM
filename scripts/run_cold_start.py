#!/usr/bin/env python3


from __future__ import annotations

from typing import Dict, List

import numpy as np

from _common import banner, base_parser, build_backend, build_personas, load_config, out_dir, provenance_note


def main() -> None:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--checkpoints", type=int, nargs="*", default=[0, 20, 40, 60, 80, 120, 160, 200])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--per-persona", action="store_true",
                        help="fit one model per persona (Sec. 3.7)")
    args = parser.parse_args()
    cfg = load_config(args)
    if args.quick:
        from evaluate import _quick

        cfg = _quick(cfg)
        args.checkpoints = [0, 20, 40, 60, 80]

    import torch

    from mcga_lm.baselines.variants import get_variant
    from mcga_lm.data.dataset import TurnEncoder
    from mcga_lm.eval import metrics as M
    from mcga_lm.eval import runner as R
    from mcga_lm.inference import AdaptiveCommunicator
    from mcga_lm.memory.graph import IntentMemoryGraph
    from mcga_lm.models.gat import graph_tensors
    from mcga_lm.safety.gate import BayesianGate
    from mcga_lm.training.personalise import train
    from mcga_lm.utils import resolve_device, save_json
    from mcga_lm.viz import bootstrap_ci, plot_learning_curve

    device = resolve_device(cfg.training.device)
    personas = build_personas(cfg)
    backend = build_backend(cfg)
    target = out_dir(cfg, "cold_start")

    banner("Cold-start personalisation (paper Sec. 6.2, Fig. 6)")
    vcfg = get_variant("MCGA-LM").apply(cfg)
    model_or_models, report = train(vcfg, personas, backend, epochs=args.epochs, head_epochs=2,
                                    pretrained=args.pretrained, per_persona=args.per_persona)
    encoder = TurnEncoder(cfg.inputs, reduced_sensor_set=cfg.simulation.reduced_sensor_set)

    # RAG-LLM reference line (no personalisation growth).
    rag_cfg = get_variant("RAG-LLM").apply(cfg)
    rag_model, rag_report = train(rag_cfg, personas, backend, epochs=args.epochs, head_epochs=2,
                                  pretrained=args.pretrained, per_persona=args.per_persona)
    rag, _ = R.run_generative_system("RAG-LLM", rag_cfg, rag_model, backend, personas,
                                     cfg.simulation.seeds, taus=rag_report.tau_by_persona, device=device)
    baseline_sact = rag.aggregate["sact"]["mean"]

    curve: Dict[int, List[float]] = {c: [] for c in args.checkpoints}
    with torch.no_grad():
        for persona in personas:
            model = R._model_for(model_or_models, persona.spec.persona_id).eval()
            graph = persona.intake_graph(fraction=0.05, seed=cfg.seed)
            working = IntentMemoryGraph.from_dict(graph.to_dict())
            tau = report.tau_by_persona.get(persona.spec.persona_id, cfg.safety.tau_default)
            communicator = AdaptiveCommunicator(
                model, backend, vcfg, gate=BayesianGate(cfg.safety, tau=tau), device=device
            )
            accepted_count = 0
            rng = np.random.default_rng(persona.seed)
            recent: List = []
            session_id = 0
            while accepted_count <= max(args.checkpoints):
                turns = persona.simulate_session(seed=1000 + session_id)
                session = encoder.encode_session(persona, turns, working, seed=1000 + session_id).to(device)
                z_ctx, _ = model.encode_context(session.batch)
                state = model.cognitive_state(z_ctx)
                query = model.build_query(z_ctx, state)
                node_scores = node_features = None
                if model.gat is not None:
                    f, ei, ew = graph_tensors(working, device)
                    node_features, node_scores = model.attend_graph(f, ei, query, ew)
                for i, turn in enumerate(turns):
                    res = communicator.run_turn(
                        persona, working, turn, query_row=query[i : i + 1],
                        node_scores=node_scores[0:1] if node_scores is not None else None,
                        node_features=node_features[0:1] if node_features is not None else None,
                        rng=rng, update_graph=True,
                    )
                    recent.append(res)
                    if res.accepted:
                        accepted_count += 1
                    if accepted_count in curve and recent:
                        curve[accepted_count].append(M.sact(recent[-10:]))
                    if accepted_count > max(args.checkpoints):
                        break
                session_id += 1
                if session_id > 30:
                    break

    xs, means, los, his = [], [], [], []
    for c in sorted(curve):
        if not curve[c]:
            continue
        m, lo, hi = bootstrap_ci(curve[c], seed=cfg.seed)
        xs.append(c); means.append(m); los.append(lo); his.append(hi)
        print(f"  after {c:4d} accepted utterances: SACT {m:.2f} [{lo:.2f}, {hi:.2f}]")
    print(f"  RAG-LLM reference SACT: {baseline_sact:.2f}")

    crossing = next((x for x, m in zip(xs, means) if m <= baseline_sact), None)
    plot_learning_curve(xs, means, los, his, baseline_sact, target / "fig6_cold_start.png",
                        full_personalisation_at=crossing)
    save_json({"provenance": provenance_note(), "baseline_sact": baseline_sact,
               "curve": [{"n_accepted": x, "sact": m, "ci_low": lo, "ci_high": hi}
                         for x, m, lo, hi in zip(xs, means, los, his)],
               "crosses_baseline_at": crossing},
              target / "cold_start.json")
    print(f"\nwritten to {target}")


if __name__ == "__main__":
    main()
