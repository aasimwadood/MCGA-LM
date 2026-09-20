#!/usr/bin/env python3


from __future__ import annotations

import numpy as np

from _common import banner, base_parser, build_backend, build_personas, load_config


def main() -> None:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--persona", type=int, default=0)
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default=None, help="trained model from scripts/train.py")
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args)

    import torch

    from mcga_lm.data.dataset import TurnEncoder
    from mcga_lm.inference import AdaptiveCommunicator
    from mcga_lm.memory.graph import IntentMemoryGraph
    from mcga_lm.memory.linearise import linearise_subgraph
    from mcga_lm.models.gat import graph_tensors
    from mcga_lm.pipeline import MCGALM
    from mcga_lm.safety.gate import BayesianGate
    from mcga_lm.utils import resolve_device

    device = resolve_device(cfg.training.device)
    personas = build_personas(cfg)
    persona = personas[args.persona]
    backend = build_backend(cfg)

    model = MCGALM(cfg).to(device)
    taus = {}
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if state.get("per_persona"):
            # One model per persona (Sec. 3.7); pick this persona's own.
            weights = state["models"].get(persona.spec.persona_id)
            if weights is None:
                raise SystemExit(
                    f"checkpoint has no model for {persona.spec.persona_id}; "
                    f"it holds {sorted(state['models'])}"
                )
            model.load_state_dict(weights)
        else:
            model.load_state_dict(state["model"])
        taus = state.get("tau", {})
    else:
        print("[warning] no --checkpoint given: running an untrained model, so retrieval is random.\n")
    model.eval()

    banner(f"MCGA-LM demo -- persona {persona.spec.persona_id} ({persona.spec.diagnosis})")
    turns = persona.simulate_session(seed=0, n_turns=args.turns)
    working = IntentMemoryGraph.from_dict(persona.graph.to_dict())
    encoder = TurnEncoder(cfg.inputs, reduced_sensor_set=cfg.simulation.reduced_sensor_set)
    session = encoder.encode_session(persona, turns, working, seed=0).to(device)

    with torch.no_grad():
        z = model.encode_context(session.batch)[0]
        state = model.cognitive_state(z)
        query = model.build_query(z, state)
        features, edge_index, edge_weight = graph_tensors(working, device)
        node_features, node_scores = model.attend_graph(features, edge_index, query, edge_weight)

        tau = taus.get(persona.spec.persona_id, cfg.safety.tau_default)
        comm = AdaptiveCommunicator(model, backend, cfg, gate=BayesianGate(cfg.safety, tau=tau), device=device)
        rng = np.random.default_rng(0)
        total_sact = 0
        for i, turn in enumerate(turns):
            ids = model.gat.select_active_subgraph(node_scores, batch_index=i).node_ids
            print(f"\n--- turn {i+1} -----------------------------------------------")
            print(f"context  : {turn.context.location_name}, with {turn.context.partner_name} "
                  f"({turn.context.partner_role}), minute {turn.context.minute:.0f}")
            print(f"fatigue  : index {float(state.fatigue[i]):.2f} -> '{state.level(i)}' "
                  f"(ground truth {turn.context.fatigue:.2f})")
            print(f"G_active : {[working.nodes[j].name for j in ids]}")
            if args.show_prompt:
                print(f"T_graph  : {linearise_subgraph(working, ids)[:200]}")
            res = comm.run_turn(persona, working, turn, query[i : i + 1],
                                node_scores[0:1], node_features[0:1], rng)
            print(f"wanted   : {turn.intent.function} {turn.intent.entities} -> \"{turn.intent.reference}\"")
            for c in res.candidates:
                print(f"candidate: \"{c}\"")
            print(f"Var(y)   : {res.variance:.4f}  (tau = {tau:.3f}) -> "
                  f"{'present' if res.gate_accepted else 'SUPPRESS + clarify'}")
            print(f"outcome  : {'ACCEPTED' if res.accepted else 'abstained'}  "
                  f"SACT={res.sact}  clarifications={res.clarification_rounds}")
            if res.accepted:
                print(f"spoken   : \"{res.utterance}\"")
            total_sact += res.sact
        print(f"\nmean SACT over {len(turns)} turns: {total_sact/len(turns):.2f} "
              f"(architectural bound {cfg.inference.max_sact}, Algorithm 1 line 24)")


if __name__ == "__main__":
    main()
