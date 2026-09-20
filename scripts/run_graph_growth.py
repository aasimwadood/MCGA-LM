#!/usr/bin/env python3


from __future__ import annotations

from typing import Dict, List

import numpy as np

from _common import banner, base_parser, build_personas, load_config, out_dir, provenance_note

# Sec. 3.4 / 4.10: the steady state the paper reports empirically.
PAPER_STEADY_STATE = (200, 500)
PAPER_STORAGE_MB = 1.0
PAPER_LORA_MB = 20.0


def _fit_exponent(x: np.ndarray, y: np.ndarray) -> float:
    """Slope of log |V| against log n -- the exponent of |V| ~ n^alpha.

    ``alpha < 1`` is what "sub-linear in accepted utterances" means, and is the
    quantity Sec. 4.10's first mechanism is asserting.
    """
    mask = (x > 0) & (y > 0)
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


def main() -> None:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--utterances", type=int, default=1500,
                        help="accepted utterances to drive through each graph")
    parser.add_argument("--days", type=float, default=180.0,
                        help="simulated days the utterances are spread over "
                             "(the 30-day half-life only bites over months)")
    parser.add_argument("--personas", type=int, default=5)
    parser.add_argument("--from-intake", action="store_true",
                        help="start from a small intake graph (Sec. 3.7) instead of the "
                             "persona's full ~400-node graph, so growth is observable")
    parser.add_argument("--intake-nodes", type=int, default=25,
                        help="size of the intake-interview graph under --from-intake")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args)
    if args.quick:
        args.utterances, args.personas, args.days = 300, 2, 90.0

    from mcga_lm.memory.graph import IntentMemoryGraph, SemanticFrame

    personas = build_personas(cfg)[: max(1, args.personas)]
    target = out_dir(cfg, "graph_growth")

    banner("Intent-graph growth and steady state (paper Sec. 4.10)")
    print(f"{len(personas)} personas x {args.utterances} accepted utterances "
          f"over {args.days:.0f} simulated days\n")

    per_persona: List[Dict] = []
    curves: Dict[str, List[Dict[str, float]]] = {}

    for persona in personas:
        rng = np.random.default_rng(abs(hash(persona.spec.persona_id)) % (2**32))

        # The entity pool this persona's life actually draws on. Sec. 4.10's
        # saturation argument is a claim about this pool being finite.
        pool = [n.name for n in persona.graph.nodes if n.name != "User"]
        partners = [name for name, _ in persona.spec.people] or ["Partner"]

        if args.from_intake:
            # Sec. 3.7: "the intent graph is initialised from a structured intake
            # interview", i.e. small. Starting from the persona's finished
            # ~400-node graph measures only the steady state, because the graph
            # is already in it -- the growth phase Sec. 4.10 describes has
            # already happened. This mode replays that phase.
            graph = IntentMemoryGraph(
                node_dim=cfg.graph.node_dim,
                half_life_days=cfg.graph.half_life_days,
                prune_threshold=cfg.graph.prune_threshold,
                max_nodes=cfg.graph.max_nodes,
            )
            for name in list(rng.choice(pool, size=min(args.intake_nodes, len(pool)), replace=False)):
                graph.update_from_frame(
                    SemanticFrame(obj=str(name), obj_type="Object", intent_nodes=(str(name),)),
                    day=0.0,
                )
        else:
            graph = persona.graph
        start_nodes = len(graph.nodes)
        # A small tail of genuinely novel entities, so the graph is not a closed
        # world: new people, places and objects do enter a life.
        novel_rate = 0.04

        curve: List[Dict[str, float]] = []
        new_nodes_since_mark = 0
        marks: List[float] = []

        for i in range(1, args.utterances + 1):
            day = args.days * i / args.utterances
            if rng.random() < novel_rate:
                entity = f"novel_{persona.spec.persona_id}_{i}"
            else:
                entity = str(rng.choice(pool))
            before = len(graph.nodes)
            graph.update_from_frame(
                SemanticFrame(
                    predicate="associated_with",
                    obj=entity,
                    obj_type="Object",
                    partner=str(rng.choice(partners)),
                    intent_nodes=(entity,),
                ),
                day=day,
            )
            new_nodes_since_mark += max(0, len(graph.nodes) - before)

            if i % max(1, args.utterances // 30) == 0 or i == args.utterances:
                summary = graph.summary()
                marginal = new_nodes_since_mark / max(1, args.utterances // 30)
                marks.append(marginal)
                new_nodes_since_mark = 0
                curve.append({
                    "utterances": float(i),
                    "day": day,
                    "n_nodes": summary["n_nodes"],
                    "n_edges": summary["n_edges"],
                    "storage_mb": summary["storage_mb"],
                    "marginal_new_nodes_per_utterance": marginal,
                })

        xs = np.array([c["utterances"] for c in curve])
        ys = np.array([c["n_nodes"] for c in curve])
        alpha = _fit_exponent(xs, ys)
        first_third = float(np.mean(marks[: max(1, len(marks) // 3)]))
        last_third = float(np.mean(marks[-max(1, len(marks) // 3):]))
        final = graph.summary()
        # Steady state: the last third of the trajectory should not be trending.
        tail = ys[-max(3, len(ys) // 3):]
        tail_slope = float(np.polyfit(np.arange(tail.size), tail, 1)[0]) if tail.size > 2 else float("nan")

        row = {
            "persona": persona.spec.persona_id,
            "nodes_start": start_nodes,
            "nodes_final": final["n_nodes"],
            "edges_final": final["n_edges"],
            "storage_mb": final["storage_mb"],
            "growth_exponent_alpha": alpha,
            "marginal_new_nodes_first_third": first_third,
            "marginal_new_nodes_last_third": last_third,
            "tail_slope_nodes_per_mark": tail_slope,
            "in_paper_steady_state": bool(PAPER_STEADY_STATE[0] <= final["n_nodes"] <= PAPER_STEADY_STATE[1]),
            "under_1mb": bool(final["storage_mb"] < PAPER_STORAGE_MB),
        }
        per_persona.append(row)
        curves[persona.spec.persona_id] = curve
        print(f"  {row['persona']:>4}: {start_nodes:4d} -> {final['n_nodes']:4d} nodes, "
              f"{final['storage_mb']:.2f} MB, alpha={alpha:.2f}, "
              f"marginal new nodes {first_third:.3f} -> {last_third:.3f} /utterance")

    # ---------------------------------------------------------------- claims #
    alphas = [r["growth_exponent_alpha"] for r in per_persona if np.isfinite(r["growth_exponent_alpha"])]
    finals = [r["nodes_final"] for r in per_persona]
    storages = [r["storage_mb"] for r in per_persona]
    firsts = [r["marginal_new_nodes_first_third"] for r in per_persona]
    lasts = [r["marginal_new_nodes_last_third"] for r in per_persona]

    claims = {
        "sublinear_node_creation": {
            "claim": "Sec. 4.10: node creation is sub-linear in accepted utterances (alpha < 1)",
            "mean_alpha": float(np.mean(alphas)) if alphas else float("nan"),
            "max_alpha": float(np.max(alphas)) if alphas else float("nan"),
            "holds": bool(alphas and max(alphas) < 1.0),
        },
        "marginal_rate_decays": {
            "claim": "Sec. 4.10: the marginal rate of new-node creation decays as coverage improves",
            "mean_first_third": float(np.mean(firsts)),
            "mean_last_third": float(np.mean(lasts)),
            "holds": bool(np.mean(lasts) < np.mean(firsts)),
        },
        "steady_state_200_500": {
            "claim": "Sec. 3.4 / 4.10: the graph converges to 200-500 nodes",
            "final_nodes_min": int(np.min(finals)),
            "final_nodes_max": int(np.max(finals)),
            "holds": bool(all(r["in_paper_steady_state"] for r in per_persona)),
        },
        "storage_under_1mb": {
            "claim": "Sec. 4.10: well under 1 MB per user, against ~20 MB for the LoRA adapter",
            "max_storage_mb": float(np.max(storages)),
            "paper_lora_mb": PAPER_LORA_MB,
            "holds": bool(max(storages) < PAPER_STORAGE_MB),
        },
    }

    banner("Sec. 4.10 claims")
    for name, c in claims.items():
        print(f"  [{'PASS' if c['holds'] else 'FAIL'}] {c['claim']}")

    trivial = not args.from_intake and abs(claims["sublinear_node_creation"]["mean_alpha"]) < 0.05
    if trivial:
        print(
            "\n  NOTE: alpha is ~0 because the persona graph starts at its full\n"
            "  ~400 nodes, already inside the steady-state band. The sub-linearity\n"
            "  claim therefore passes trivially -- no growth phase was observed,\n"
            "  rather than growth being observed and found sub-linear. Re-run with\n"
            "  --from-intake to start from a Sec. 3.7 intake graph and measure the\n"
            "  growth phase itself."
        )
    claims["sublinear_node_creation"]["trivially_satisfied"] = bool(trivial)

    from mcga_lm.utils import save_json

    save_json(
        {
            "provenance": provenance_note(),
            "settings": {"utterances": args.utterances, "days": args.days,
                         "personas": len(personas), "from_intake": args.from_intake,
                         "intake_nodes": args.intake_nodes if args.from_intake else None},
            "per_persona": per_persona,
            "curves": curves,
            "claims": claims,
            "caveat": (
                "Sec. 4.10's own caveat applies and is the important one: these are "
                "properties of a simulation and an analytic decay model, not of "
                "longitudinal use. The persona entity pool is finite by construction, "
                "which is the saturation the sub-linearity claim assumes rather than "
                "evidence for it. Growth and drift over months of real deployment can "
                "only be established by the clinical study of Sec. 6.3."
            ),
        },
        target / "graph_growth.json",
    )

    try:
        from mcga_lm.viz import plot_graph_growth

        plot_graph_growth(curves, target / "graph_growth.png", steady_state=PAPER_STEADY_STATE)
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"  (plot skipped: {exc})")

    print(f"\nwritten to {target}")


if __name__ == "__main__":
    main()
