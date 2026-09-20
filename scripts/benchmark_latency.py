#!/usr/bin/env python3


from __future__ import annotations

import platform
import time
from typing import Dict, List

import numpy as np

from _common import banner, base_parser, build_backend, build_personas, load_config, out_dir

PAPER_TABLE_6_MS = {
    "Preprocessing": 34.0,
    "Perceiver IO": 45.0,
    "TFT": 22.0,
    "GAT retrieval": 12.0,
    "LLM generation": 280.0,
    "MC Dropout (N=20)": 56.0,
    "Post-proc. & UI": 8.0,
}


def main() -> None:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--turns", type=int, default=50)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args)
    if args.quick:
        from evaluate import _quick

        cfg = _quick(cfg)

    import torch

    from mcga_lm.data.dataset import TurnEncoder
    from mcga_lm.models.gat import graph_tensors
    from mcga_lm.pipeline import MCGALM
    from mcga_lm.utils import resolve_device, save_json
    from mcga_lm.eval.cost import PAPER_SHARES, cost_report
    from mcga_lm.viz import plot_latency_breakdown

    device = resolve_device(cfg.training.device)
    personas = build_personas(cfg)
    backend = build_backend(cfg)
    persona = personas[0]
    model = MCGALM(cfg).to(device).eval()
    encoder = TurnEncoder(cfg.inputs)

    banner("Per-turn response-latency benchmark (paper Table 6)")
    print(f"device={device}  platform={platform.platform()}  backend={cfg.llm.backend}")

    turns = persona.simulate_session(seed=0, n_turns=max(args.turns, 8))
    timings: Dict[str, List[float]] = {k: [] for k in
                                       ("Preprocessing", "Perceiver IO", "TFT", "GAT retrieval",
                                        "LLM generation", f"MC Dropout (N={cfg.safety.mc_passes})",
                                        "Post-proc. & UI")}
    features, edge_index, edge_weight = graph_tensors(persona.graph, device)

    with torch.no_grad():
        for i in range(min(args.turns, len(turns))):
            t = time.perf_counter(); session = encoder.encode_session(persona, turns[i : i + 1], persona.graph, seed=i).to(device)
            timings["Preprocessing"].append((time.perf_counter() - t) * 1e3)

            t = time.perf_counter(); z = model.context_encoder(session.batch)
            timings["Perceiver IO"].append((time.perf_counter() - t) * 1e3)

            t = time.perf_counter(); state = model.cognitive_state(z)
            timings["TFT"].append((time.perf_counter() - t) * 1e3)

            query = model.build_query(z, state)
            t = time.perf_counter(); nf, ns = model.attend_graph(features, edge_index, query, edge_weight)
            timings["GAT retrieval"].append((time.perf_counter() - t) * 1e3)

            from mcga_lm.llm.prompt import DialogueTurn, build_prompt
            from mcga_lm.memory.linearise import linearise_subgraph
            from mcga_lm.models.tft import fatigue_level

            ids = model.gat.select_active_subgraph(ns, batch_index=0).node_ids
            prompt = build_prompt(persona.profile, fatigue_level(float(state.fatigue[0])),
                                  linearise_subgraph(persona.graph, ids),
                                  [DialogueTurn(s, x) for s, x in turns[i].history], cfg.llm,
                                  active_nodes=tuple((persona.graph.nodes[j].name,
                                                      persona.graph.nodes[j].type, 0.9) for j in ids))
            rng = np.random.default_rng(i)
            t = time.perf_counter(); cands = backend.generate(prompt, n=3, temperature=1.0, top_p=0.9, rng=rng)
            timings["LLM generation"].append((time.perf_counter() - t) * 1e3)
            if not cands:
                continue

            context = model.scoring_context(query[0:1], nf[0, ids, :])
            tokens = torch.from_numpy(backend.embed_tokens(cands[0].text)).unsqueeze(0).to(device)
            t = time.perf_counter(); model.uncertainty(context, tokens)
            timings[f"MC Dropout (N={cfg.safety.mc_passes})"].append((time.perf_counter() - t) * 1e3)

            t = time.perf_counter(); _ = cands[0].text.strip()
            timings["Post-proc. & UI"].append((time.perf_counter() - t) * 1e3)

    means = {k: float(np.mean(v)) if v else 0.0 for k, v in timings.items()}
    sds = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in timings.items()}
    total = sum(means.values())
    # Table 6 note: "the quoted +/-74 ms is the linear sum of component SDs".
    # That is the paper's convention, not quadrature, and it is deliberately
    # conservative -- the stages are serial, so it is the worst case rather than
    # the independent-errors case. Both are printed so the gap is visible.
    total_sd_linear = sum(sds.values())
    total_sd_quadrature = float(np.sqrt(sum(v**2 for v in sds.values())))
    shares = {k: (v / total if total > 0 else float("nan")) for k, v in means.items()}

    print("\n| Component | measured mean ± SD (ms) | share | paper Table 6 (Jetson) | paper share |")
    print("|---|---|---|---|---|")
    for k in means:
        stage = k.split(" (N=")[0]
        paper = PAPER_TABLE_6_MS.get(stage + (" (N=20)" if k.startswith("MC Dropout") else ""), None)
        paper_s = f"{paper:.0f}" if paper else "-"
        paper_share = PAPER_SHARES.get(stage)
        share_s = "-" if paper_share is None else f"{paper_share * 100:.1f}%"
        print(f"| {k} | {means[k]:.1f} ± {sds[k]:.1f} | {shares[k] * 100:.1f}% | {paper_s} | {share_s} |")
    print(f"| **Total response latency** | **{total:.1f} ± {total_sd_linear:.1f}** | 100% | **457 ± 74** | |")
    print(f"\nTotal SD by the paper's linear-sum convention: ±{total_sd_linear:.1f} ms "
          f"(in quadrature it would be ±{total_sd_quadrature:.1f} ms).")
    print("Sensor sliding window (2000 ms) is a continuous buffer and is excluded from the total.")
    print(f"Conversational-turn-taking budget (~1 s, Table 6): "
          f"{'within' if total < 1000 else 'EXCEEDED'} at {total:.0f} ms.")

    # --- Sec. 4.9: the analytic cost model beside the measurement ---------- #
    banner("Analytic computational cost (paper Sec. 4.9)")
    report = cost_report(cfg=cfg, n_tokens=cfg.llm.max_new_tokens, measured_ms=means)
    print(report.to_markdown())

    target = out_dir(cfg, "latency")
    plot_latency_breakdown(means, target / "fig3_latency.png", errors=sds)
    save_json({"device": str(device), "platform": platform.platform(), "backend": cfg.llm.backend,
               "means_ms": means, "sds_ms": sds, "total_ms": total,
               "total_sd_linear_ms": total_sd_linear, "total_sd_quadrature_ms": total_sd_quadrature,
               "measured_shares": shares,
               "paper_table6_ms": PAPER_TABLE_6_MS, "paper_total_ms": 457.0, "paper_total_sd_ms": 74.0,
               "paper_shares": PAPER_SHARES,
               "sensor_window_ms_excluded": 2000.0,
               "computational_cost_sec_4_9": report.as_dict(),
               "note": "Measured on this machine with the configured backend; not comparable to the paper's Jetson AGX Orin figures."},
              target / "latency.json")
    print(f"\nwritten to {target}")


if __name__ == "__main__":
    main()
