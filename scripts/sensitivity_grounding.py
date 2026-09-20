#!/usr/bin/env python3


from __future__ import annotations

import numpy as np

from _common import banner, base_parser, build_personas, load_config


def main() -> None:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--turns", type=int, default=25)
    args = parser.parse_args()
    cfg = load_config(args)

    from mcga_lm.eval.hallucination import HallucinationDetector, summarise
    from mcga_lm.llm.prompt import DialogueTurn, build_prompt
    from mcga_lm.llm.template_backend import TemplateLanguageBackend
    from mcga_lm.memory.linearise import linearise_subgraph

    personas = build_personas(cfg)
    detector = HallucinationDetector()

    banner("Hallucination sensitivity to the grounding-score ratio (A-30)")
    print(f"{'gain':>6} {'prior':>6} {'T':>5} {'grounded%':>10} {'hard%':>7} {'soft%':>7}")
    for gain in (2.0, 4.0, 8.0):
        for prior in (0.1, 0.2, 0.4):
            backend = TemplateLanguageBackend(global_prior=prior, grounded_gain=gain)
            for temperature in (0.5, 1.2):
                verdicts, grounded = [], []
                for pi, persona in enumerate(personas):
                    turns = persona.simulate_session(seed=0)
                    rng = np.random.default_rng(pi)
                    for turn in turns[: args.turns]:
                        ids = [persona.graph.node_id(e) for e in turn.intent.entities
                               if persona.graph.has_node(e)][:5]
                        names = [persona.graph.nodes[i].name for i in ids]
                        prompt = build_prompt(
                            persona.profile, "low", linearise_subgraph(persona.graph, ids),
                            [DialogueTurn(s, t) for s, t in turn.history], cfg.llm,
                            active_nodes=tuple(zip(names,
                                                   [persona.graph.nodes[i].type for i in ids],
                                                   [0.9] * len(ids))),
                        )
                        for c in backend.generate(prompt, n=1, temperature=temperature,
                                                  top_p=cfg.llm.top_p, rng=rng):
                            verdicts.append(detector.classify(c.entities, persona.graph, turn.history,
                                                              [p for p, _ in persona.partners]))
                            grounded.append(c.grounded)
                s = summarise(verdicts)
                print(f"{gain:6.1f} {prior:6.2f} {temperature:5.2f} {np.mean(grounded)*100:10.1f} "
                      f"{s['hard_rate']*100:7.1f} {s['soft_rate']*100:7.1f}")


if __name__ == "__main__":
    main()
