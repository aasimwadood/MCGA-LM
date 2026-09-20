#!/usr/bin/env python3


from __future__ import annotations

import json

from _common import banner, base_parser, build_personas, load_config, out_dir


def main() -> None:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--save-graphs", action="store_true", help="write each persona graph to JSON")
    args = parser.parse_args()
    cfg = load_config(args)

    banner("Building the synthetic persona suite (paper Sec. 4.1)")
    personas = build_personas(cfg)
    target = out_dir(cfg, "personas")

    summary = []
    for persona in personas:
        info = {
            "persona_id": persona.spec.persona_id,
            "cohort": persona.spec.cohort,
            "diagnosis": persona.spec.diagnosis,
            "age": persona.spec.age,
            "n_people": len(persona.spec.people),
            "routines": len(persona.routines),
            **persona.graph.summary(),
        }
        summary.append(info)
        if args.save_graphs:
            persona.graph.save(target / f"{persona.spec.persona_id}.json")

    with open(target / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    sizes = [s["n_nodes"] for s in summary]
    print(f"personas: {len(personas)}")
    print(f"cohorts:  {cfg.simulation.n_als} ALS / {cfg.simulation.n_cerebral_palsy} CP / "
          f"{cfg.simulation.n_brainstem_stroke} brainstem stroke  (Sec. 4.1)")
    print(f"graph size: mean {sum(sizes)/len(sizes):.0f} nodes, range [{min(sizes)}, {max(sizes)}] "
          f"(paper: ~400 nodes per persona, 200-500 steady state)")
    print(f"storage per user: {max(s['storage_mb'] for s in summary):.2f} MB "
          f"(paper Sec. 4.10: well under 1 MB)")
    print(f"written to {target}")


if __name__ == "__main__":
    main()
