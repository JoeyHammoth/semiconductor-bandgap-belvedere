# Roadmap

Eight phases, each with acceptance criteria recorded as a GitHub epic. Milestones mirror the phases; the project board tracks status. The MVP is the end of Phase 3 (v0.1.0).

| Phase | Week | Ships | Epic | Milestone |
|---|---|---|---|---|
| 0 — Walking skeleton | 1 | repo, CI/CD, Pages, design tokens, data audit | [#1](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/1) | Phase 0 |
| 1 — Physics, AD, sampler, hero | 2 | `bgo-physics`, `bgo-ad`, MCLMC, S0, S1 | [#2](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/2) | Phase 1 |
| 2 — Evidence, sloppiness, swarm | 3 | nested sampling, Fisher, S2, S3, S4, S10 | [#3](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/3) | Phase 2 |
| 3 — Amortised inference and MVP | 4 | NPE engine, calibration, PWA, v0.1.0 | [#4](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/4) | Phase 3 |
| 4 — Thermal twin | 5 | UKF/RTS, GP discrepancy, DMDc, S6 | [#5](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/5) | Phase 4 |
| 5 — Design and transport | 6 | EIG recommender, S5, S8, conformal | [#6](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/6) | Phase 5 |
| 6 — Crystal to colour | 7 | EPM, DOS, S7, Bose–Einstein evidence | [#7](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/7) | Phase 6 |
| 7 — Polish and stretch | 8 | tour, share card, docs, KPM / constellation | [#8](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/8) | Phase 7 |

## Phase 0 tasks

| Task | Issue |
|---|---|
| Repo bootstrap: workspace, docs, CI/CD, Pages | [#9](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/9) |
| Data audit: five diode V–T datasets | [#10](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/10) |
| Physics audit: `E_g(0)`, Varshni/Bose–Einstein, `R_set` | [#11](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/11) |
| WASM walking skeleton and worker protocol | [#12](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/12) |
| SvelteKit instrument shell | [#13](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/13) |
| Quality gates: axe, Lighthouse, budgets | [#14](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/14) |
| Reproduce report Table 1 | [#15](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/15) |

## Sequencing rules

- The physics audit gates public copy, not scaffolding.
- The data audit and the Table 1 regression gate every claim about `E_g`.
- Phase 4 can start before Phase 3 finishes, but the MVP (Phase 3) is what gets released first.
- Stretch items (KPM alloy, symbolic regression) are droppable without affecting the story.

## What "done" means for a phase

1. Every box in the epic's acceptance criteria is checked and the evidence is linked.
2. `docs/validation.md` rows that the phase owns are marked **verified**, not **pending**.
3. The relevant scene has an accessible equivalent and a reduced-motion state.
4. Budgets still pass.
