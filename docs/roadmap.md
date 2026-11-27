# Roadmap

Eight phases, each with acceptance criteria recorded as a GitHub epic. Milestones mirror the phases; the project board tracks status. The MVP is the end of Phase 3 (v0.1.0).

| Phase | Week | Ships | Epic | Milestone |
|---|---|---|---|---|
| 0 — Walking skeleton | 1 | repo, CI/CD, Pages, design tokens, data provenance audit | [#1](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/1) | Phase 0 |
| 1 — Physics, AD, simulation-first sampler and hero | 2 | `bgo-physics`, `bgo-ad`, synthetic M0–M4 references, MCLMC, S0, S1 | [#2](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/2) | Phase 1 |
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
| Data provenance audit: recordings lost; no dataset published | [#10](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/10) |
| Physics audit: `E_g(0)`, Varshni/Bose–Einstein, `R_set` | [#11](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/11) |
| WASM walking skeleton and worker protocol | [#12](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/12) |
| SvelteKit instrument shell | [#13](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/13) |
| Quality gates: axe, Lighthouse, budgets | [#14](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/14) |
| Audit report Table 1 arithmetic; reproduction unavailable | [#15](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/15) |

## Phase 1 tasks

| Task | Issue |
|---|---|
| Physics core: diode, gap models, Lambert W and colour | [#23](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/23) |
| Automatic differentiation: dual and hyper-dual numbers | [#24](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/24) |
| Simulation and NumPyro reference: synthetic M0–M4 datasets | [#25](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/25) |
| MCLMC and diagnostics: sampler, R-hat, ESS and energy error | [#26](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/26) |
| Infer view and simulation-labelled S0/S1 scenes | [#27](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/27) |

## Phase 4 tasks

| Task | Issue |
|---|---|
| First thermal-twin slice: plate/junction/sensor chain, lag recovery and S6 | [#5](https://github.com/JoeyHammoth/semiconductor-bandgap-belvedere/issues/5) |
| UKF/RTS smoother and Kalman marginal likelihood | planned under #5 |
| State-space GP discrepancy and Hankel-DMDc time constants | planned under #5 |

## Sequencing rules

- The physics audit gates public copy, not scaffolding.
- The physics audit and the report-arithmetic audit gate every public claim about `E_g`. No raw-data claim is made while the original recordings are lost.
- The original logs no longer exist; #10 and #15 were re-scoped under [ADR 0006](adr/0006-no-recorded-dataset.md).
- Phase 1 is simulation-first: synthetic M0–M4 datasets, committed configs and seeds, and NumPyro golden posteriors. Report-anchored cases are labelled synthetic; no real-data comparison is made.
- Phase 4 can start before Phase 3 finishes, but the MVP (Phase 3) is what gets released first.
- Stretch items (KPM alloy, symbolic regression) are droppable without affecting the story.

## What "done" means for a phase

1. Every box in the epic's acceptance criteria is checked and the evidence is linked.
2. `docs/validation.md` rows that the phase owns are marked **verified** or **unavailable**, not **pending**.
3. The relevant scene has an accessible equivalent and a reduced-motion state.
4. Budgets still pass.
