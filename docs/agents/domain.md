# Domain docs

Layout: **single-context**.

- `GLOSSARY.md` at the repo root is the canonical vocabulary. Use its terms in code, issues, PRs and public copy.
- Architectural decisions live in `docs/adr/` as numbered, immutable records. Read the ADRs that touch an area before changing it; supersede rather than rewrite.
- The normative model is `docs/model.md`; validation claims and tolerances are `docs/validation.md`; deployment and boundaries are `docs/architecture.md`.
- `source/` holds immutable input material (report, handout, design document). Do not edit it; extract into `docs/` instead.

## Consumer rules

1. Read `GLOSSARY.md` before naming a new concept.
2. Read relevant ADRs before changing a boundary (worker protocol, sampler choice, payload budget, no-backend rule).
3. If a change contradicts an ADR, add a superseding ADR instead of silently diverging.
4. If a term is missing from the glossary and will appear in a scene label, add it in the same PR.
