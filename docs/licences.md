# Licence register

Track every third-party asset and dependency that ships to the browser. Record the licence, the version, and whether attribution is required. Prefer OFL, MIT, Apache-2.0 and BSD.

## 1. Project licence

Source code and original written material: MIT (see [`LICENSE`](../LICENSE)). The recorded dataset and the source report are **not** covered by the MIT grant; their publication terms are tracked in [`data.md`](data.md) and must be resolved before the site uses real data publicly.

## 2. Fonts

| Asset | Licence | Attribution | Notes |
|---|---|---|---|
| Instrument Serif | SIL OFL 1.1 | not required in UI, retain licence file | display face |
| Hanken Grotesk | SIL OFL 1.1 | not required in UI, retain licence file | UI/body |
| JetBrains Mono | SIL OFL 1.1 | not required in UI, retain licence file | data and equations |

Self-hosted through Fontsource, Latin subset, `woff2`, `font-display: swap`. Keep the upstream licence files in `web/static/fonts/licenses/`.

## 3. Libraries

| Library | Licence | Ships to browser | Notes |
|---|---|---|---|
| Svelte / SvelteKit | MIT | yes (compiled) | |
| three.js | MIT | yes | tree-shaken, lazy |
| D3 (scales, axes, shapes) | ISC | yes | only the modules used |
| KaTeX | MIT | yes, lazy | fonts subset |
| fflate | MIT | yes | permalink compression |
| wasm-bindgen / serde-wasm-bindgen | MIT / Apache-2.0 | yes (glue) | |
| Playwright | Apache-2.0 | no | test only |
| Lighthouse CI | Apache-2.0 | no | test only |
| NumPyro, JAX, dynesty, ArviZ, PySR | Apache-2.0 / BSD / MIT | no | offline only |

## 4. Data and figures

| Asset | Status | Action |
|---|---|---|
| Lab report text and Table 1 | coursework, permission pending | do not publish until consent recorded |
| Report figures (`source/Journal_Report/images/`) | coursework, permission pending | provenance only; not shipped |
| Measured runs | coursework, permission pending | simulator mode until resolved |
| CIE 1931 fit (Wyman et al.) | published analytic approximation | cite the paper |
| EPM form factors | published values | cite Cohen & Bergstresser (1966) |
| Varshni / Bose–Einstein parameters | published values | cite originals; verify before use |

## 5. Review rule

A dependency may only be added to `web/` or `bgo-wasm` if its licence appears in this table in the same PR. If a licence is unclear, it does not ship.
