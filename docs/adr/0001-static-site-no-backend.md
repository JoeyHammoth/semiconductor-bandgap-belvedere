# 0001 — Static site on GitHub Pages, no backend

- Status: accepted
- Date: 2026-11-18

## Context

The project must be free to host and free to operate indefinitely, and it is a portfolio piece that anyone should be able to open without an account. The scientific work is inference over a small dataset, which does not need a server if the sampler runs in the browser.

## Decision

Ship a fully static site on GitHub Pages. No server, database, account system, paid API or remote compute at any stage. All state lives in the URL hash, `localStorage` or IndexedDB. Inference runs in WebAssembly.

## Consequences

- Inference performance is bounded by the visitor's device; diagnostics and fallbacks are mandatory.
- No `SharedArrayBuffer`, because Pages cannot send COOP/COEP headers. Parallelism is several workers with separate WASM instances.
- GitHub Pages requires a public repository and a base path of `/semiconductor-bandgap-belvedere`.
- Datasets and weights must be small plain files in git; LFS is not served by Pages.
- The same build output deploys unchanged to Cloudflare Pages or Netlify if Pages limits ever bite.
