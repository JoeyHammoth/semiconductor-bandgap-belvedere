# 0005 — Worker-parallel chains without SharedArrayBuffer

- Status: accepted
- Date: 2026-11-18

## Context

GitHub Pages cannot set COOP/COEP response headers on the free plan, so `SharedArrayBuffer` and WASM threads are unavailable. Chains still need to run concurrently to keep inference interactive.

## Decision

Run one WASM instance per Web Worker and map chains, nested-sampling replicates and EPM sweeps across the pool. Compile the WASM module once on the main thread and transfer the compiled `WebAssembly.Module` to each worker. Communicate with structured messages and transferable typed arrays. Cancellation terminates and respawns the worker.

## Consequences

- No cross-chain shared state; each worker owns its RNG stream and its state.
- Memory scales with pool size, which is capped at `min(hardwareConcurrency − 1, 4)` and 2 on mobile.
- Progress must be streamed explicitly; there is no shared progress buffer.
- Rejected alternatives: `coi-serviceworker` hacks (fragile, breaks first load) and a hosted backend (violates the no-server constraint).
