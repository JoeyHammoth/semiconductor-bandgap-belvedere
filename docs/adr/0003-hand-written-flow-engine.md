# 0003 — Hand-written Rust flow engine, no ONNX Runtime Web

- Status: accepted
- Date: 2026-11-18

## Context

Amortised neural posteriors need to run client-side within a strict payload budget. ONNX Runtime Web is several megabytes before any weights, which is hostile to free hosting and mobile. The model is a small conditional rational-quadratic spline flow.

## Decision

Write a minimal inference engine in `bgo-nn`: dense layers, SiLU/ReLU, rational-quadratic spline transforms, inverse sampling and log-probability evaluation, loading quantised weights plus a JSON architecture manifest. Train offline in PyTorch/JAX under `train/`.

## Consequences

- Payload is essentially the weights, targeting under 700 KB gzip.
- The engine must match PyTorch golden vectors on log-probability and a deterministic transform path, or it does not ship.
- We own the numerical care: spline domain handling, monotonicity, quantisation error.
- Adding a new architecture means writing Rust, so architectures are chosen conservatively.
