# 0015 — Unmodified MIT in LICENSE, carve-out in NOTICE

- Status: accepted
- Date: 2026-11-28

## Context

`LICENSE` held the MIT text followed by a sentence excluding recorded data, the lab report and the source figures. The exclusion is correct and has to stay: the report and its figures are coursework material that is not redistributed, and no dataset is published at all.

The cost was metadata. GitHub's licence detector cannot match a file with extra clauses to a known licence, so the repository sidebar reported **Other**, while the README badge and `CITATION.cff` both said MIT. To a visitor deciding whether the code is usable, "Other" reads as a custom or unclear licence, which is the opposite of the project's honesty rule.

The carve-out is not a restriction on the code. It states what was never part of this repository to license.

## Decision

`LICENSE` is the unmodified MIT text. The carve-out lives in [`NOTICE`](../../NOTICE), linked from the README and from [`docs/licences.md`](../licences.md). `CITATION.cff` keeps `license: MIT`.

Nothing about the grant changes: MIT still covers the source code and the original written material, and nothing that was excluded becomes covered.

## Consequences

- GitHub detects and displays MIT, matching the badge and the citation metadata.
- The exclusions stay binding and one link away; `NOTICE` is the single place to edit if they change.
- Redistributors are asked to ship `NOTICE` with `LICENSE`, which is the ordinary convention for a file-level carve-out.
- `docs/licences.md` remains the per-asset register; `NOTICE` states the categories, not the individual terms.
- Rejected alternative: leaving one file and changing the badge and citation to "Other", which would advertise the code as unlicensed to anyone scanning the repo page.
