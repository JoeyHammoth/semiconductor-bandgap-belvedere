# Issue tracker: GitHub

Issues, specs and phase epics for this repo live as GitHub issues. Use the `gh` CLI for all operations.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body-file <file>` with a heredoc or file body.
- **Read an issue**: `gh issue view <number> --json number,title,body,labels,milestone,assignees,comments`.
- **List issues**: `gh issue list --state open --json number,title,labels,milestone --jq '.[] | {number, title, labels: [.labels[].name], milestone: .milestone.title}'`.
- **Sub-issues**: use `gh issue create --parent <parent>` or `gh issue edit <parent> --add-sub-issue <child>`. Without sub-issue support, put `Part of #<parent>` at the top of the child body.
- **Blocking**: prefer native dependencies (`gh issue edit <child> --add-blocked-by <blocker>`); otherwise a `Blocked by: #<n>` line at the top.
- **Comment**: `gh issue comment <number> --body "..."`.
- **Labels**: `gh issue edit <number> --add-label "..."`.
- **Close**: comment with the explanation, then `gh issue close <number>`.

The repo is `JoeyHammoth/semiconductor-bandgap-belvedere`. Never file into another repository.

## Label vocabulary

- `type:*` — epic, feature, bug, docs, chore, research, ci, test, design
- `area:*` — physics, ad, inference, info-geometry, ot, ssm, nn, epm, wasm, web, data, infra
- `phase:phase-0` … `phase:phase-7`
- `tier:t1` (MVP), `tier:t2` (release 1.0), `tier:t3` (stretch)
- `priority:p0/p1/p2`
- Triage: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `blocked`, `wontfix`

## Milestones

One per phase plus `v0.1.0 — MVP`. Every issue must belong to a phase milestone.

## Pull requests as a request surface

**PRs as a request surface: no.** PRs are implementation, not feature requests.

## When a skill says "publish to the issue tracker"

Create a GitHub issue, label it, assign it, and add it to the matching milestone and the project board.
