## Summary

<!-- The smallest view that makes the change clear: tree, diff, call tree, or one paragraph. -->

## Linked issue

Closes #

## Acceptance criteria

<!-- Copy the boxes this PR satisfies from the issue, then tick them. -->

- [ ] 

## Evidence

- **Before:**
- **After:**

<!-- Test output, screenshots, bundle sizes, or reference numbers. "Compiles" is not evidence for behaviour. -->

## Merge danger

**Door:** two-way

**Blast radius:**

## Checklist

- [ ] `cargo test --workspace` / `npm run check` pass locally, or the limitation is stated above
- [ ] Budgets still hold (`scripts/size-budget.mjs`)
- [ ] New dependencies listed in `docs/licences.md`
- [ ] `docs/validation.md` updated if this makes a claim measurable
- [ ] No server, database, paid API, `SharedArrayBuffer`, or Git LFS introduced
- [ ] Accessible equivalent and reduced-motion state exist for any new visual
