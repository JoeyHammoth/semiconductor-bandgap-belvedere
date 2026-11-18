# Security policy

## Scope

The site is static and has no backend, no accounts and no stored personal data. The meaningful risks are:

- a dependency compromise that ships code to visitors,
- a leaked token or secret in the repository or CI logs,
- a service-worker cache poisoning or stale-cache issue,
- the optional local hardware mode being reachable from the public site (it must not be).

## Reporting

Do not open a public issue for a suspected vulnerability. Use GitHub's private vulnerability reporting on this repository, or contact the maintainer through the profile listed on GitHub. Include the affected path, a reproduction, and the impact you believe it has.

Expect an acknowledgement within a few days. This is a personal project, so fixes are best-effort, but confirmed issues that affect visitors will be prioritised over feature work.

## Handling secrets

- `.env` is git-ignored and must never be committed.
- CI uses the repository `GITHUB_TOKEN` with minimal permissions; no long-lived PAT is stored in Actions secrets for deployment.
- If a token is ever exposed, revoke it first and rotate, then clean history; do not rely on a force-push alone.
