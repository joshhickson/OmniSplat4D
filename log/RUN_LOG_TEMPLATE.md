# Run Log Template

Use this format for every dated run log entry.

## YYYY-MM-DD - Title (assistant|user|external)
- Scope: 1-3 sentences on what changed and why.
- Files changed:
  - path/to/file.ext - one-line purpose
- Validation:
  - Command: exact command
  - Result: pass/fail summary
- Commit:
  - Hash: short hash or `Not committed`
  - Status: `committed` | `staged` | `not committed`
- Next steps:
  - blockers, deferred work, handoff notes
