# Operations Logging Standard

This project uses a date-stamped run-log system for long-term, high-velocity development and agent handoffs.

## File Layout
- `log/YYYY-MM-DD_run-log.md` - daily operational log
- `log/INDEX.md` - index of all dated logs (newest first)
- `log/RUN_LOG_TEMPLATE.md` - required entry format
- `docs/CLAUDE_HANDOFF_INDEX.md` - canonical onboarding map

## Required Fields Per Log Entry
- Scope
- Files changed
- Validation command and result
- Commit hash and status
- Next steps or blockers

## Naming Rules
- Use ISO date prefix: `YYYY-MM-DD`
- Use lowercase suffix: `_run-log.md`
- No undated operational log files

## Update Checklist (End of Work Session)
1. Add an entry to today’s dated log file.
2. Update `log/INDEX.md` if a new dated file was created.
3. Update `docs/CLAUDE_HANDOFF_INDEX.md` if docs or navigation changed.
4. Include exact validation commands and outcomes.

## Quick Onboarding Checklist (New Agent)
1. Read `docs/CLAUDE_HANDOFF_INDEX.md`.
2. Open `log/INDEX.md` and read the newest dated run log.
3. Review `docs/ROADMAP.md` and `docs/OPEN_QUESTIONS.md`.
4. Confirm current sprint target before coding.
5. Append your own dated run-log entry when you finish.
6. Leave clear next-step or blocker notes for handoff.
