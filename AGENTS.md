# AGENTS.md — robosuite (SOARM101 fork · sim-to-real oracle)

## Research journaling → `lerobot-journals/`

Experiments, evaluations, theory, and notable development in this repo are recorded in the shared
**research-journals hub** — the single source of truth — not in scattered notes.

- **Experiment journal:** `EXPERIMENT_JOURNAL.md` at the repo root (a symlink into
  `lerobot-journals/mujoco-sim2real/`). Add/update an entry — local numbering from `E00` =
  problem/hypothesis, then E01, E02 … — whenever you run an experiment, validate the real EKF against
  sim ground truth, or make a significant change.
- **Engineering/design journal:** `robosuite_private/JOURNAL.md` stays the detailed in-repo design log
  (§7 Running Log = the dated changelog). The hub's experiment journal *cites* it; don't duplicate it.
- **Conventions + the per-project source map:** `lerobot-journals/handoff.md`.
- **Shared cross-project planner:** `WEEKLY_PLAN.md` (also symlinked at the repo root).

This repo is the **MuJoCo ground-truth oracle** for the real SO-ARM101 force-aware pipeline. Keep the
strict one-way import contract: `robosuite_private` must never import `lerobot*`; lerobot imports it.
