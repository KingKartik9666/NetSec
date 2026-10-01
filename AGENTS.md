# Project guidance

Read `docs/PROJECT_CONTEXT.md` completely before planning, modifying, or evaluating
this project. It is the persistent research and architecture context; verify its
implementation-specific claims against current source and executed checks.

- Keep our code in the existing project directories and external repositories isolated.
- Preserve the distinction between real HEM, DemoDetector, simplified XAI,
  TemplateProvider, real LLM inference, synthetic data and research results.
- Never silently substitute a demo component while claiming research execution.
- Default execution must remain offline, CPU-only and credential-free.
- Use portable paths and deterministic seeds; do not commit secrets or large data/models.
- After changes, run `python run_demo.py`, all four `--sample` options and
  `python -m pytest evaluation -v` in the project environment.
