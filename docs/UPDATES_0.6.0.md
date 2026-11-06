# What's new in NetSec 0.6.0

A new look, explanations in plain English through a chat, and ping sweeps now reported as recon scans.

- **New interface.** Frosted-glass surfaces in light and dark themes, with a theme switch that sweeps across the window. Background dots respond to the pointer, sections ease in as you scroll, and wheel scrolling glides. The navigation bar has a sliding lens you can drag between pages. The three CSV pages are now one page, **CSV analysis**, and model choices are compact chips. Reduced-motion, high-contrast and reduced-transparency settings are respected.
- **Explanations in a chat.** "Explain in plain English" now opens a chat beside the result. The local language model (Qwen3-4B) answers one verified sentence at a time, and each sentence is labelled as model-written or template. It runs on the GPU when there is enough memory and falls back to the CPU otherwise. On the CPU most sentences fall back to templates. Chat stays on this PC and is cleared when the app closes.
- **Ping sweeps are reported as recon scans.** Every CSV model now adds the ping-sweep probability to recon scan and shows ping sweep as 0%. On the study's validation tuning rows this raised the default model's accuracy from 97.55% to 97.66%. Ping-sweep rows that a model calls benign are not changed. The held-out test set has not been used.
- **Live screening** no longer shows the retired partial-input and binary packet models. No live detection model is installed in this release, and research caveat fields are no longer shown on that page.

Known limits:
- The explainer's planned live red-team, final review and security scan were not run before release.
- Some obfuscated misuse requests can reach the language model. The answer verifier is meant to catch what reaches it, but this has not been checked against the live model.
- The 2.4 GB language-model file is not inside the installer, because Windows installers of this type are limited to about 2 GB. Copy it into the installed app's `resources\runtime\artifacts\llm\` folder. The app checks its hash before first use, and the rest of the app works without it.
- This Windows build remains unsigned.
