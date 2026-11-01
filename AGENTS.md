# Scope of this repository

- This private repository presents completed train500 π0.5 adaptation experiments. It is not the live server checkout.
- Main cohort: Base/A/B/C/D, 500 train demonstrations, 61,750 frames, 3000 updates per trained group, 500 official Spatial initial-state episodes per evaluated group. E is deferred in this cohort.
- Preserve raw evidence and provenance. Historical train450 results remain separately labeled.
- Presentation commands must not launch training or evaluation. `build_demo_data.py` and `plot_results.py` read recorded results only.
- Keep data, base weights, full checkpoints and environments out of Git. Selected recorded videos/starts/traces are included for the user-authorized demo.
- Source under src/pi05 is an exported experiment snapshot. Explain any future modification separately from the run's source identity.
- Repository stays private unless the user explicitly changes that preference.
- Do not claim unseen-task generalization, real-robot deployment, full-model fine-tuning for C, or controlled causal module effects.
