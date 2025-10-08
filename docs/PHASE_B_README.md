Phase B: Trust cards and normalized events
Purpose: turn signals into PR ready evidence, make readers consistent
Deliverables:
• Normalized event schema: src rldk io event_schema.py, upgrade ingest and diff to produce Event objects with step, wall_time, metrics, rng, data_slice, model_info, notes
• Cards as first class artifacts: src rldk cards determinism.py drift.py reward.py, save JSON plus tiny PNGs to runs run_id rldk_cards and rldk_reports
• CLI: rldk card determinism runA, rldk card drift runA runB, rldk card reward runA
• Docs: card field reference, stable filenames
Acceptance: identical runs produce identical Determinism Cards, doctored logs trigger a Drift Card with a precise first divergence, Reward Card renders on clean and doctored fixtures
