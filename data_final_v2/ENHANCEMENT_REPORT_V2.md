# Data Enhancement v2 Report
Run at: 2026-07-30T05:31:09.939620+00:00Z
Mode: LIVE (claude-sonnet-5)

## Final totals (train+val+test+loo pool)
- Total unique records: 9874
- Attack: 3895 (39.4%)
- Benign: 5979 (60.6%)
- Train/Val/Test: 6911 / 1480 / 1483
- Leave-one-out categories: 30
- Held-out paraphrase test set (NOT in train/val/test/LOO): 438 (219 attack / 219 benign)

## What was added this run
- Hand-crafted attacks (v2): 800
- Hard negatives: 1000
- Paraphrased attacks: 437 total (218 in training pool, 219 held out)
- Long-context examples: 300 (from 50 carriers)

## Records by source
- synthetic_benign_calendar: 1958
- synthetic_benign_messaging: 1500
- synthetic_benign_email: 1016
- claude_v2_hard_negative: 987
- claude_v2_handcrafted: 800
- injecagent_enhanced_data_stealing: 544
- injecagent_data_stealing: 544
- injecagent_enhanced_direct_harm: 510
- injecagent_direct_harm: 510
- claude_v2_long_context_needle: 250
- claude_v2_paraphrase: 218
- handcrafted_mcp: 215
- synthetic_benign_filedoc: 200
- synthetic_benign_search: 180
- agentdojo_goal_synthetic_carrier: 138
- claude_augmented: 118
- claude_v2_long_context_benign: 50
- claude_long_context_needle: 48
- injecagent_direct_harm_benign_counterpart: 17
- injecagent_enhanced_data_stealing_benign_counterpart: 17
- injecagent_data_stealing_benign_counterpart: 17
- injecagent_enhanced_direct_harm_benign_counterpart: 17
- claude_long_context_benign: 12
- github_readme:numpy/numpy: 1
- github_readme:torvalds/linux: 1
- github_readme:pallets/flask: 1
- github_readme:fastapi/fastapi: 1
- github_readme:scikit-learn/scikit-learn: 1
- github_readme:huggingface/transformers: 1
- github_readme:langchain-ai/langchain: 1
- github_readme:django/django: 1
