# Data Enhancement Report
Run at: 2026-07-30T04:12:20.459233+00:00Z
Mode: LIVE (claude-sonnet-5)

## Final totals
- Total unique records: 10295
- Attack: 2627 (25.5%)
- Benign: 7668 (74.5%)
- Train/Val/Test: 7205 / 1544 / 1546
- Leave-one-out categories: 20

## What was added this run
- InjecAgent enhanced set: 2108 records
- Claude-augmented hand-crafted attacks: 118 records
- Long-context needle-in-haystack examples: 60 records (from 12 carrier documents)

## Records by source
- synthetic_benign_messaging: 3000
- synthetic_benign_calendar: 3000
- synthetic_benign_email: 1200
- injecagent_enhanced_data_stealing: 544
- injecagent_data_stealing: 544
- injecagent_enhanced_direct_harm: 510
- injecagent_direct_harm: 510
- handcrafted_mcp: 215
- synthetic_benign_filedoc: 200
- synthetic_benign_search: 180
- agentdojo_goal_synthetic_carrier: 138
- claude_augmented: 118
- claude_long_context_needle: 48
- injecagent_enhanced_direct_harm_benign_counterpart: 17
- injecagent_data_stealing_benign_counterpart: 17
- injecagent_direct_harm_benign_counterpart: 17
- injecagent_enhanced_data_stealing_benign_counterpart: 17
- claude_long_context_benign: 12
- github_readme:django/django: 1
- github_readme:torvalds/linux: 1
- github_readme:numpy/numpy: 1
- github_readme:pallets/flask: 1
- github_readme:langchain-ai/langchain: 1
- github_readme:fastapi/fastapi: 1
- github_readme:huggingface/transformers: 1
- github_readme:scikit-learn/scikit-learn: 1
