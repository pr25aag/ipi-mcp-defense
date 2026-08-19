# Data Enhancement v4 Report (FPR fix)
Run: 2026-08-19T13:06:09.902413+00:00Z
Mode: claude-sonnet-5

## The problem this targets
v3 benchmark: recall 1.0 but FPR 0.37-0.53 on test_hard. Models over-flag benign
referential/indirect text because that style of hard-negative was only in test, not train.

## Main corpus (train/val/test/loo)
- Total unique: 14372 (attack 7753 / benign 6619)
- Train/Val/Test: 10060/2156/2156
- LOO categories: 70

## What was added to TRAINING
- Matched hard negatives (the fix): 1250
- Tricky attack top-up: 1194

## Test sets
- test_hard.jsonl: 1500 (UNCHANGED from v3 — for clean before/after FPR)
- test_hard_v4.jsonl: 800 (NEW, fresh matched-style, never trained on)
- paraphrase_holdout.jsonl: 438 (carried forward)

## All existing data preserved: yes — v3 train/val/test folded into the new split, test_hard and paraphrase_holdout carried forward as-is.
