# Data Enhancement v3 Report
Run: 2026-08-06T12:58:26.007252+00:00Z
Mode: claude-sonnet-5

## Main corpus (train/val/test/loo pool)
- Total unique: 11928  (attack 6559 / benign 5369)
- Train/Val/Test: 8349/1789/1790
- LOO categories: 70

## NEW hard held-out test set (test_hard.jsonl)
- Size: 1500  (attack 800 / benign 700)
- Composition: tricky attacks across subtle/obfuscated/authority/multi_hop tiers + matched indirect-language hard negatives. None of it in training.

## paraphrase_holdout.jsonl
- Carried from v2: 438 records

## Tricky TRAIN attacks added, by tier
- claude_v3_tricky_subtle: 595
- claude_v3_tricky_multi_hop: 592
- claude_v3_tricky_obfuscated: 589
- claude_v3_tricky_authority: 588
