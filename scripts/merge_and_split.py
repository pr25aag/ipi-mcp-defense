"""
Merge all processed shards into one corpus, dedup, and produce:
  - train/val/test split (stratified by label)
  - a separate leave-one-category-out split per attack_category
    (for the generalisation study in Section 5.6 of the proposal)
"""
import json, random
from pathlib import Path
from collections import defaultdict

random.seed(42)
SHARDS = [
    "attacks/processed/injecagent_unified.jsonl",
    "attacks/processed/agentdojo_unified.jsonl",
    "attacks/processed/handcrafted_mcp.jsonl",
    "benign_corpus/raw/github_readmes.jsonl",
    "benign_corpus/raw/synthetic_benign.jsonl",
]
OUT_DIR = Path("experiments/data")

def load_all():
    seen_ids = set()
    records = []
    for shard in SHARDS:
        p = Path(shard)
        if not p.exists():
            continue
        for line in open(p):
            r = json.loads(line)
            if r["id"] in seen_ids:
                continue
            seen_ids.add(r["id"])
            records.append(r)
    return records

def stratified_split(records, ratios=(0.7, 0.15, 0.15)):
    by_label = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)
    train, val, test = [], [], []
    for label, group in by_label.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train += group[:n_train]
        val += group[n_train:n_train + n_val]
        test += group[n_train + n_val:]
    random.shuffle(train); random.shuffle(val); random.shuffle(test)
    return train, val, test

def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def main():
    records = load_all()
    train, val, test = stratified_split(records)
    write_jsonl(train, OUT_DIR / "train.jsonl")
    write_jsonl(val, OUT_DIR / "val.jsonl")
    write_jsonl(test, OUT_DIR / "test.jsonl")

    # leave-one-category-out folds (only over label==1 records; benign always included)
    categories = sorted({r["attack_category"] for r in records if r["label"] == 1})
    benign = [r for r in records if r["label"] == 0]
    for held_out in categories:
        loo_train = [r for r in records if r["label"] == 0 or (r["label"] == 1 and r["attack_category"] != held_out)]
        loo_test_attacks = [r for r in records if r["label"] == 1 and r["attack_category"] == held_out]
        write_jsonl(loo_train, OUT_DIR / "loo" / f"{held_out}__train.jsonl")
        write_jsonl(loo_test_attacks + benign, OUT_DIR / "loo" / f"{held_out}__test.jsonl")

    print(f"Total unique records: {len(records)}")
    print(f"  train={len(train)} val={len(val)} test={len(test)}")
    print(f"Attack categories found ({len(categories)}): {categories}")

if __name__ == "__main__":
    main()
