"""Quick reproducible stats check on the merged dataset — run any time
after merge_and_split.py to sanity-check label balance and category
coverage before starting a training run."""
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path("experiments/data")

def load(p):
    return [json.loads(l) for l in open(p)]

def main():
    all_recs = []
    for split in ["train", "val", "test"]:
        all_recs += load(DATA_DIR / f"{split}.jsonl")

    labels = Counter(r["label"] for r in all_recs)
    n = len(all_recs)
    print(f"Total records: {n}")
    print(f"  benign (0): {labels[0]} ({100*labels[0]/n:.1f}%)")
    print(f"  attack (1): {labels[1]} ({100*labels[1]/n:.1f}%)")

    w0 = n / (2 * labels[0]) if labels[0] else float("inf")
    w1 = n / (2 * labels[1]) if labels[1] else float("inf")
    print(f"\nSuggested class weights (for cross-entropy class_weight=):")
    print(f"  benign(0): {w0:.3f}  attack(1): {w1:.3f}")

    cats = Counter(r["attack_category"] for r in all_recs if r["label"] == 1)
    print(f"\nAttack categories ({len(cats)}):")
    for cat, c in cats.most_common():
        print(f"  {cat}: {c}")

    tools = Counter(r["tool_type"] for r in all_recs)
    print(f"\nTool type coverage ({len(tools)} types):")
    for t, c in tools.most_common(15):
        print(f"  {t}: {c}")

if __name__ == "__main__":
    main()
