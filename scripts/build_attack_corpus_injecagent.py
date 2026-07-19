"""
Convert InjecAgent test cases into the unified guard-classifier schema:
    {id, text, label, attack_category, tool_type, source}

label: 1 = injected (contains adversarial instruction), 0 = benign
For every injected sample we also emit a benign counterpart by rendering
the same tool-response template WITHOUT the attacker instruction, so the
classifier sees matched pairs (same tool, same template, different content)
rather than learning a spurious "which tool" shortcut.
"""
import json, re, hashlib
from pathlib import Path

RAW = Path("attacks/raw/InjecAgent/data")
OUT = Path("attacks/processed/injecagent_unified.jsonl")

def make_id(*parts):
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]

def load(path):
    return json.load(open(path))

def process_file(path, ds_or_dh):
    records = load(path)
    out = []
    for r in records:
        tool_type = r.get("User Tool", "unknown")
        category = r.get("Attack Type", ds_or_dh)
        injected_text = r.get("Tool Response", "")
        template = r.get("Tool Response Template", "")

        # injected sample
        out.append({
            "id": make_id(ds_or_dh, tool_type, injected_text),
            "text": injected_text,
            "label": 1,
            "attack_category": category,
            "tool_type": tool_type,
            "source": f"injecagent_{ds_or_dh}",
        })

        # benign counterpart: strip the placeholder / attacker instruction out
        # of the template so the tool schema stays identical but content is clean
        benign_text = re.sub(r"<Attacker Instruction>", "", template)
        benign_text = re.sub(r"\s+", " ", benign_text).strip()
        if benign_text:
            out.append({
                "id": make_id("benign", ds_or_dh, tool_type, benign_text),
                "text": benign_text,
                "label": 0,
                "attack_category": "none",
                "tool_type": tool_type,
                "source": f"injecagent_{ds_or_dh}_benign_counterpart",
            })
    return out

def main():
    all_records = []
    all_records += process_file(RAW / "test_cases_ds_base.json", "data_stealing")
    all_records += process_file(RAW / "test_cases_dh_base.json", "direct_harm")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    n_pos = sum(1 for r in all_records if r["label"] == 1)
    n_neg = sum(1 for r in all_records if r["label"] == 0)
    print(f"Wrote {len(all_records)} records to {OUT}")
    print(f"  injected (label=1): {n_pos}")
    print(f"  benign   (label=0): {n_neg}")

if __name__ == "__main__":
    main()
