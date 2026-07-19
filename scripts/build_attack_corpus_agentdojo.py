"""
Extract attacker GOAL strings from AgentDojo's injection_tasks.py modules
across all default_suites versions, and render each into a small set of
synthetic "tool output" carrier templates (document snippet, calendar
invite body, search result snippet) so they match the shape of real
MCP tool responses. Labelled 1 (injected).
"""
import ast, hashlib, json, glob
from pathlib import Path

AGENTDOJO_ROOT = Path("/usr/local/lib/python3.12/dist-packages/agentdojo/default_suites")
OUT = Path("attacks/processed/agentdojo_unified.jsonl")

CARRIER_TEMPLATES = [
    "Document excerpt (shared drive): ... {goal} ... [end of document]",
    "Calendar event notes: Meeting prep — {goal}",
    "Search result snippet: \"{goal}\" - relevant-looking-domain.com",
]

def make_id(*parts):
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]

def extract_goals(py_file):
    tree = ast.parse(open(py_file).read())
    goals = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if getattr(target, "id", None) == "GOAL":
                            try:
                                val = ast.literal_eval(item.value)
                            except Exception:
                                # f-strings resolve to JoinedStr; fall back to unparse
                                val = ast.unparse(item.value)
                            goals.append(val)
    return goals

def main():
    files = glob.glob(str(AGENTDOJO_ROOT / "**" / "injection_tasks.py"), recursive=True)
    records = []
    for fpath in files:
        suite = Path(fpath).parent.name
        goals = extract_goals(fpath)
        for g in goals:
            g = str(g)
            for template in CARRIER_TEMPLATES:
                text = template.format(goal=g)
                records.append({
                    "id": make_id("agentdojo", suite, text),
                    "text": text,
                    "label": 1,
                    "attack_category": f"agentdojo_{suite}",
                    "tool_type": suite,
                    "source": "agentdojo_goal_synthetic_carrier",
                })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Found injection_tasks.py in {len(files)} suite versions")
    print(f"Wrote {len(records)} records to {OUT}")

if __name__ == "__main__":
    main()
