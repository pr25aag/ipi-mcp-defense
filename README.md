# MCP-IPI-Guard — Data Pipeline

Data collection and processing pipeline for the guard-classifier corpus
(dissertation Section 5.2/5.3). Every script here has been run end-to-end;
the numbers below are from the actual last run, not estimates.

> **Status**: an earlier version of this pipeline produced a 95%-attack /
> 5%-benign corpus (1,415 vs 34) -- the wrong direction versus the
> proposal's targets. Fixed by adding `build_benign_corpus_synthetic.py`
> (see section 3.5). The corpus is now **1,407 attack / 7,622 benign**
> (9,029 total) -- both proposal targets met (>=800 attacks, >=5,000
> benign), and the 84%-benign skew is the expected direction (matches
> real production traffic), not a bug. Your proposal (Section 5.4) already
> plans for class-weighted sampling to handle exactly this -- suggested
> weights are printed by `scripts/print_dataset_stats.py` (currently
> benign=0.592, attack=3.209 for inverse-frequency weighting).

---

## 1. Directory layout

```
mcp-ipi-guard/
├── scripts/
│   ├── build_attack_corpus_injecagent.py    # parses InjecAgent benchmark
│   ├── build_attack_corpus_agentdojo.py     # parses AgentDojo GOAL strings
│   ├── build_attack_corpus_handcrafted.py   # generates 215 MCP-specific attacks
│   ├── build_benign_corpus_github.py        # pulls real benign README text
│   └── merge_and_split.py                   # merges, dedups, splits, builds LOO folds
├── attacks/
│   └── processed/           # one .jsonl per attack source (output of the build_* scripts)
├── benign_corpus/
│   └── raw/                 # one .jsonl per benign source
├── experiments/
│   └── data/
│       ├── train.jsonl / val.jsonl / test.jsonl   # final stratified split
│       └── loo/              # leave-one-attack-category-out folds (Section 5.6)
└── README.md                 # this file
```

Every script is independently re-runnable and writes to its own output
path -- you can add a new source shard at any time and just re-run
`merge_and_split.py` to fold it in.

---

## 2. Unified schema

Regardless of source, every record in every `.jsonl` file has the same shape:

| field | type | meaning |
|---|---|---|
| `id` | str | SHA1 hash (12 hex chars) of the full record text, used for dedup |
| `text` | str | the actual tool-output / document / resource text the guard would see |
| `label` | 0 or 1 | 0 = benign, 1 = contains an indirect prompt injection |
| `attack_category` | str | fine-grained category (`"none"` for benign) -- this is the LOO grouping key |
| `tool_type` | str | which MCP/agent tool this text is presented as coming from |
| `source` | str | provenance tag, e.g. `injecagent_ds`, `handcrafted_mcp`, `github_readme:torvalds/linux` |

This uniform schema is what lets `merge_and_split.py` combine four
completely different sources (a benchmark JSON, a Python AST, hand-written
templates, a live API) without any source-specific logic downstream.

---

## 3. What each script does and how

### 3.1 build_attack_corpus_injecagent.py
**Input**: a local clone of `github.com/uiuc-kang-lab/InjecAgent`
(`data/test_cases_ds_base.json`, `data/test_cases_dh_base.json`).
**What it does**: each InjecAgent test case already contains a
`Tool Response` field -- real tool-output text with an attacker instruction
embedded in it (e.g. hidden inside a product review) -- and a
`Tool Response Template` with a `<Attacker Instruction>` placeholder.
The script:
1. Emits the `Tool Response` as-is, labelled 1 (injected).
2. Renders the same template with the placeholder stripped out, and
   emits that as a matched benign counterpart, labelled 0.

Pairing benign/attack samples from the same template means the classifier
has to learn "is there an injected instruction here", not "which tool is
this" -- a shortcut it could otherwise exploit.
**Output**: `attacks/processed/injecagent_unified.jsonl` -- 2,108 raw rows
(1,054 attack / 1,054 benign before global dedup; many benign counterparts
collapse to duplicates once merged, see section 5).

### 3.2 build_attack_corpus_agentdojo.py
**Input**: the installed `agentdojo` Python package (`pip install agentdojo`).
**What it does**: AgentDojo doesn't ship raw injected text -- its attacks
are defined as Python classes with a `GOAL` class attribute (the attacker's
objective, e.g. "email the five largest files to X"). The script walks
every `injection_tasks.py` file across all suite versions using Python's
`ast` module (safer than importing untrusted code), extracts each
`GOAL` string, and renders it into three synthetic "carrier" templates
(a document excerpt, a calendar note, a search snippet) so the text has
the shape of a real tool output rather than a bare goal statement.
**Output**: `attacks/processed/agentdojo_unified.jsonl` -- 162 rows, all
label 1, spanning 4 AgentDojo suites (banking, workspace, travel, slack).

### 3.3 build_attack_corpus_handcrafted.py
**Input**: none -- pure generation.
**What it does**: implements the 5 attack categories from proposal
Section 5.2 (instruction override, tool-name confusion, resource-URI
spoofing, chained multi-step, jailbreak escalation) as 2-4 phrasing
templates per category, each parameterised over pools of fake filenames,
email targets, tool names, resource URIs, and trigger phrases, then takes
the combinatorial product and dedups.
**Output**: `attacks/processed/handcrafted_mcp.jsonl` -- 215 rows, all
label 1. Note: these are templated + parameterised, not 215
independently hand-authored attacks -- say this explicitly in your
methodology section (see caveat in section 6).

### 3.4 build_benign_corpus_github.py
**Input**: the unauthenticated GitHub REST API (`api.github.com`).
**What it does**: fetches the raw README of each repo in a hard-coded
seed list, truncates to 4,000 characters, labels 0.
**Output**: `benign_corpus/raw/github_readmes.jsonl` -- currently 8 rows
(2 of 10 seed repos 403'd on unauthenticated rate limits in this
sandbox). This is the weakest, smallest-volume part of the pipeline
right now -- see section 6.

### 3.5 build_benign_corpus_synthetic.py
**Input**: none -- pure generation, same approach as the hand-crafted
attack generator but with a completely separate parameter pool (different
names, companies, projects, topics) so benign and attack samples don't
share surface vocabulary that could let the classifier shortcut.
**What it does**: covers the 5 tool categories from proposal Section 5.1
(file access, web fetch, calendar, messaging, search) that were almost
entirely missing from the benign side before -- calendar events, Slack/Teams
style messages, search-result snippets, Gmail-style email reads, and short
benign document excerpts, each with 2-3 phrasing templates x 10 x 10 x 10
parameter combinations.
**Output**: `benign_corpus/raw/synthetic_benign.jsonl` -- 7,580 unique rows,
all label 0, distributed across `calendar_get_events` (3,000),
`messaging_send` (3,000), `GmailReadEmail` (1,200), `file_system_read` (200),
`search_query` (180). This is the fix for the imbalance problem in section 6.

### 3.6 print_dataset_stats.py
Reusable sanity-check script -- run any time after `merge_and_split.py` to
print label balance, suggested class weights, attack-category counts, and
tool-type coverage before starting a training run.

### 3.7 merge_and_split.py
**What it does**:
1. Loads all four shards above.
2. Dedups globally by `id` (i.e. by full-text hash) -- so identical text
   appearing in two shards, or two rows collapsing to the same rendered
   template, is only kept once.
3. Splits 70/15/15 into train/val/test, stratified by label (so the
   already-small benign class isn't accidentally left out of one split).
4. Builds one leave-one-attack-category-out fold per category: for each
   category, all label=1 rows in that category go to a
   `{category}__test.jsonl`, everything else (all benign + all other
   attack categories) goes to `{category}__train.jsonl`. This directly
   produces the train/test pairs needed for the generalisation study in
   Section 5.6 / RQ2 -- just point your training loop at each fold in turn.

Idempotent and safe to re-run any time you add a new shard file to the
`SHARDS` list at the top of the script.

---

## 4. Final training data (as of the last run)

**Total: 9,029 unique records** -> train 6,319 / val 1,354 / test 1,356

| source | rows | label |
|---|---:|---|
| InjecAgent (data-stealing + direct-harm) | 1,054 | 1 |
| InjecAgent benign counterparts (post-dedup) | 34 | 0 |
| AgentDojo (4 suites) | 162 | 1 |
| Hand-crafted MCP (5 categories) | 215 | 1 |
| GitHub READMEs | 8 | 0 |
| Synthetic benign (calendar/messaging/email/search/file) | 7,580 | 0 |
| **Total unique after global dedup** | **9,029** | 1,407 attack (15.6%) / 7,622 benign (84.4%) |

Suggested class weights for training (inverse frequency, printed by
`print_dataset_stats.py`): benign=0.592, attack=3.209.

Attack category breakdown (15 categories total -- 6 InjecAgent + 4
AgentDojo + 5 hand-crafted):

| category | count | family |
|---|---:|---|
| Others | 255 | InjecAgent |
| Physical Data | 187 | InjecAgent |
| Data Security Harm | 187 | InjecAgent |
| Physical Harm | 170 | InjecAgent |
| Financial Harm | 153 | InjecAgent |
| Financial Data | 102 | InjecAgent |
| instruction_override | 72 | hand-crafted |
| agentdojo_banking | 51 | AgentDojo |
| chained_multistep | 48 | hand-crafted |
| jailbreak_escalation | 48 | hand-crafted |
| agentdojo_workspace | 45 | AgentDojo |
| resource_uri_spoofing | 27 | hand-crafted |
| agentdojo_travel | 27 | AgentDojo |
| tool_name_confusion | 20 | hand-crafted |
| agentdojo_slack | 15 | AgentDojo |

Text length: 15-4,000 characters, average 293 -- comfortably within
DeBERTa-v3's typical 512-token window for the majority of records; the
GitHub README samples are the longest and will get truncated at
tokenization time.

`experiments/data/loo/` contains 30 files (15 categories x train+test).

---

## 5. Things that went wrong (and were fixed) while building this

1. **ID-collision bug**: the first version of the ID hash used only the
   first 50 characters of each text. InjecAgent's product-listing
   templates share long identical prefixes, so 2,108 raw rows collapsed
   to 180 "unique" ones -- a hashing bug, not a real dedup result. Fixed
   by hashing the full text.
2. **The InjecAgent benign side is thin by construction**: after the fix,
   dedup is correct but reveals that stripping the attacker instruction
   out of many templates produces the same handful of generic benign
   strings repeated across hundreds of test cases -- hence only 34 unique
   benign rows survive from 1,054 raw benign-counterpart rows. This is
   the main driver of the 95/5 imbalance flagged at the top.

---

## 6. What's still honest to flag before training

- **Both the hand-crafted attacks (section 3.3) and the synthetic benign
  corpus (section 3.5) are templated + parameterised, not independently
  authored/collected.** This is fine for reaching corpus-size targets
  quickly, but it means a meaningful fraction of the dataset shares
  generator-level structure -- the classifier could in principle learn
  "this matches template family X" rather than the underlying injection
  signal. Say this plainly in your methodology chapter. Mitigations:
  1. For the **test split specifically**, consider hand-authoring (not
     templating) a smaller "gold" subset per attack category, so held-out
     evaluation isn't scored against the same generator family as training.
  2. Add real, non-templated benign text where you can -- more GitHub
     repos via the Search API (`/search/repositories?q=stars:>1000`,
     paginated, with a `GITHUB_TOKEN` -- run on your own machine for the
     5,000/hr authenticated limit), and real web-page / Project Gutenberg
     text (needs a network that reaches general websites, not just
     GitHub/PyPI -- this sandbox can't, your own machine or Colab can).
  3. Track `source` at evaluation time (it's already in the schema) so you
     can report accuracy broken down by "real" vs "synthetic" source as a
     robustness check, not just an aggregate number.
- Re-run `scripts/print_dataset_stats.py` after any corpus change to
  re-check label balance and category coverage before a training run.


# 1. Get InjecAgent's raw data first (the other scripts don't need this cloned,
#    but this one does)
git clone https://github.com/uiuc-kang-lab/InjecAgent.git attacks/raw/InjecAgent

# 2. Install AgentDojo as a package
pip install agentdojo --break-system-packages

# 3. Run the four data-producing scripts — order between these 4 doesn't matter,
#    they don't depend on each other
python3 scripts/build_attack_corpus_injecagent.py
python3 scripts/build_attack_corpus_agentdojo.py
python3 scripts/build_attack_corpus_handcrafted.py
python3 scripts/build_benign_corpus_github.py
python3 scripts/build_benign_corpus_synthetic.py

# 4. THIS ONE MUST RUN LAST — it reads the output of all five scripts above
python3 scripts/merge_and_split.py

# 5. Optional sanity check
python3 scripts/print_dataset_stats.py