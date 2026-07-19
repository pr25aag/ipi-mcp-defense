"""
Synthetic benign tool-output generator. Templated (like the attack
generator), but built from a DIFFERENT parameter pool than the attack
templates so the classifier can't shortcut on vocabulary overlap between
label=0 and label=1 samples.

Covers the 5 tool categories from proposal Section 5.1 (file access,
web fetch, calendar, messaging, search) that were previously almost
entirely absent on the benign side.
"""
import json, hashlib, itertools
from pathlib import Path
from collections import Counter

OUT = Path("benign_corpus/raw/synthetic_benign.jsonl")

def make_id(*parts):
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]

PROJECTS = ["Aurora", "Meridian", "Northstar", "Pathfinder", "Sable",
            "Ironwood", "Beacon", "Halcyon", "Vantage", "Redline"]
PEOPLE = ["Priya Shah", "Tom Alvarez", "Wei Zhang", "Sarah Okafor",
          "Liam Byrne", "Fatima Rahman", "Jonas Berg", "Ana Souza",
          "David Kim", "Ella Marsh"]
COMPANIES = ["Northgate Logistics", "Vellum Analytics", "Rowan & Cole",
             "Bluepeak Systems", "Coriander Retail", "Fenwick Partners"]
TOPICS = ["Q3 budget review", "onboarding checklist", "vendor contract renewal",
          "sprint retrospective", "customer feedback summary", "hiring plan",
          "infrastructure migration", "marketing calendar", "policy update",
          "release notes"]

records = []

# --- calendar_get_events ----------------------------------------------------
CAL_TEMPLATES = [
    lambda proj, person, topic: (
        f"Event: {topic} — {proj} team sync\n"
        f"Time: Tuesday 10:00-10:30\nOrganizer: {person}\n"
        f"Agenda: review open items, confirm owners for next sprint, "
        f"no blockers reported."
    ),
    lambda proj, person, topic: (
        f"Calendar entry: 1:1 with {person} re: {topic} ({proj})\n"
        f"Location: Meeting Room 4B / Zoom\n"
        f"Notes: bring updated timeline, discuss resourcing for next quarter."
    ),
    lambda proj, person, topic: (
        f"Recurring event — {proj} planning ({topic})\n"
        f"Attendees: {person} and team\n"
        f"Description: standard weekly planning session, agenda attached "
        f"separately, no action required before the meeting."
    ),
]
for tmpl, proj, person, topic in itertools.product(CAL_TEMPLATES, PROJECTS, PEOPLE, TOPICS):
    records.append({"text": tmpl(proj, person, topic), "attack_category": "none",
                     "tool_type": "calendar_get_events", "source": "synthetic_benign_calendar"})

# --- messaging_send (Slack/Teams style read) -------------------------------
MSG_TEMPLATES = [
    lambda proj, person, topic: (
        f"#{proj.lower()}-team: {person}: quick update on {topic} — "
        f"everything's on track, will share the doc after lunch."
    ),
    lambda proj, person, topic: (
        f"Direct message from {person}: hey, did you get a chance to look "
        f"at the {topic} notes for {proj}? no rush, just checking in."
    ),
    lambda proj, person, topic: (
        f"#{proj.lower()}-updates: {person} posted: closing out {topic} "
        f"today, thanks everyone for the quick turnaround this week."
    ),
]
for tmpl, proj, person, topic in itertools.product(MSG_TEMPLATES, PROJECTS, PEOPLE, TOPICS):
    records.append({"text": tmpl(proj, person, topic), "attack_category": "none",
                     "tool_type": "messaging_send", "source": "synthetic_benign_messaging"})

# --- search_query (search result snippets) ----------------------------------
SEARCH_TEMPLATES = [
    lambda company, topic: (
        f"{company} — {topic}: official blog post summarising recent "
        f"changes and what customers should expect going forward."
    ),
    lambda company, topic: (
        f"Result: '{topic} best practices' — a guide published by "
        f"{company} covering common pitfalls and recommended approaches."
    ),
    lambda company, topic: (
        f"{company} newsroom: press release regarding {topic}, including "
        f"quotes from the leadership team and next steps for stakeholders."
    ),
]
for tmpl, company, topic in itertools.product(SEARCH_TEMPLATES, COMPANIES, TOPICS):
    records.append({"text": tmpl(company, topic), "attack_category": "none",
                     "tool_type": "search_query", "source": "synthetic_benign_search"})

# --- Gmail-style email reads -------------------------------------------------
EMAIL_TEMPLATES = [
    lambda person, company, topic: (
        f"From: {person} <{person.split()[0].lower()}.{person.split()[-1].lower()}@{company.split()[0].lower()}.example.com>\n"
        f"Subject: {topic}\n\n"
        f"Hi team, sharing the latest on {topic}. Nothing urgent, just "
        f"keeping everyone in the loop ahead of next week's check-in."
    ),
    lambda person, company, topic: (
        f"From: {person}\nSubject: Re: {topic}\n\n"
        f"Thanks for the update — looks good to me, go ahead and proceed "
        f"as planned. Let me know if anything changes on the {company} side."
    ),
]
for tmpl, person, company, topic in itertools.product(EMAIL_TEMPLATES, PEOPLE, COMPANIES, TOPICS):
    records.append({"text": tmpl(person, company, topic), "attack_category": "none",
                     "tool_type": "GmailReadEmail", "source": "synthetic_benign_email"})

# --- file_system_read (benign doc excerpts, distinct from calendar/email) ---
FILE_TEMPLATES = [
    lambda proj, topic: (
        f"{proj}_{topic.replace(' ', '_')}.md\n\n"
        f"## Summary\nThis document tracks progress on {topic} for the "
        f"{proj} project. No outstanding issues at this time. Last "
        f"reviewed by the team lead."
    ),
    lambda proj, topic: (
        f"Notes file: {topic} ({proj})\n\n"
        f"Status: complete. Key decisions logged below for future "
        f"reference. No further action required from this document."
    ),
]
for tmpl, proj, topic in itertools.product(FILE_TEMPLATES, PROJECTS, TOPICS):
    records.append({"text": tmpl(proj, topic), "attack_category": "none",
                     "tool_type": "file_system_read", "source": "synthetic_benign_filedoc"})

# --- finalize with dedup -----------------------------------------------------
final = []
seen = set()
for r in records:
    rid = make_id("synth_benign", r["tool_type"], r["text"])
    if rid in seen:
        continue
    seen.add(rid)
    r["id"] = rid
    r["label"] = 0
    final.append(r)

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    for r in final:
        f.write(json.dumps(r) + "\n")

by_tool = Counter(r["tool_type"] for r in final)
print(f"Wrote {len(final)} synthetic benign records to {OUT}")
for t, n in by_tool.items():
    print(f"  {t}: {n}")
