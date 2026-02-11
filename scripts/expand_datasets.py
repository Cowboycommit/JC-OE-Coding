#!/usr/bin/env python3
"""
Expand existing seed datasets to 1000 rows each.

Strategy:
- Use existing responses as seeds
- Generate new variations via phrase recombination, synonym substitution,
  and natural linguistic variation
- Preserve thematic distribution and authentic survey-response character
- Maintain segment/topic proportions from originals
"""

import pandas as pd
import random
import re
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

# ─── Variation engine ────────────────────────────────────────────────────────

SYNONYMS = {
    "amazing": ["incredible", "outstanding", "fantastic", "wonderful", "exceptional", "superb"],
    "terrible": ["awful", "dreadful", "horrible", "appalling", "abysmal", "atrocious"],
    "good": ["great", "solid", "decent", "satisfactory", "positive", "pleasant"],
    "bad": ["poor", "disappointing", "subpar", "inadequate", "unsatisfactory", "lacking"],
    "helpful": ["supportive", "accommodating", "attentive", "responsive", "useful", "considerate"],
    "friendly": ["warm", "welcoming", "approachable", "kind", "courteous", "personable"],
    "long": ["extended", "prolonged", "lengthy", "excessive", "unreasonable", "drawn-out"],
    "clean": ["spotless", "immaculate", "pristine", "tidy", "well-maintained", "sanitary"],
    "dirty": ["filthy", "unsanitary", "grimy", "unclean", "messy", "unhygienic"],
    "fast": ["quick", "speedy", "prompt", "swift", "rapid", "efficient"],
    "slow": ["sluggish", "delayed", "unhurried", "tardy", "behind schedule", "lagging"],
    "excellent": ["superb", "outstanding", "top-notch", "first-rate", "exemplary", "stellar"],
    "professional": ["competent", "skilled", "experienced", "capable", "qualified", "adept"],
    "comfortable": ["at ease", "relaxed", "content", "settled", "calm", "reassured"],
    "confusing": ["unclear", "bewildering", "perplexing", "muddled", "hard to follow", "vague"],
    "important": ["crucial", "vital", "essential", "significant", "critical", "key"],
    "happy": ["pleased", "satisfied", "delighted", "content", "glad", "thrilled"],
    "frustrated": ["annoyed", "irritated", "exasperated", "upset", "aggravated", "displeased"],
    "worried": ["concerned", "anxious", "nervous", "uneasy", "troubled", "apprehensive"],
    "exhausted": ["drained", "burnt out", "wiped out", "fatigued", "depleted", "spent"],
    "struggling": ["having difficulty", "finding it hard", "grappling", "dealing with challenges", "coping"],
    "love": ["really enjoy", "appreciate", "value", "am passionate about", "cherish"],
    "hate": ["can't stand", "despise", "strongly dislike", "detest", "loathe"],
    "better": ["improved", "superior", "enhanced", "upgraded", "stronger"],
    "worse": ["deteriorated", "declined", "worsened", "diminished", "degraded"],
    "beautiful": ["gorgeous", "stunning", "lovely", "elegant", "exquisite"],
    "exciting": ["thrilling", "exhilarating", "electrifying", "captivating", "riveting"],
    "boring": ["dull", "tedious", "monotonous", "uninspiring", "lackluster"],
    "difficult": ["challenging", "tough", "demanding", "hard", "arduous"],
    "easy": ["straightforward", "simple", "effortless", "manageable", "smooth"],
}

INTENSIFIERS = ["really", "very", "incredibly", "absolutely", "truly", "genuinely", "seriously"]
HEDGES = ["kind of", "somewhat", "a bit", "slightly", "fairly", "rather", "pretty"]

# Casual typos/style markers for authenticity
TYPO_PAIRS = [
    ("definitely", "definately"), ("received", "recieved"), ("separate", "seperate"),
    ("accommodate", "accomodate"), ("occasionally", "occassionally"),
]

CASUAL_MARKERS = [
    ("!!!", "!!!"), ("???", "???"), ("...", "..."),
    ("&amp;", "&amp;"), ("   ", "   "),  # extra spaces
]

FILLER_STARTERS = [
    "Honestly, ", "To be honest, ", "I gotta say, ", "In my experience, ",
    "From my perspective, ", "I'd say that ", "Looking back, ",
    "Overall, ", "Basically, ", "I mean, ", "Truthfully, ",
    "In all honesty, ", "I have to say, ", "All things considered, ",
]

FILLER_ENDERS = [
    " overall", " in general", " to be honest", " I think",
    " honestly", " for sure", " really", " tbh",
    " in my opinion", " if I'm being honest", " all in all",
]


def apply_synonym_substitution(text, probability=0.3):
    """Replace some words with synonyms."""
    words = text.split()
    result = []
    for word in words:
        lower = word.lower().strip(".,!?;:")
        if lower in SYNONYMS and random.random() < probability:
            punct = ""
            if word and word[-1] in ".,!?;:":
                punct = word[-1]
            replacement = random.choice(SYNONYMS[lower])
            if word[0].isupper():
                replacement = replacement.capitalize()
            result.append(replacement + punct)
        else:
            result.append(word)
    return " ".join(result)


def add_intensifier_or_hedge(text, probability=0.2):
    """Add intensifiers or hedges before adjectives."""
    adjectives = list(SYNONYMS.keys())
    words = text.split()
    result = []
    for i, word in enumerate(words):
        lower = word.lower().strip(".,!?;:")
        if lower in adjectives and random.random() < probability:
            if i == 0 or words[i - 1].lower() not in INTENSIFIERS + HEDGES:
                modifier = random.choice(INTENSIFIERS if random.random() > 0.5 else HEDGES)
                result.append(modifier)
        result.append(word)
    return " ".join(result)


def apply_casual_style(text, probability=0.15):
    """Add casual survey-response stylistic markers."""
    if random.random() < probability:
        text = random.choice(FILLER_STARTERS) + text[0].lower() + text[1:]
    if random.random() < probability:
        text = text.rstrip(".!?,") + random.choice(FILLER_ENDERS)
    if random.random() < 0.05:
        text = text.upper()
    if random.random() < 0.08:
        for correct, typo in TYPO_PAIRS:
            if correct in text.lower():
                text = re.sub(re.escape(correct), typo, text, flags=re.IGNORECASE, count=1)
                break
    return text


def recombine_phrases(responses, n=1):
    """Create new responses by recombining clause-level phrases from existing ones."""
    new_responses = []
    if len(responses) < 2:
        return [apply_synonym_substitution(responses[0]) if responses else "no comment"] * n
    for _ in range(n):
        # Pick two source responses
        r1, r2 = random.sample(responses, 2)
        # Split on clause boundaries
        parts1 = re.split(r'[.!?,;&]+\s*', r1)
        parts2 = re.split(r'[.!?,;&]+\s*', r2)
        parts1 = [p.strip() for p in parts1 if len(p.strip()) > 10]
        parts2 = [p.strip() for p in parts2 if len(p.strip()) > 10]
        if parts1 and parts2:
            p1 = random.choice(parts1)
            p2 = random.choice(parts2)
            connector = random.choice([" and ", ". ", ", also ", " but ", ". Additionally, ", " - "])
            new = p1 + connector + p2[0].lower() + p2[1:] if len(p2) > 1 else p1
            new_responses.append(new.strip())
        else:
            new_responses.append(random.choice(responses))
    return new_responses


def generate_short_response():
    """Generate minimal/low-effort responses that appear in real surveys."""
    shorts = [
        "fine", "ok", "good", "nothing", "meh", "idk", "whatever",
        "ok I guess", "fine I guess", "good overall", "nothing special",
        "ok service", "not bad", "it was alright", "n/a", "no comment",
        "fine nothing special", "decent", "alright", "satisfactory",
        "average", "it was okay", "no issues", "nothing to add",
    ]
    return random.choice(shorts)


def generate_variation(seed_response, all_responses_for_topic):
    """Generate a single new response variation from a seed."""
    strategy = random.random()

    if strategy < 0.05:
        # 5%: short/low-effort response
        return generate_short_response()
    elif strategy < 0.35:
        # 30%: synonym substitution + style variation
        varied = apply_synonym_substitution(seed_response, probability=0.35)
        varied = add_intensifier_or_hedge(varied, probability=0.2)
        varied = apply_casual_style(varied, probability=0.2)
        return varied
    elif strategy < 0.60:
        # 25%: phrase recombination from same topic
        results = recombine_phrases(all_responses_for_topic, n=1)
        return apply_casual_style(results[0], probability=0.15)
    elif strategy < 0.80:
        # 20%: light synonym + add/remove filler
        varied = apply_synonym_substitution(seed_response, probability=0.2)
        varied = apply_casual_style(varied, probability=0.3)
        return varied
    else:
        # 20%: heavier variation - multiple transforms
        varied = apply_synonym_substitution(seed_response, probability=0.4)
        varied = add_intensifier_or_hedge(varied, probability=0.3)
        varied = apply_casual_style(varied, probability=0.25)
        return varied


def make_unique_id(text, index):
    """Create a deterministic but unique hash-based check to avoid exact duplicates."""
    return hashlib.md5(f"{text}_{index}".encode()).hexdigest()[:8]


# ─── Dataset expansion ───────────────────────────────────────────────────────

def expand_dataset(input_path, output_path, target_rows=1000):
    """Expand a dataset from its seed to target_rows."""
    df = pd.read_csv(input_path)
    original_rows = len(df)
    cols = list(df.columns)
    id_col = cols[0]        # 'id'
    text_col = cols[1]      # 'response'
    respondent_col = cols[2] # 'patient_id' / 'respondent_id' / 'participant_id'
    timestamp_col = cols[3]  # 'timestamp'
    segment_col = cols[4]    # department / demographic_segment / age_group / topic

    # Get segment distribution from original
    segment_counts = df[segment_col].value_counts(normalize=True)
    segments = list(segment_counts.index)
    segment_probs = list(segment_counts.values)

    # Group responses by segment for topic-aware recombination
    segment_responses = {}
    for seg in segments:
        segment_responses[seg] = df[df[segment_col] == seg][text_col].tolist()

    all_responses = df[text_col].tolist()
    needed = target_rows - original_rows

    # Determine respondent ID prefix
    sample_rid = str(df[respondent_col].iloc[0])
    rid_prefix = re.match(r'^[A-Za-z]+', sample_rid)
    rid_prefix = rid_prefix.group() if rid_prefix else "R"

    # Generate timestamps continuing from last
    last_ts = pd.to_datetime(df[timestamp_col].iloc[-1])
    time_delta = timedelta(minutes=random.randint(5, 30))

    new_rows = []
    seen_texts = set(all_responses)

    for i in range(needed):
        # Pick segment maintaining original proportions
        seg = random.choices(segments, weights=segment_probs, k=1)[0]

        # Pick a seed response from this segment
        seed = random.choice(segment_responses[seg])

        # Generate variation
        attempts = 0
        while attempts < 5:
            new_text = generate_variation(seed, segment_responses[seg])
            # Ensure reasonable length (not too short unless intentionally short)
            if new_text not in seen_texts or attempts >= 4:
                break
            attempts += 1

        seen_texts.add(new_text)

        new_id = original_rows + i + 1
        new_rid = f"{rid_prefix}{new_id}"
        last_ts += timedelta(minutes=random.randint(3, 45), seconds=random.randint(0, 59))
        new_ts = last_ts.strftime("%Y-%m-%d %H:%M:%S")

        new_rows.append({
            id_col: new_id,
            text_col: new_text,
            respondent_col: new_rid,
            timestamp_col: new_ts,
            segment_col: seg,
        })

    # Combine original + new
    new_df = pd.DataFrame(new_rows, columns=cols)
    expanded = pd.concat([df, new_df], ignore_index=True)
    expanded[id_col] = range(1, len(expanded) + 1)

    expanded.to_csv(output_path, index=False)
    print(f"  ✅ {input_path} ({original_rows} rows) -> {output_path} ({len(expanded)} rows)")
    return expanded


# ─── Main ────────────────────────────────────────────────────────────────────

DATASETS = [
    {
        "input": "data/Healthcare_Patient_Feedback_300.csv",
        "output": "data/Healthcare_Patient_Feedback_1000.csv",
    },
    {
        "input": "data/Market_Research_Survey_300.csv",
        "output": "data/Market_Research_Survey_1000.csv",
    },
    {
        "input": "data/Psychology_Wellbeing_Study_300.csv",
        "output": "data/Psychology_Wellbeing_Study_1000.csv",
    },
    {
        "input": "data/Remote_Work_Experiences_200.csv",
        "output": "data/Remote_Work_Experiences_1000.csv",
    },
    {
        "input": "data/cricket_responses.csv",
        "output": "data/cricket_responses_1000.csv",
    },
    {
        "input": "data/fashion_responses.csv",
        "output": "data/fashion_responses_1000.csv",
    },
]


if __name__ == "__main__":
    print("Expanding datasets to 1000 rows each...\n")

    project_root = Path(__file__).resolve().parent.parent

    for ds in DATASETS:
        input_path = project_root / ds["input"]
        output_path = project_root / ds["output"]
        expand_dataset(str(input_path), str(output_path), target_rows=1000)

    print("\nDone! All datasets expanded to 1000 rows.")
