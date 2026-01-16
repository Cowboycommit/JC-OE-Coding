#!/usr/bin/env python3
"""
Create representative sentiment dataset samples.

These are carefully curated examples that follow the characteristics
of the original benchmark datasets (SST-2, SST-5, IMDB) for demonstration
and testing purposes.
"""

import os
import random
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# SST-2 Style sentences (movie review fragments)
SST2_POSITIVE = [
    "A warm, funny, engaging film.",
    "The movie is delightfully entertaining from start to finish.",
    "An exceptional achievement in storytelling.",
    "Brilliant performances by the entire cast.",
    "A masterpiece of modern cinema.",
    "Highly recommended for all audiences.",
    "The script is witty and intelligent.",
    "A thoroughly enjoyable experience.",
    "Outstanding direction and cinematography.",
    "One of the best films of the year.",
    "A touching and heartfelt story.",
    "The acting is superb throughout.",
    "A refreshing take on the genre.",
    "Beautifully crafted and emotionally resonant.",
    "The humor is sharp and well-timed.",
    "Captivating from the first frame to the last.",
    "A powerful and moving experience.",
    "An inventive and original work.",
    "The performances are nothing short of remarkable.",
    "A triumph of visual storytelling.",
    "Smart, stylish, and thoroughly entertaining.",
    "The chemistry between the leads is electric.",
    "A genuinely funny and clever comedy.",
    "The film succeeds on every level.",
    "An absolute delight from beginning to end.",
    "Wonderfully crafted with attention to detail.",
    "The best movie I have seen in months.",
    "A perfect blend of drama and comedy.",
    "The cinematography is breathtaking.",
    "An emotionally satisfying conclusion.",
    "The dialogue sparkles with wit.",
    "A tour de force of acting talent.",
    "Compelling and thought-provoking.",
    "The pacing is perfect throughout.",
    "A feel-good movie that actually works.",
    "Surprisingly deep and meaningful.",
    "The score perfectly complements the visuals.",
    "An instant classic in the making.",
    "Clever writing and superb acting combine beautifully.",
    "The film has real heart and soul.",
    "A joyous celebration of cinema.",
    "Utterly charming and endlessly watchable.",
    "The performances elevate the material.",
    "A rare gem worth discovering.",
    "The direction is confident and assured.",
    "A wonderfully uplifting experience.",
    "Smart entertainment that respects its audience.",
    "The ensemble cast is fantastic.",
    "A movie that stays with you long after.",
    "Genuinely moving and beautifully told.",
    "The visuals are stunning throughout.",
    "A richly rewarding viewing experience.",
    "Perfectly cast and expertly directed.",
    "The film finds the perfect tone.",
    "A heartwarming story told with skill.",
    "Memorable characters and sharp dialogue.",
    "An achievement in filmmaking craft.",
    "The movie exceeds all expectations.",
    "Fresh and original storytelling.",
    "A crowd-pleaser in the best sense.",
    "Engrossing from start to finish.",
    "The screenplay is razor sharp.",
    "A beautiful film in every way.",
    "Outstanding performances across the board.",
    "The film strikes the perfect balance.",
    "Intelligent and deeply satisfying.",
    "A movie that delivers on its promise.",
    "The chemistry is undeniable.",
    "Masterfully constructed narrative.",
    "An absolute pleasure to watch.",
    "The film has genuine emotional weight.",
    "Superbly acted and beautifully shot.",
    "A standout achievement this year.",
    "Effortlessly entertaining.",
    "The direction is flawless.",
]

SST2_NEGATIVE = [
    "A complete waste of time.",
    "The plot is confusing and poorly executed.",
    "Terrible acting and weak dialogue.",
    "Boring and predictable from the start.",
    "The worst film I have seen this year.",
    "Disappointing on every level.",
    "A tedious and uninspiring effort.",
    "The script is lazy and unoriginal.",
    "Fails to deliver on its promise.",
    "An unnecessary and forgettable sequel.",
    "The pacing is painfully slow.",
    "Poorly directed and badly edited.",
    "A joyless and dreary experience.",
    "The characters are one-dimensional.",
    "Not worth the price of admission.",
    "The film never finds its footing.",
    "A messy and unfocused narrative.",
    "Lacking any real substance.",
    "The performances feel phoned in.",
    "A formulaic and uninspired effort.",
    "The dialogue is cringe-worthy.",
    "Overlong and overstuffed.",
    "A dull and lifeless affair.",
    "The plot holes are distracting.",
    "Fails to engage on any level.",
    "The tone is wildly inconsistent.",
    "A shallow and superficial film.",
    "The ending is deeply unsatisfying.",
    "Derivative and unimaginative.",
    "The film simply does not work.",
    "A missed opportunity from start to finish.",
    "The chemistry between leads is nonexistent.",
    "Poorly conceived and badly executed.",
    "A forgettable and bland experience.",
    "The humor falls completely flat.",
    "Tedious beyond belief.",
    "The script desperately needs work.",
    "A soulless and empty production.",
    "The direction lacks any vision.",
    "Painfully predictable throughout.",
    "The film drags interminably.",
    "A disappointment in every regard.",
    "Wooden performances throughout.",
    "The pacing is all wrong.",
    "An exercise in tedium.",
    "The movie has no emotional resonance.",
    "A cynical and hollow effort.",
    "The plot makes no sense.",
    "Completely lacking in charm.",
    "A film best forgotten quickly.",
    "The editing is choppy and confusing.",
    "Fails to justify its runtime.",
    "A tired and worn out concept.",
    "The film never comes alive.",
    "Aggressively mediocre at best.",
    "A frustrating viewing experience.",
    "The twist is obvious from the start.",
    "Uninspired and by the numbers.",
    "A waste of talented actors.",
    "The film collapses under its own weight.",
    "Dreadfully dull throughout.",
    "A failed experiment in filmmaking.",
    "The story goes nowhere.",
    "An unpleasant experience overall.",
    "Lacking any genuine emotion.",
    "The film feels rushed and incomplete.",
    "A chore to sit through.",
    "The potential is squandered entirely.",
    "An utter mess from beginning to end.",
    "The worst of the franchise by far.",
    "A slog from start to finish.",
    "Deeply disappointing result.",
    "The film never earns its moments.",
    "A thoroughly unpleasant experience.",
    "Fails on its own terms.",
]

# SST-5 Style sentences with fine-grained sentiment
SST5_SAMPLES = [
    # Very negative (0)
    ("An absolutely terrible movie with no redeeming qualities whatsoever.", 0),
    ("One of the worst films ever made in the history of cinema.", 0),
    ("A disaster from start to finish that insults the audience.", 0),
    ("Unwatchable garbage that should never have been released.", 0),
    ("The most boring and pointless movie I have ever endured.", 0),
    ("A complete and utter failure on every conceivable level.", 0),
    ("Painfully bad acting combined with an incoherent plot.", 0),
    ("An abomination that wastes everyone's time and money.", 0),
    ("Truly dreadful in ways I cannot begin to describe.", 0),
    ("The worst kind of cynical filmmaking possible.", 0),
    ("An embarrassment to everyone involved in its making.", 0),
    ("Insultingly stupid and remarkably incompetent.", 0),
    ("A new low in filmmaking standards.", 0),
    ("Aggressively awful from the opening frame.", 0),
    ("So bad it approaches self-parody.", 0),
    # Negative (1)
    ("The film fails to deliver on its promising premise.", 1),
    ("Disappointing performances undermine the story throughout.", 1),
    ("Below average with several weak moments that stand out.", 1),
    ("Not as good as expected and rather forgettable overall.", 1),
    ("The movie has some significant issues that are hard to overlook.", 1),
    ("A lackluster effort that never quite comes together.", 1),
    ("The pacing issues hurt an otherwise serviceable film.", 1),
    ("Falls short of its potential in frustrating ways.", 1),
    ("Mediocre execution of a decent concept.", 1),
    ("The weak script hampers the solid performances.", 1),
    ("An uneven film with more misses than hits.", 1),
    ("Doesn't live up to the hype surrounding it.", 1),
    ("A letdown considering the talent involved.", 1),
    ("The film never quite finds its voice.", 1),
    ("Somewhat tedious despite a few bright spots.", 1),
    # Neutral (2)
    ("An average film with some decent moments throughout.", 2),
    ("Neither great nor terrible in any particular way.", 2),
    ("The movie is watchable but nothing particularly special.", 2),
    ("A mixed bag with both good and bad elements present.", 2),
    ("Fairly standard fare for this type of genre film.", 2),
    ("The performances are adequate but unremarkable.", 2),
    ("It does exactly what you expect and nothing more.", 2),
    ("A competent if somewhat forgettable production.", 2),
    ("Middle of the road entertainment.", 2),
    ("Decent enough to pass the time.", 2),
    ("The film has its moments both good and bad.", 2),
    ("Neither offensive nor particularly memorable.", 2),
    ("An okay film for undemanding viewers.", 2),
    ("Serviceable entertainment without ambition.", 2),
    ("Takes no risks and achieves modest results.", 2),
    # Positive (3)
    ("A good film with several enjoyable performances.", 3),
    ("Well-made and entertaining from start to finish.", 3),
    ("Above average with several memorable scenes throughout.", 3),
    ("Solid storytelling combined with good direction.", 3),
    ("An enjoyable watch that delivers on its promise.", 3),
    ("The film works more often than it doesn't.", 3),
    ("A pleasant surprise with genuine appeal.", 3),
    ("Strong performances carry the material effectively.", 3),
    ("Better than expected in several ways.", 3),
    ("An engaging film with real charm.", 3),
    ("Successfully balances drama and entertainment.", 3),
    ("Worth watching for fans of the genre.", 3),
    ("The film achieves what it sets out to do.", 3),
    ("A satisfying experience overall.", 3),
    ("Manages to rise above its modest premise.", 3),
    # Very positive (4)
    ("An absolute masterpiece of cinematic achievement.", 4),
    ("One of the best films ever made without question.", 4),
    ("A brilliant achievement in storytelling and craft.", 4),
    ("Exceptional filmmaking at its absolute finest.", 4),
    ("A perfect movie that exceeds all possible expectations.", 4),
    ("A tour de force that redefines the genre entirely.", 4),
    ("Stunning in every conceivable way imaginable.", 4),
    ("An extraordinary film that sets a new standard.", 4),
    ("Flawless execution of a brilliant concept.", 4),
    ("A transcendent cinematic experience.", 4),
    ("Nothing short of a miraculous achievement.", 4),
    ("Sets a new benchmark for excellence.", 4),
    ("An instant classic that will endure forever.", 4),
    ("Phenomenal performances matched by superb direction.", 4),
    ("A landmark achievement in filmmaking history.", 4),
]

# IMDB Style reviews (longer format)
IMDB_POSITIVE = [
    "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout. The director clearly had a vision and executed it flawlessly. I highly recommend this to anyone who enjoys quality cinema with heart and substance.",
    "One of the best films I've seen in years. The director did an amazing job bringing this story to life. The performances were outstanding and emotionally moving. Every scene felt purposeful and contributed to the overall narrative.",
    "A masterpiece of storytelling that deserves all the praise it receives. Every scene was crafted with care and attention to detail. The cinematography was breathtaking and the score perfectly complemented every emotional beat.",
    "I was thoroughly entertained from beginning to end. The characters were well-developed and the dialogue was sharp and witty. This is the kind of movie that reminds you why you love cinema in the first place.",
    "An incredible journey that left me thinking long after the credits rolled. This is what great filmmaking looks like when all the elements come together perfectly. The casting was spot-on and the pacing was excellent.",
    "Brilliant! The cast delivered amazing performances and the story was both touching and thought-provoking. I found myself completely immersed in this world. It's rare to find a film that works on so many levels.",
    "A beautiful film that exceeded all my expectations. The visual effects were stunning and the story was deeply moving. This is a movie that will stay with me for a long time. Absolutely worth watching multiple times.",
    "Outstanding movie with a perfect blend of drama and humor that never feels forced. Every actor gave their best performance and the chemistry between the leads was electric. A rare gem in today's cinema landscape.",
    "What a phenomenal film! The storytelling was gripping, the performances were nuanced, and the production values were top-notch. I was on the edge of my seat the entire time. This deserves every award it receives.",
    "This film is a triumph in every sense of the word. The script is intelligent, the direction is assured, and the performances are uniformly excellent. It's the kind of movie that makes you fall in love with cinema all over again.",
    "Absolutely loved this movie from start to finish. The attention to detail in every frame is remarkable. The lead actor gives a career-defining performance, and the supporting cast is equally impressive. A must-see film.",
    "I went in with high expectations and this film exceeded every single one of them. The pacing is perfect, the story is compelling, and the emotional payoffs are genuinely earned. This is storytelling at its finest.",
    "A stunning achievement in filmmaking that sets a new standard for the genre. The technical aspects are impeccable, the performances are powerful, and the message is timely and important. Truly essential viewing.",
    "This is the kind of movie that reminds you of the magic of cinema. Every element works in perfect harmony to create an unforgettable experience. I laughed, I cried, and I left the theater deeply moved.",
    "An absolutely captivating film that held my attention from the first frame to the last. The writing is sharp, the direction is confident, and the performances are filled with depth and nuance. Highly recommended.",
    "Exceptional filmmaking from a visionary director. This movie takes risks and they all pay off beautifully. The ensemble cast delivers some of the best performances of their careers. A modern classic.",
    "I cannot say enough good things about this movie. It's smart, it's moving, it's beautifully crafted, and it stays with you long after viewing. This is why we go to the movies. Pure cinematic excellence.",
    "What a wonderful surprise this film turned out to be. The storytelling is masterful, weaving together multiple threads into a satisfying whole. The performances are uniformly excellent. Do not miss this one.",
]

IMDB_NEGATIVE = [
    "I really wanted to like this movie but it was a major disappointment from start to finish. The plot was confusing, the acting was wooden, and the pacing was all over the place. I cannot recommend this to anyone.",
    "One of the worst films I've ever had the misfortune to sit through. The dialogue was cringe-worthy and the story made no sense whatsoever. I found myself checking my watch repeatedly. A complete waste of time.",
    "A complete waste of time and money that I wish I could get back. The director seemed to have no vision and the result is a mess of a movie. None of the characters are likeable or interesting.",
    "Boring from start to finish with no redeeming qualities. The pacing was terrible and I found myself checking my watch constantly. The plot was predictable and the acting was mediocre at best.",
    "The trailers were completely misleading about what this movie actually is. This movie had none of the excitement promised and left me feeling cheated. The actual film is a dull, poorly paced disappointment.",
    "Terrible acting, poor writing, and sloppy direction make this a chore to watch. I cannot believe this got made with actual studio funding. Everything about this movie feels half-baked and lazy.",
    "An uninspired sequel that adds nothing to the franchise while ruining beloved characters. Skip this one and save your money for something worthwhile. This is cash-grab filmmaking at its worst.",
    "Predictable and unoriginal from the opening scene. Every plot twist was telegraphed from a mile away. The script desperately needed another draft, and the performances are phoned in.",
    "This movie is a perfect example of how not to make a film. The narrative is incoherent, the characters are unlikeable, and the ending is laughably bad. I wanted to walk out multiple times.",
    "What a massive disappointment this turned out to be given the talent involved. The script is a mess, the pacing is glacial, and the performances are wasted on material this weak. Skip it.",
    "I have rarely seen a movie fail so completely on every level. The story goes nowhere, the characters are paper-thin, and the dialogue is embarrassingly bad. A truly terrible film.",
    "Avoid this movie at all costs. Two hours of my life I will never get back. The plot makes no sense, the acting is atrocious, and the direction is amateurish at best.",
    "How this movie got greenlit is beyond comprehension. It fails as entertainment, it fails as art, and it fails as a coherent narrative. One of the worst theater experiences of my life.",
    "Absolutely dreadful in every conceivable way. The script is a disaster, the performances are wooden, and the film has no idea what it wants to be. A frustrating and unpleasant experience.",
    "This is the kind of movie that makes you question Hollywood's judgment entirely. Lazy writing, poor direction, and performances that suggest no one involved cared about the final product.",
    "A joyless and tedious experience from beginning to end. The film drags interminably, the plot is nonsensical, and the attempts at humor fall completely flat. Truly unwatchable.",
    "I cannot believe the positive reviews for this film. It's poorly written, badly acted, and directed without any vision or style. One of the most overrated movies in recent memory.",
    "What a colossal waste of potential. Great premise, talented cast, and it all adds up to absolutely nothing. The execution is so poor that you wonder if anyone actually read the script.",
]

def create_sst2_dataset():
    """Create SST-2 style dataset."""
    print("Creating SST-2 sentiment dataset...")

    rows = []
    for text in SST2_POSITIVE:
        rows.append({'text': text, 'label': 1, 'sentiment': 'positive'})
    for text in SST2_NEGATIVE:
        rows.append({'text': text, 'label': 0, 'sentiment': 'negative'})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_path = os.path.join(DATA_DIR, "SST-2 Sentiment Dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df)} samples to {output_path}")
    return df


def create_sst5_dataset():
    """Create SST-5 style dataset."""
    print("Creating SST-5 sentiment dataset...")

    rows = []
    label_map = {0: "very negative", 1: "negative", 2: "neutral", 3: "positive", 4: "very positive"}

    for text, label in SST5_SAMPLES:
        rows.append({'text': text, 'label': label, 'sentiment': label_map[label]})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_path = os.path.join(DATA_DIR, "SST-5 Sentiment Dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df)} samples to {output_path}")
    return df


def create_imdb_dataset():
    """Create IMDB style dataset."""
    print("Creating IMDB movie reviews dataset...")

    rows = []
    for text in IMDB_POSITIVE:
        rows.append({'text': text, 'label': 1, 'sentiment': 'positive'})
    for text in IMDB_NEGATIVE:
        rows.append({'text': text, 'label': 0, 'sentiment': 'negative'})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_path = os.path.join(DATA_DIR, "IMDB Movie Reviews.csv")
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df)} samples to {output_path}")
    return df


def main():
    """Create all sample datasets."""
    print("=" * 60)
    print("Creating Representative Sentiment Datasets")
    print("=" * 60)
    print()

    os.makedirs(DATA_DIR, exist_ok=True)

    create_sst2_dataset()
    print()
    create_sst5_dataset()
    print()
    create_imdb_dataset()
    print()

    print("=" * 60)
    print("Datasets created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
