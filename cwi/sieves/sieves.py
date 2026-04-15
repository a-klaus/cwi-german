# -*- coding: utf-8 -*-

"""
Length-related sieves for rule-based complex word identification.
"""

import re
import string

import numpy as np

from nltk.tokenize import SyllableTokenizer


# Syllable tokenizer
GERMAN_SONORITY_HIERARCHY = [
    "aeiouyüöä",                # vowels
    "lmnr",                     # nasals
    "zvsfßw",                   # fricatives
    "bcdgtkpqxhj0123456789 ",   # stops
]
SYLL_TOKENIZER = SyllableTokenizer(sonority_hierarchy=GERMAN_SONORITY_HIERARCHY)
# Punctuation
PUNCTUATION = set(string.punctuation) - {"-"}  # hyphens are accepted
# Number words from 1 to 20 in German
NUMBER_WORDS_DE = {
    "zwei",
    "drei",
    "vier",
    "fünf",
    "sechs",
    "sieben",
    "acht",
    "neun",
    "zehn",
    "elf",
    "zwölf",
    "dreizehn",
    "vierzehn",
    "fünfzehn",
    "sechzehn",
    "siebzehn",
    "achtzehn",
    "neunzehn",
    "zwanzig"
}


def number_of_characters(target, threshold=10):
    """Check if number of characters exceeds threshold.

    Args:
        target (spacy.tokens.doc.Doc): target span
        threshold (int): the threshold for considering a word long

    Returns:
        bool: True if threshold is exceeded
    """
    # If a word consists of multiple visually (by hyphen) seperated parts:
    # Only use lengths of parts
    if "-" in target.text:
        target = [t for token in target for t in token.text.split("-")]
    return np.max([len(t) for t in target]) > threshold


def number_of_syllables(target, threshold=2):
    """Check if number of syllables exceeds threshold.

    Args:
        target (spacy.tokens.doc.Doc): target span
        threshold (int): the threshold for considering a word long

    Returns:
        bool: True if threshold is exceeded
    """
    # If a word consists of multiple visually seperated parts:
    # Only use lengths of parts
    if "-" in target.text:
        target = [t for token in target for t in token.text.split("-")]
    else:
        # Purpose of this step is to have the same data type on both routes
        target = [t.text for t in target]
    return np.max([len(SYLL_TOKENIZER.tokenize(t)) for t in target]) > threshold


def is_nominalization(target):
    """Check if target is nominalization (by  some very simple heuristics).

    Args:
        target (spacy.tokens.doc.Doc): target span

    Returns:
        bool: True if target is nominalization
    """
    # Currently ignoring cases like "das Grauen"
    target = target[-1]  # that or I do it for every part of a target
    s = ["ierung", "ation", "ierungen", "ationen"]
    return target.tag_ == "NN" and any([target.lemma_.endswith(i) for i in s])


def contains_punctuation_characters(target):
    """True if target contains punctuation characters, else False.

    Args:
        target (spacy.tokens.doc.Doc): target span

    Returns:
        bool: True if target contains punctuation characters
    """
    return len(target.text) > 1 and any(c in PUNCTUATION for c in target.text)


def is_textbased_abbreviation(target):
    """True if target is (or contains) text-based abbreviation, else False.

    Args:
        target (spacy.tokens.doc.Doc): target span

    Returns:
        bool: True if target is text-based abbreviation
    """
    return bool(re.match(r"[a-zA-Z]+.*\.$", target.text))


def identify_numberwords(target):
    """Identify numbers in spelled out form.

    Args:
        target (spacy.tokens.doc.Doc): target span

    Returns:
        bool: True if target is number word
    """
    return target.lemma_.lower() in NUMBER_WORDS_DE

