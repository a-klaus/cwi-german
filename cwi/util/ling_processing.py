# -*- coding: utf-8 -*-

import re

from spacy.tokens import Span


PATTERNS = {
    # nominalizations
    r"^([A-Z].+)nden$": r"\1nde",
    r"^([A-Z].+)ens$": r"\1en",
    # nouns ending on -heit, -keit or -ung
    r"^([A-Z].+)([hk]eit|ung|schaft|ion)en$": r"\1\2",
    # adjectives
    r"^([a-z].+)(end|wert|lich|bar)e([rsn]?|re[rsn]?)$": r"\1\2",
    r"^([a-z]+)(isch)e([rsn]?|re[rsn]?)$": r"\1\2",
    # derived adjectives
    r"^(un)?(ver|zer|be|ge)(.+t)e([rsn]?|re[rsn]?)$": r"\1\2\3",
    # verbs
    r"^(ver|zer|be)(.+?)(ss)?(s?t|e)$": r"\1\2\3en"
}


def lemmatize(word):
    """Simple rule-based lemmatization using a couple of heuristics."""
    for pattern, replacement in PATTERNS.items():
        if re.match(pattern, word):
            return re.sub(pattern, replacement, word)
    return word


def ne_retokenize(doc):
    """Join named entities into single tokens."""
    with doc.retokenize() as retokenizer:
        for ne in doc.ents:
            # possibly strip punctuation, sometimes trailing the named entity
            # in spacy models for German
            x = 0
            if ne[-1].pos_ == "PUNCT":
                x = 1
            # merge into one token
            retokenizer.merge(
                doc[ne.start:ne.end-x],
                attrs={"LEMMA": str(doc[ne.start:ne.end-x]),
                       "_": {"is_ne": True}}
            )
    # reset doc ents, preserving the label
    zipped_ents = zip([i.label_ for i in doc.ents],
                      [tok for tok in doc if tok._.is_ne])
    doc.set_ents([Span(doc, tok.i, tok.i+1, c)
                  for c, tok in zipped_ents],
                 default="outside")
    return doc

