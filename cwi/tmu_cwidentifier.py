# -*- coding: utf-8 -*-

"""
Complex Word Identification on German using the features as in the TMU paper.
"""

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .baseline import BaselineFeatureBasedCWIdentifier
from .util.errors import InvalidFeatureSettingError
from .util.ling_processing import lemmatize


class TMUCWIdentifier(BaselineFeatureBasedCWIdentifier):
    """Classifier using the number of characters and word frequency counts.

    This classifier is based on the TMU system submitted to the 2018 Shared
    Task on Complex Word Identification. If utilizes frequency counts and
    unigram probabilities in different corpora as well as the character length
    and number of tokens, classified with a Random Forest classifier. The
    corpora used to assess frequencies and probabilities are a 40 Mio token
    wikipedia dump, a x Mio token wikinews dump and the German subsection of
    the lang8 corpus.

    Args:
        wikipedia_frequencies (nltk.probability.FreqDist): frequency
            distribution of wikipedia dump of appropriate size
        wikinews_frequencies (nltk.probability.FreqDist): frequency
            distribution of wikinews dump of appropriate size
        lang8_frequencies (nltk.probability.FreqDist): frequency
            distribution of the German subsection of the lang8 corpus
        spacy_model (spacy.lang.de.German): spacy model to be used for
            preprocessing (should be a German model, as this classifier was
            designed for German)

    Attributes:
        classifier (classifier): model that can be fit and used to predict,
            default is a pipeline of sklearn's StandardScaler and
            NearestCentroid classifier
    """

    def __init__(
        self,
        wikipedia_frequencies,
        wikinews_frequencies,
        lang8_frequencies,
        spacy_model=None,
        lemma_dict={}
    ):
        self._spacy_model = spacy_model
        # Frequency distributions
        self._wiki_fdist = wikipedia_frequencies
        self._wikinews_fdist = wikinews_frequencies
        self._lang8_fdist = lang8_frequencies
        # parameter for feature generation
        self.char_len_method = "sum"
        # Lemmatization
        self._lemma_dict = lemma_dict | {v: v for v in lemma_dict.values()}
        self.lemmatize_fallback = lemmatize
        # Sentence dictionary for more effective preprocessing
        self._sent_dict = {}
        self._features = {
            # wordlevel
            "number_of_characters": self._get_number_of_characters,
            "number_of_tokens": len,
            "wikipedia_freq": self._get_wikipedia_freq,
            "wikinews_freq": self._get_wikinews_freq,
            "lang8_freq": self._get_lang8_freq,
            "wikipedia_prob": self._get_wikipedia_prob,
            "wikinews_prob": self._get_wikinews_prob,
            "lang8_prob": self._get_lang8_prob
        }
        self._unused_features = {}
        self.classifier = make_pipeline(StandardScaler(),
                                        RandomForestClassifier(n_estimators=100,
                                                               random_state=666))

    def _get_wikipedia_freq(self, target):
        """Get raw frequency count from wikipedia frequency distribution."""
        lemma = self._lemmatize(target).lower()
        return self._wiki_fdist[lemma]

    def _get_wikinews_freq(self, target):
        """Get raw frequency count from wikinews frequency distribution."""
        lemma = self._lemmatize(target).lower()
        return self._wikinews_fdist[lemma]

    def _get_lang8_freq(self, target):
        """Get raw frequency count from lang8 frequency distribution."""
        lemma = self._lemmatize(target).lower()
        return self._lang8_fdist[lemma]

    def _get_wikipedia_prob(self, target):
        """Get unigram probability from wikipedia frequency distribution."""
        lemma = self._lemmatize(target).lower()
        return self._wiki_fdist.freq(lemma)

    def _get_wikinews_prob(self, target):
        """Get unigram probability from wikinews frequency distribution."""
        lemma = self._lemmatize(target).lower()
        return self._wikinews_fdist.freq(lemma)

    def _get_lang8_prob(self, target):
        """Get unigram probability from lang8 frequency distribution."""
        lemma = self._lemmatize(target).lower()
        return self._lang8_fdist.freq(lemma)

    def _get_number_of_characters(self, target):
        """Get the mean character length for the words in the target span."""
        if self.char_len_method == "sum":
            return np.sum([len(t) for t in target])
        if self.char_len_method == "mean":
            return np.mean([len(t) for t in target])
        else:
            raise InvalidFeatureSettingError

    def __str__(self):
        """Return representation of classifier."""
        return f"TMU Complex Word Identifier: {self.classifier}"

