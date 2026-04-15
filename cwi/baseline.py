# -*- coding: utf-8 -*-

"""
Simple Complex Word Identifier using word frequency and length as features,
classifying with a Nearest Centroid classifier.
"""

import numpy as np
import pandas as pd

from nltk.tokenize import SyllableTokenizer
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .abstract_cwidentifier import AbstractCWIdentifier
from .util.ling_processing import lemmatize


# tqdm setup
tqdm.pandas(desc="Feature Vector Generation")


GERMAN_SONORITY_HIERARCHY = [
    "aeiouyüöä",                # vowels
    "lmnr",                     # nasals
    "zvsfßw",                   # fricatives
    "bcdgtkpqxhj0123456789 ",   # stops
]
SYLL_TOKENIZER = SyllableTokenizer(sonority_hierarchy=GERMAN_SONORITY_HIERARCHY)
GERMAN_VOWELS = {"a", "e", "i", "o", "u", "ä", "ö", "ü"}


class BaselineFeatureBasedCWIdentifier(AbstractCWIdentifier):
    """Baseline classifier using the number of characters and word frequency.

    This classifier uses the aforementioned features and a Nearest Centroid
    classifier to perform complex word identification. This simple approach is
    used as a baseline system for evaluation of other classifiers.

    Args:
        freq_dict (dict): frequency distribution from which frequency values
            for words can be retrieved, i.e. a dictionary mapping words to
            frequencies
        spacy_model (spacy.lang.de.German): spacy model to be used for
            preprocessing (should be a German model, as this classifier was
            designed for German)
        lemma_dict (dict): a dictionary for assigning correct lemmata to word
            forms; if a target is not in the dictionary a simple rule-based
            fallback method will be used

    Attributes:
        classifier (classifier): model that can be fit and used to predict,
            default is a pipeline of sklearn's StandardScaler and
            NearestCentroid classifier
        lemmatize_fallback (function): simple lemmatizing function to use as a
            fallback forwords not found in the lemma dictionary, default here
            is for German
    """

    def __init__(self, freq_dict, spacy_model=None, lemma_dict={}):
        self._spacy_model = spacy_model
        # Sentence dictionary for more effective preprocessing
        self._sent_dict = {}
        self._freq_dict = freq_dict
        self._lemma_dict = lemma_dict | {v: v for v in lemma_dict.values()}
        self.lemmatize_fallback = lemmatize
        self._features = {
            "number_of_characters": self._get_number_of_characters,
            "frequency": self._get_frequency,
            "number_of_syllables": self._get_number_of_syllables,
            "number of vowels": self._get_number_of_vowels
        }
        self._unused_features = {}
        self.classifier = make_pipeline(StandardScaler(),
                                        NearestCentroid())

    def _lemmatize(self, target):
        """Lemmatize a target by lookup in dict with rule-based fallback."""
        # Use lemma from dict if possible, else lemmatize by fallback function
        return self._lemma_dict.get(target.text,
                                    self.lemmatize_fallback(target.text))

    def _get_frequency(self, word):
        """Retrieve frequency of a word from a frequency distribution."""
        lemma = self._lemmatize(word)
        return self._freq_dict[lemma]

    @staticmethod
    def _get_number_of_vowels(target):
        """Retrieve frequency of a word from a frequency distribution."""
        return sum(1 for c in target.text if c in GERMAN_VOWELS)

    @staticmethod
    def _get_number_of_syllables(target):
        """Retrieve frequency of a word from a frequency distribution."""
        return sum([len(SYLL_TOKENIZER.tokenize(t.text)) for t in target])

    @staticmethod
    def _get_number_of_characters(target):
        """Number of characters in a given target expression."""
        return sum([len(t.text) for t in target])

    def train(self, instances, labels):
        """Fitting classifier with instance vectors and labels.

        Args:
            instances (interable[feature vecs]): iterable of feature vectors
            labels (iterable[int]): iterable of labels (0 for simple and 1 for
                complex)
        """
        self.classifier.fit(instances, labels)

    def create_feature_vector(self, raw_instance, list_feat_names=False):
        """Create feature vector for single instance.

        Args:
            raw_instance (tuple): the instance to be classified as a tuple
                consisting of a sentence (str), a start index within the
                sentence (int), and an end index within the sentence (int),
                constituting the target as a character span
            list_feat_names (bool): whether to return a list of feature names
                along with the feature vector

        Returns:
            feature vector (np.array), feature name list, if list_feat_names is
                set to True; else feature vector (np.array)
        """
        # Retrieve target span from sentence
        target_span = self.get_target_span(raw_instance)
        feats = []
        feat_names = []
        for feat_name, extract_feature in self._features.items():
            # Get feature value
            val = extract_feature(target_span)
            # Append feature name(s) to feature name list
            # And feature values to feature value list
            if isinstance(val, list):
                feat_names += [f"{feat_name}_{str(i)}" for i in range(len(val))]
                feats += val
            else:
                feats.append(val)
                feat_names.append(f"{feat_name}_0")
        # Return items as required
        if list_feat_names:
            return np.array(feats), feat_names
        return np.array(feats)

    def create_feature_vector_batch(self, raw_instances, list_feat_names=False):
        """Apply feature vector generation to batch of instances.

        Args:
            raw_instances (iterable[tuples]): The spans of the
                sentence to be classified as complex (1) or simple (0) in an
                iterable form, represented as tuples consisting of a
                sentence (str), a start index within the sentence (int), and
                an end index within the sentence (int), constituting the target
                as a character span
            list_feat_names (bool): whether to return a list of feature names
                along with the feature vector

        Returns:
            iterable of feature vectors (iterable[np.array]), feature name
                list, if list_feat_names is set to True; else iterable of
                feature vectors (iterable[np.array])
        """
        if list_feat_names:
            # Retrieve feature names along with feature vector for first item
            fv_0, feat_names = self.create_feature_vector(raw_instances[0],
                                                          list_feat_names=True)
            # Create feature vectors for the rest of the instances
            raw_instances = pd.Series(raw_instances[1:])
            fv = raw_instances.progress_apply(self.create_feature_vector)
            return [fv_0] + list(fv), feat_names
        else:
            # Create and return feature vectors for all instances
            raw_instances = pd.Series(raw_instances)
            fv = raw_instances.progress_apply(self.create_feature_vector)
            return list(fv)

    def predict_single_instance(self, target):
        """Predict label for a single target string.

        Args:
            target (str): The word to be classified as complex (1) or
                simple (0)

        Returns:
            int: 1 if target is deemed complex, else 0
        """
        # Go via batch predict
        return self.predict_batch([target])

    def predict_batch(self, targets):
        """Predict label for a batch of target strings.

        Args:
            targets (iterable[str]): The strings to be classified as
                complex (1) or simple (0) in an iterable form

        Returns:
            iterable[int]: 1 if target string is deemed complex, else 0
        """
        return self.classifier.predict(targets)

    def predict_raw_instance(self, raw_instance):
        """Predict label for a single raw span.

        Args:
            raw_instance (tuple): the instance to be classified as a tuple
                consisting of a sentence (str), a start index within the
                sentence (int), and an end index within the sentence (int),
                constituting the target as a character span

        Returns:
            int: 1 if target span is deemed complex, else 0
        """
        # Prepare instance for classification
        instance = self.create_feature_vector(raw_instance)
        return self.predict_single_instance(instance)

    def predict_raw_batch(self, raw_instances):
        """Predict label for a batch of raw target spans.

        Args:
            raw_instances (iterable[tuples]): The spans of the
                sentence to be classified as complex (1) or simple (0) in an
                iterable form, represented as tuples consisting of a
                sentence (str), a start index within the sentence (int), and
                an end index within the sentence (int), constituting the target
                as a character span

        Returns:
            iterable[int]: 1 if target span is deemed complex, else 0
        """
        # Prepare instances for classification
        instances = self.create_feature_vector_batch(raw_instances)
        return self.predict_batch(instances)

    def annotate_text(self, text):
        """Identify complex words in a text."""
        # Prepare feature vectors
        raw_instances = self.prepare_instances_from_text(text)
        feat_vecs = self.create_feature_vector_batch(raw_instances)
        # Predict instances
        pred = self.predict_batch(feat_vecs)
        return raw_instances, pred

    def list_features(self):
        """Print names of all used and unused features to console."""
        print("Currently enabled features:\n\t-",
              "\n\t- ".join(list(self._features.keys())))
        print("Currently disabled features:\n\t-",
              "\n\t- ".join(list(self._unused_features.keys())))

    def enable_feature(self):
        """Move implemented feature from disabled to enabled features."""
        if feat_name in self._unused_features:
            # Move feature to other group
            feat_func = self._unused_features.pop(feat_name)
            self._features[feat_name] = feat_func
        else:
            # Print feedback for user to console
            if feat_name in self._features:
                print(f'Feature "{feat_name}" is already enabled.')
            else:
                print(f'No feature with the name "{feat_name}" implemented.')

    def disable_feature(self, feat_name):
        """Move implemented feature from enabled to disabled features."""
        if feat_name in self._features:
            # Move feature to other group
            feat_func = self._features.pop(feat_name)
            self._unused_features[feat_name] = feat_func
        else:
            # Print feedback for user to console
            if feat_name in self._unused_features:
                print(f'Feature "{feat_name}" is already disabled.')
            else:
                print(f'No feature with the name "{feat_name}" implemented.')

    def __str__(self):
        """Return representation of classifier."""
        return f"Baseline Complex Word Identifier: {self.classifier}"

