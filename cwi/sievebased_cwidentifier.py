# -*- coding: utf-8 -*-

"""
Complex Word Identification on German using rules of "Leichte Sprache"-guides.
"""

import pandas as pd

from tqdm import tqdm

from .abstract_cwidentifier import AbstractCWIdentifier
from .sieves.sieves import *
from .util.errors import NoClassifierProvidedError
from .util.ling_processing import lemmatize


# tqdm setup
tqdm.pandas(desc="Feature Vector Generation")


class SievebasedCWIdentifier(AbstractCWIdentifier):
    """Sieve-based Complex Word Identifier. Designed with German data in mind.

    This classifier uses a collection of sieves to identify complex words in a
    sentence. A word is labeled as complex (1) or simple (0), if it is caught
    by any of the sieves.

    The sieves enabled by default are:
        - length
        - is_nominalization
        - contains_punctuation_character
        - is_textbased_abbreviation
        - is_numberword
        - in_vocab

    Implemented, but per default not enabled are:
        - number_of_characters
        - number_of_syllables
        - frequency
        - vocab_filtered_frequency

    Args:
        spacy_model (spacy.lang.de.German): spacy model to be used for
            preprocessing (should be a German model, as this classifier was
            designed for German)
        lemma_dict (dict): a dictionary for assigning correct lemmata to word
            forms; if a target is not in the dictionary a simple rule-based
            fallback method will be used

    Attributes:
        frequency_threshold (int): threshold for labeling words as rare
        vocab (set): set of words deemed as generally known, ideally in a
            lemmatized form
        familiarity_classifier (sklearn.pipeline.Pipeline): classifier for
            frequency-based familiarity classification
        character_threshold (int): threshold for labeling words as long in
            character count
        syllable_threshold (int): threshold for labeling words as long in
            syllable count
        lemmatize_fallback (function): simple lemmatizing function to use as a
            fallback forwords not found in the lemma dictionary, default here
            is for German
    """

    def __init__(self,
                 spacy_model,
                 lemma_dict={}):
        self._spacy_model = spacy_model
        # Sentence dict for more efficient preprocessing
        self._sent_dict = {}
        # Tools for frequency-based sieves
        self.frequency_dict = {}
        self.vocab = {}
        self.familiarity_classifier = None
        # Thresholds
        self.frequency_threshold = 1
        self.character_threshold = 10
        self.syllable_threshold = 2
        # Lemmatization assets
        self._lemma_dict = lemma_dict | {v: v for v in lemma_dict.values()}
        self.lemmatize_fallback = lemmatize
        # Enabled sieves
        self._sieves = {
            "length": self._get_length,
            "is_nominalization": is_nominalization,
            "contains_punctuation_characters": contains_punctuation_characters,
            "is_textbased_abbreviation": is_textbased_abbreviation,
            "identify_numberwords": identify_numberwords,
            "frequency": self._get_frequency
        }
        # Disabled sieves
        self._unused_sieves = {
            "number_of_characters": self._get_number_of_characters,
            "number_of_syllables": self._get_number_of_syllables,
            "familiarity_classification": self._classify_familiarity,
            "vocab_filtered_frequency": self._get_vocab_filtered_frequency,
            "in_vocab": self._is_in_vocab
        }

    def _lemmatize(self, target):
        """Lemmatize a target by lookup in dict with rule-based fallback."""
        # Use lemma from dict if possible, else lemmatize by fallback function
        return self._lemma_dict.get(target.text,
                                    self.lemmatize_fallback(target.text))

    def _get_length(self, target):
        """Check if both character and syllable length exceed thresholds."""
        # Combined version of length measurements:
        # Sieve returns True if both character and syllable length return True
        return all([number_of_characters(target, self.character_threshold),
                    number_of_syllables(target, self.syllable_threshold)])

    def _get_number_of_characters(self, target):
        """Check if character length exceeds threshold."""
        # If number of characters is higher than threshold: sieve returns True
        return number_of_characters(target, self.character_threshold)

    def _get_number_of_syllables(self, target):
        """Check if syllable length exceeds threshold."""
        # If number of syllables is higher than threshold: sieve returns True
        return number_of_syllables(target, self.syllable_threshold)

    def _get_frequency(self, target):
        """Check if frequency of target is lower than threshold."""
        lemma = self._lemmatize(target).lower()
        # If word count is lower than threshold: sieve returns True
        return self.frequency_dict.get(lemma, 0) < self.frequency_threshold

    def _is_in_vocab(self, target):
        """Check if target is in predefined vocabulary list."""
        lemma = self._lemmatize(target).lower()
        # Sieve returns True if word not in core vocabulary
        return lemma not in self.vocab

    def _classify_familiarity(self, target):
        """Check if provided classifier deems word unknown."""
        if self.familiarity_classifier is None:
            raise NoClassifierProvidedError("Please provide a classifier for "
                                            "the classify_familiarity sieve.")
        lemma = self._lemmatize(target).lower()
        # Get prediction via classifier
        pred = self.familiarity_classifier.predict_single_instance(lemma,
                                                                   is_str=True)
        return bool(pred)

    def _get_vocab_filtered_frequency(self, target):
        """Check frequency of targets that are not in the vocabulary list."""
        lemma = self._lemmatize(target).lower()
        # Words from the core vocabulary are are deemed simple
        # Even if they are not present in the referenced frequency distribution
        if lemma in self._vocab:
            return False
        # If word count is lower than threshold: sieve returns True
        return self._freq_dict.get(lemma, 0) < self.frequency_threshold

    def predict_single_instance(self, target_span, return_sieves=False):
        """Predict label for a single spacy target span.

        Args:
            target_span (spacy.tokens.span.Span): The span of the sentence to
                be classified as complex (1) or simple (0)
            return_sieves (bool): whether to return the sieves that returned
                "True", default: False

        Returns:
            int: 1 if target span is deemed complex, else 0
        """
        # Punctuation might be caught by some sieves
        # Therefore it is manually excluded here
        if len(target_span) == 1 and target_span[0].pos_ == "PUNCT":
            if return_sieves:
                return 0, []
            return 0
        if return_sieves:
            # Collect sieves returning True for instance
            sieves = [sieve_name for sieve_name, sieve in self._sieves.items()
                      if sieve(target_span)]
            # Derive classification result from sieve list for instance
            return int(bool(sieves)), sieves
        # Return 1 if any of the sieves returned True, else 0
        return int(any(sieve(target_span) for sieve in self._sieves.values()))

    def predict_batch(self, target_spans, return_sieves=False):
        """Predict label for a batch of spacy target spans.

        Args:
            target_spans (iterable[spacy.tokens.span.Span]): The spans of the
                sentence to be classified as complex (1) or simple (0) in an
                iterable form
        return_sieves (bool): whether to return the sieves that returned
            "True", default: False

        Returns:
            list[int]: 1 if target span is deemed complex, else 0
        """
        # Create and return list of labels for instances
        return [self.predict_single_instance(target_span, return_sieves)
                for target_span in target_spans]

    def predict_raw_instance(self, raw_instance, return_sieves=False):
        """Predict label for a single raw span.

        Args:
            raw_instance (tuple): the instance to be classified as a tuple
                consisting of a sentence (str), a start index within the
                sentence (int), and an end index within the sentence (int),
                constituting the target as a character span
            return_sieves (bool): whether to return the sieves that returned
                "True", default: False

        Returns:
            int: 1 if target span is deemed complex, else 0
        """
        # Create spacy span from character span in sentence
        target_span = self.get_target_span(raw_instance)
        return self.predict_single_instance(target_span, return_sieves)

    def predict_raw_batch(self, raw_instances, return_sieves=False):
        """Predict label for a batch of raw target spans.

        Args:
            raw_instances (iterable[tuples]): The spans of the
                sentence to be classified as complex (1) or simple (0) in an
                iterable form, represented as tuples consisting of a
                sentence (str), a start index within the sentence (int), and
                an end index within the sentence (int), constituting the target
                as a character span
            return_sieves (bool): whether to return the sieves that returned
                "True", default: False

        Returns:
            list[int]: 1 if target span is deemed complex, else 0
        """
        # Create spacy spans for list of character spans
        raw_instances = pd.Series(raw_instances)
        spans = raw_instances.progress_apply(self.get_target_span)
        return self.predict_batch(spans, return_sieves)

    def annotate_text(self, text, return_sieves=False):
        """Annotate raw text, to be implemented."""
        instances = self.prepare_instances_from_text(text)
        pred = self.predict_raw_batch(instances, return_sieves)
        return instances, pred

    def list_sieves(self):
        """Print names of all used and unused sieves to console."""
        print("Currently enabled sieves:\n\t-",
              "\n\t- ".join(list(self._sieves.keys())))
        print("Currently disabled sieves:\n\t-",
              "\n\t- ".join(list(self._unused_sieves.keys())))

    def enable_sieve(self, sieve_name):
        """Move implemented sieve from disabled to enabled sieves."""
        if sieve_name in self._unused_sieves:
            # Move sieve to other group
            sieve_func = self._unused_sieves.pop(sieve_name)
            self._sieves[sieve_name] = sieve_func
        else:
            # Print feedback for user to console
            if sieve_name in self._sieves:
                print(f'Feature "{sieve_name}" is already enabled.')
            else:
                print(f'No sieve with the name "{sieve_name}" implemented.')

    def disable_sieve(self, sieve_name):
        """Move implemented sieve from enabled to disabled sieves."""
        if sieve_name in self._sieves:
            # Move sieve to other group
            sieve_func = self._sieves.pop(sieve_name)
            self._unused_sieves[sieve_name] = sieve_func
        else:
            # Print feedback for user to console
            if sieve_name in self._unused_sieves:
                print(f'Feature "{sieve_name}" is already disabled.')
            else:
                print(f'No sieve with the name "{sieve_name}" implemented.')

    def __str__(self):
        """Return representation of self stating currently set thresholds."""
        strings = [
            "-"*50,
            "Rulebased Context Word Identifier\n",
            f"Sieves:\n",
            "".join([f" - {sieve}\n" for sieve in self._sieves.keys()]),
            "Thresholds:",
            " - frequency: " + str(self.frequency_threshold),
            " - number of characters: " + str(self.character_threshold),
            " - number of syllables: " + str(self.syllable_threshold),
            "-"*50
        ]
        return "\n".join(strings)

