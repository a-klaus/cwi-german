# -*- coding: utf-8 -*-

"""Template for all Complex Word Identifiers."""

from abc import ABC, abstractmethod

from .util.ling_processing import ne_retokenize


class AbstractCWIdentifier(ABC):
    """Abstract Complex Word Identifier as a base for all my CWIdentifiers.

    This abstract complex word identifier combines abstract methods, which have
    to be implemented by all classifiers and general methods that are needed by
    all classifiers for this task, which can therefore be inherited.

    Args:
        spacy_model (spacy.lang.de.German): spacy model to be used for
            preprocessing (should be a German model, as this classifier was
            designed for German)
    """

    def __init__(self, spacy_model=None):
        self._spacy_model = spacy_model
        self._sent_dict = {}

    def get_target_span(self, raw_instance):
        """Convert target (character) span to spacy span.

        Args:
            raw_instance (tuple): the instance to be classified as a tuple
                consisting of a sentence (str), a start index within the
                sentence (int), and an end index within the sentence (int),
                constituting the target as a character span

        Returns:
            spacy.tokens.span.Span: the span of the sentence as a spacy span

        Raises:
            ValueError: if no spacy model is provided
        """
        # Check if spacy model is there
        if self._spacy_model is None:
            raise ValueError("No spacy model provided")
        # Create (or retrieve) spacy doc for sentence
        if raw_instance[0] in self._sent_dict:
            doc = self._sent_dict[raw_instance[0]]
        else:
            doc = self._spacy_model(raw_instance[0])
            self._sent_dict[raw_instance[0]] = doc
        # Get span with target
        target_span = doc.char_span(raw_instance[1], raw_instance[2])
        # If unable to align character indices with spacy tokenization
        if target_span is None:
            start_i, end_i = raw_instance[1], raw_instance[2]
            # Find token that contains character index
            for token in doc:
                start = doc[token.i:token.i+1].start_char
                end = doc[token.i:token.i+1].end_char
                if raw_instance[1] >= start and raw_instance[1] <= end:
                    start_i = start
                if raw_instance[2] >= start and raw_instance[2] <= end:
                    end_i = end
            target_span = doc.char_span(start_i, end_i)
        return target_span

    def prepare_instances_from_text(self, text):
        """Create instances from a text.

        Args:
            text (str): the text to be prepared

        Returns:
            list: the raw instances from the text
        """
        if isinstance(text, str):
            # Tokenize
            text = self._spacy_model(text)
            # text = ne_retokenize(text)
        # Prepare instances
        return [
            (t.sent.text,
             text[t.i:t.i+1].start_char - t.sent.start_char,
             text[t.i:t.i+1].end_char - t.sent.start_char)
            for t in text
        ]

    @abstractmethod
    def predict_single_instance(self):
        pass

    @abstractmethod
    def predict_batch(self):
        pass

    @abstractmethod
    def predict_raw_instance(self):
        pass

    @abstractmethod
    def predict_raw_batch(self):
        pass

    @abstractmethod
    def annotate_text(self):
        pass

