# -*- coding: utf-8 -*-

"""
Frequency-based classifier to be used as a sieve in the sieve-based approach.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .baseline import BaselineFeatureBasedCWIdentifier


class FrequencyDistCWIdentifier(BaselineFeatureBasedCWIdentifier):
    """Classifier using word frequency counts and unigram probabilities.

    This classifier is meant to be used as a sieve in the sieve-based
    approach.

    Args:
        frequency_distributions (nltk.probability.FreqDist): list of frequency
            distributions to be used

    Attributes:
        classifier (classifier): model that can be fit and used to predict,
            default is a pipeline of sklearn's StandardScaler and
            NearestCentroid classifier
    """

    def __init__(self, frequency_distributions):
        self._fdists = frequency_distributions
        self._features = {
            "frequency_counts": self._get_freq_counts,
            "unigram_probabilities": self._get_unigram_prob
        }
        self._unused_features = {}
        self.classifier = make_pipeline(StandardScaler(),
                                        RandomForestClassifier(n_estimators=100,
                                                               random_state=666))

    def _get_freq_counts(self, target):
        """Get raw frequency count from wikipedia frequency distribution."""
        return [fdist[target] for fdist in self._fdists]

    def _get_unigram_prob(self, target):
        """Get raw frequency count from wikinews frequency distribution."""
        # .freq(...) actually is unigram probability in nltk's FreqDist
        return [fdist.freq(target) for fdist in self._fdists]

    def create_feature_vector(self, raw_instance, is_str=False):
        """Create feature vector for single instance.

        Args:
            raw_instance (tuple): the instance to be classified as a tuple
                consisting of a sentence (str), a start index within the
                sentence (int), and an end index within the sentence (int),
                constituting the target as a character span
            is_str (bool): as this classifier is meant to be used as a sieve
                in another classifier, it allows strings as raw instances

        Returns:
            list: list of feature vectors
        """
        if not is_str:
            raw_instance = raw_instance[0][raw_instance[1]:raw_instance[2]]
        return [v for extract_feature_values in self._features.values()
                for v in extract_feature_values(raw_instance)]

    def create_feature_vector_batch(self, raw_instances):
        """Apply feature vector generation to batch of instances.

        Args:
            raw_instances (iterable[tuples]): The spans of the
                sentence to be classified as complex (1) or simple (0) in an
                iterable form, represented as tuples consisting of a
                sentence (str), a start index within the sentence (int), and
                an end index within the sentence (int), constituting the target
                as a character span

        Returns:
            iterable[list]): list of feature vectors
        """
        return [self.create_feature_vector(i) for i in raw_instances]

    def predict_single_instance(self, instance, is_str=False):
        """Get prediction for a single instance, can be str or tuple.

        Args:
            instance (str/tuple): if instance is string, the target to be
                classified; if instance is tuple, the target to be classified
                in the format (sentence, start_index, end_index)
            is_str (bool): whether instance format is string (if is_str is set
                to False, a tuple of the aforementioned format will be
                expected)

        Returns:
            int: 1 if classifier judges word to be complex, else 0
        """
        fv = self.create_feature_vector(instance, is_str)
        return self.classifier.predict([fv])[0]

    def __str__(self):
        """Return representation of classifier."""
        return f"Corpus-distribution-based Complex Word Sieve: {self.classifier}"

