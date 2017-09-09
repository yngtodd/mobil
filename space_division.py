"""
Divide each hyperparameter interval into subintervals to be distributed across
many machines.

If every hyperparameter interval has at least 4 elements, we ultimately will have
2 ** N combinations of hyperparmeter spaces to search over where N is the number of
hyperparameters.
"""
import numpy as np
from math import floor
from collections import OrderedDict
from collections import defaultdict
import itertools
from itertools import product
from skopt.space import Space


class IntervalLengthError(Exception):
    pass


class HyperSpace(object):
    def __init__(self, hyperparams):
        self.hyperparams = hyperparams

    def _subset_hparams(self, min_seq_len):
        """
        Separates away intervals smaller than min_seq_len to prevent them from
        being split into too small of subintervals.

        Keyword arguments:
        hparam_dict - dictionary of hyperparameters (string): search space (list)
        Returns:
        short_intervals - dictionary of intervals too short to split into subintervals
        long_intervals - dictionary of intervals to be split into subintervals
        """
        ordered = OrderedDict(self.hyperparams)
        short_intervals = {key: value for key, value in ordered.items() if len(value) < min_seq_len}
        long_intervals = {key: value for key, value in ordered.items() if len(value) >= min_seq_len}
        return short_intervals, long_intervals

    def _sep_keys_values(self, long_intervals):
        """
        Orders the hyperparameter dictionary and returns keys, values as lists

        Keyword arguments:
        long_intervals - dictionary returned by subset_hparams
        Returns:
        keys - keys of long_intervals
        values - values of long_intervals
        """
        ordered = OrderedDict(long_intervals)
        keys, values = zip(*ordered.items())
        return keys, values

    def _split_interval(self, hparam_interval, overlap):
        """
        Divides the original hyperparameter interval into two subintervals with
        a given amount of overlap.

        Keyword arguments:
        hparam_interval -- original hyperparameter interval to be searched over
        overlap -- percent overlap between the resulting subintervals
        Returns:
        subseq_low - lower half of hyperpameter interval with given overlap of higher interval
        subseq_hi - higher half of hyperparamter interval with given overlap of lower interval
        """
        hparam_interval = list(hparam_interval)
        hparam_interval.sort()
        seq_len = len(hparam_interval)

        if seq_len < 4:
            raise IntervalLengthError("Hyperparameter interval to be divided must have\
                                      length greater than 3!")

        halfway = seq_len//2
        seq_overlap = floor(seq_len * overlap)//2

        subseq_low = hparam_interval[:halfway+seq_overlap]
        subseq_hi = hparam_interval[halfway-seq_overlap:]
        return subseq_low, subseq_hi

    def _merge_dicts(self, *dict_args):
        """
        Given any number of dictionaries, shallow copy and merge into a new dict,
        precedence goes to key value pairs in latter dicts.
        """
        result = {}
        for dictionary in dict_args:
            result.update(dictionary)
        return result

    def fold_space(self, min_seq_len=4, overlap=0.50):
        """
        Primary force of the class. Splits any hyperparameter interval longer than
        four elements with a given amount of overlap between intervals.

        Keyword arguments:
        overlap - percent overlap between hyperparameters
        Returns:
        all_intervals - dictionary of all hyperparameter subintervals
        """

        short_intervals, long_intervals = self._subset_hparams(min_seq_len)
        long_keys, long_values = self._sep_keys_values(long_intervals)
        subseq_low, subseq_high = zip(*[self._split_interval(x, overlap) for x in long_values])

        keys_lo = (str(x) + '_lo' for x in long_keys)
        keys_lo = tuple(keys_lo)
        keys_hi = (str(x) + '_hi' for x in long_keys)
        keys_hi = tuple(keys_hi)

        intervals_low = dict(zip(keys_lo, subseq_low))
        intervals_high = dict(zip(keys_hi, subseq_high))

        all_intervals = self._merge_dicts(intervals_low, intervals_high, short_intervals)
        return all_intervals

    def hyper_permute(self, all_intervals):
        """
        Creates all possible combinations of hyperparameter intervals to be
        distributed.

        Keyword arguments:
        all_intervals - dictionary of all hyperparameter subintervals (fold_space)
        Returns:
        hyperspaces - list of dictionaries: all possible combinations of hyperparameter subintervals
        """
        pool = defaultdict(list)
        for key in all_intervals:
            base = key.split('_')[0]
            pool[base].append(key)

        hyperspaces = [{key: all_intervals[key] for key in keys} for keys in product(*pool.values())]
        print('Number of hyperparameter spaces to be distributed: {}'.format(len(hyperspaces)))
        return hyperspaces

    def format_hyperspace(self, hyperspaces):
        """
        Formats each hyperparameter subspace for scikit-optimize.

        Keyword arguments:
        hyperspaces - dictionary of hyperparameter subspaces returned from hyper_permute()
        Returns:
        subspace_keys - keys of each hyperparameter subspace as a list of list (usefule for MPI scatter)
        subspace_boundaries - lower and upper bounds for scikit-opt's 'space'. Each hyperparameter
        boundary is represented as a tuple, each space is a list of tuples, and finally is wrapped
        in a list to make use of MPI's scatter method.
        """
        subspace_keys = [list(x.keys()) for x in hyperspaces]
        subspace_values = [list(x.values()) for x in hyperspaces]

        subspace_boundaries = []
        for i in range(len(subspace_values)):
            subspace_boundaries.append([(min(i), max(i)) for i in subspace_values[i]])

        hyperspace_bounds = []
        for i in range(len(subspace_boundaries)):
            hyperspace_bounds.append(Space(subspace_boundaries[i]))

        return subspace_keys, hyperspace_bounds
