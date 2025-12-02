import random
from functools import reduce

import numpy as np
from sklearn.cluster import AgglomerativeClustering


class DataSplitter:
    """Base clase for data spliters that split data into training sets and test sets.
    Child classes should implement at least two methods: __init__ and split.
    Implement generate_splits for efficiency.

    Methods
    -------
    generate_splits: should be the most used method.
    split: generate a single split. Must be implemented by child classes.
    """

    def __init__(self):
        raise NotImplementedError

    def split(self, samples):
        """Generate one split.

        Return:
        -------
        A list of test set samples.
        """
        raise NotImplementedError

    def generate_splits(self, samples, n, **kwargs):  # TODO:  check for repeated splits
        """Generate many splits by calling self.split repeatedly.
        Note that some splitters should implement this method for efficiency.

        Parameters:
        -----------
        samples: list of samples
        n: int. Number of splits to generate.

        Returns:
        --------
        A list of test set samples.
        """
        return [self.split(samples, **kwargs) for _ in range(n)]


class RandomSplitter(DataSplitter):
    """Random data split.

    Parameters
    ----------
    test_set_ratio: float, default=0.2
        Ratio of test set size to the whole data set.

    """

    def __init__(self, test_set_ratio=0.2):
        self.test_set_ratio = test_set_ratio

    def split(self, samples):
        return np.random.choice(
            samples, size=int(len(samples) * self.test_set_ratio), replace=False
        )


class LargeTreeTraverseOOCSplitter(DataSplitter):
    """Out-of-clade data split by iterating tree clades. Applicable to large trees.

    Parameters
    ----------
    tree: ete3.Tree object
    test_set_range: tuple
        Range of test set size ratio.
    single_clades: list or None, default=None
        List of pre-computed single clades for the clade selection step. Precompute this to save time.
    n_max_clade: int, default=2
        Max number of seperate clades in the test set.
    prefer_small_clade: bool, default=False
        If true, single clade test sets are more likely to be selected.
    growth_data: pd.Series
        Binary growth data. Used to when min_zeros and min_ones are larger than 1.
    min_zeros, min_ones: int, default=0
        Minimum sample numbers of zeros/ones in the test set.
    time_out_iter: int or None, default=None
        Maximum iterations when trying to construct a test set.
    """

    def __init__(
        self,
        tree,
        test_set_range=(0.2, 0.3),
        single_clades=None,
        n_max_clade=2,
        prefer_small_clade=False,
        growth_data=None,
        min_zeros=0,
        min_ones=0,
        time_out_iter=None,
    ):
        self.tree = tree
        self.single_clades = single_clades
        self.growth_data = growth_data
        self.test_set_range = test_set_range
        self.min_zeros = min_zeros
        self.min_ones = min_ones

        if (bool(min_ones) or bool(min_zeros)) and growth_data is None:
            raise ValueError(
                "Growth data is required if min_zeros or min_ones is larger than 1. "
            )

        self.prefer_small_clade = prefer_small_clade
        self.n_max_clade = n_max_clade
        if time_out_iter is None:
            self.time_out_iter = len(tree.get_leaves()) * 3
        else:
            self.time_out_iter = time_out_iter

    def split(self, samples, **kwargs):
        if self.single_clades is None:
            self.single_clades = self.compute_single_clades(self.tree, samples)
        i_iter = 0
        while True:
            if self.prefer_small_clade:
                clades = random.sample(
                    self.single_clades[1:], np.random.randint(self.n_max_clade) + 1
                )
            else:
                clades = random.sample(self.single_clades, self.n_max_clade)
            test_samples = reduce(np.union1d, clades)
            if self.is_good_split(test_samples, samples):
                break
            i_iter += 1
            if self.time_out_iter and i_iter > self.time_out_iter:
                raise ValueError("Timeout in finding a good split.")
        return test_samples

    def compute_single_clades(self, tree, samples):
        clade_size_range = (
            int(self.test_set_range[0] * len(samples)),
            int(self.test_set_range[1] * len(samples)),
        )

        tree = self.tree.copy()
        tree.prune(samples, preserve_branch_length=True)
        # tree = tree.copy()
        # tree_cleaned = False
        # while not tree_cleaned:
        #     tree_cleaned = True
        #     for l in tree.get_leaves():
        #         if l.name == '' or l.name not in samples:
        #             l.delete()
        #             tree_cleaned = False

        single_clades = [[]]  # including an empty clade
        for t in tree.traverse():
            if t.is_leaf():
                continue
            else:
                leaves = [l.name for l in t.get_leaves()]
                if len(leaves) > clade_size_range[1]:
                    continue
                else:
                    single_clades.append(leaves)

        return single_clades

    def is_good_split(self, test_samples, samples):
        if (len(test_samples) / len(samples)) < self.test_set_range[0] or (
            len(test_samples) / len(samples)
        ) > self.test_set_range[1]:
            return False
        if self.growth_data is None:
            return True
        if (
            self.min_zeros is not None
            and (self.growth_data[test_samples] == 0).sum() < self.min_zeros
        ):
            return False
        if (
            self.min_ones is not None
            and (self.growth_data[test_samples] == 1).sum() < self.min_ones
        ):
            return False
        return True


class InCladeSplitter(DataSplitter):
    """In-clade data split.

    Parameters
    ----------
    tree: ete3.Tree object
        Tree to traverse.
    test_set_ratio: float, default=0.2
        Ratio of test set size to the whole data set.

    """

    def __init__(self, tree, distance_df, test_set_ratio=0.2):
        self.tree = tree
        self.distance_df = distance_df
        self.test_set_ratio = test_set_ratio

    def split(self, samples):
        distance_df_subset = self.distance_df.loc[samples, samples]
        # NOTE: Alternatively you can use distance_threshold instead of n_clusters
        # Modified simple square root rule: 2 * √(n/2)
        n_clusters = int(2 * np.sqrt(len(samples) / 2))
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="average"
        )
        clustering.fit(distance_df_subset)
        labels = clustering.labels_
        unique_labels = np.unique(labels)
        test_samples = []
        for label in unique_labels:
            cluster_samples = [
                samples[i] for i in range(len(labels)) if labels[i] == label
            ]
            n_test = int(len(cluster_samples) * self.test_set_ratio)
            test_samples.extend(
                np.random.choice(cluster_samples, size=n_test, replace=False)
            )
        return test_samples
