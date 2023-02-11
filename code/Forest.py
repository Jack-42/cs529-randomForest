import pandas as pd
import numpy as np

from DecTree import DecisionTree

from utils import get_subsample

class RandomForest:

    def __init__(self, feature_r: float, metric_fn, alpha: float, bag_r: float, 
                 tree_count: int, seed_for_bag_seed_generator: int = None, seed_for_feat_bag_seed_generator: int = None):
        self.feature_r: float = feature_r
        self.metric_fn = metric_fn
        self.alpha: float = alpha
        self.bag_r: float = bag_r
        self.tree_count: int = tree_count

        self.bag_seed_generator = np.random.default_rng(seed_for_bag_seed_generator)
        self.feat_bag_seed_generator = np.random.default_rng(seed_for_feat_bag_seed_generator)
        int_type_info = np.iinfo(np.int64)
        self.min_int = int_type_info.min
        self.max_int = int_type_info.max

        self.trees = []

    def train(self, df: pd.DataFrame, class_col: str = "class", missing_val="?"):
        for i in range(self.tree_count):
            bag = get_subsample(df, self.bag_r, random_state=self.bag_seed_generator.integers(self.min_int, high=self.max_int, endpoint=True))
            tree = DecisionTree(self.feature_r, self.metric_fn, self.alpha)
            tree.train(bag, tree.root, class_col=class_col, missing_val=missing_val, random_state=self.feat_bag_seed_generator.integers(self.min_int, high=self.max_int, endpoint=True))
            self.trees.append(tree)
