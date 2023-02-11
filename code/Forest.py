import pandas as pd
import numpy as np

from DecTree import DecisionTree

from utils import get_subsample


class RandomForest:

    def __init__(self, feature_r: float, metric_fn, alpha: float, bag_r: float,
                 tree_count: int, seed_for_bag_seed_generator: int = None,
                 seed_for_feat_bag_seed_generator: int = None):
        self.feature_r: float = feature_r
        self.metric_fn = metric_fn
        self.alpha: float = alpha
        self.bag_r: float = bag_r
        self.tree_count: int = tree_count

        self.bag_seed_generator = np.random.default_rng(seed_for_bag_seed_generator)
        self.feat_bag_seed_generator = np.random.default_rng(seed_for_feat_bag_seed_generator)
        int_type_info = np.iinfo(np.int32)
        self.min_int = 0
        self.max_int = int_type_info.max

        self.trees = []
        self._init_trees()

    def _init_trees(self):
        for i in range(self.tree_count):
            tree = DecisionTree(self.feature_r, self.metric_fn, self.alpha)
            self.trees.append(tree)

    def train(self, df: pd.DataFrame, class_col: str = "class", missing_val: str = "?"):
        for i in range(self.tree_count):
            tree = self.trees[i]
            seed1 = self.bag_seed_generator.integers(self.min_int, high=self.max_int,
                                                     endpoint=True)
            seed2 = self.feat_bag_seed_generator.integers(self.min_int, high=self.max_int,
                                                          endpoint=True)
            bag = get_subsample(df, self.bag_r, random_state=seed1)
            tree.train(bag, tree.root, class_col=class_col, missing_val=missing_val, random_state=seed2)


if __name__ == "__main__":
    from utils import entropy

    pth = "../data/agaricus-lepiota-training.csv"
    df1 = pd.read_csv(pth)
    df_train = df1.drop(columns="id")  # this is important!
    metric = entropy
    feat_r = 0.25
    alpha = 0.01
    r_forest = RandomForest(feat_r, metric, alpha=alpha, bag_r=0.5, tree_count=10, seed_for_feat_bag_seed_generator=42,
                            seed_for_bag_seed_generator=42)
    r_forest.train(df_train)
    print("done train")
