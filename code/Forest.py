import pandas as pd
import numpy as np

from DecTree import DecisionTree

from utils import get_subsample


class RandomForest:

    def __init__(self, feature_r: float, metric_fn, alpha: float, bag_r: float,
                 tree_count: int, seed_for_bag_seed_generator: int = None,
                 seed_for_feat_bag_seed_generator: int = None):
        """
        Create a random forest with multi-tree majority voting classification. 
        :param feature_r: float, use only a subset of features of
            size total_feats * feature_ratio (min 1 feature used) at each split
        :param metric_fn: function->float, metric used to calculate
            information gain
        :param alpha: float, (1-confidence level) for chi-square critical value
        :param bag_r: float, how much (%) of the data each tree trains on
        :param tree_count: int, number of trees in the forest
        :param seed_for_bag_seed_generator: int, seed to make random seeds for bagging
        :param seed_for_feat_bag_seed_generator: int, seed to make random seeds for feature bagging per node
        """
        self.feature_r: float = feature_r
        self.metric_fn = metric_fn
        self.alpha: float = alpha
        self.bag_r: float = bag_r
        self.tree_count: int = tree_count

        self.bag_seed_generator = np.random.default_rng(
            seed_for_bag_seed_generator)
        self.feat_bag_seed_generator = np.random.default_rng(
            seed_for_feat_bag_seed_generator)
        int_type_info = np.iinfo(np.int32)
        self.min_int = 0
        self.max_int = int_type_info.max

        self.trees = []
        self._init_trees()

    def _init_trees(self):
        for i in range(self.tree_count):
            tree = DecisionTree(self.feature_r, self.metric_fn, self.alpha)
            self.trees.append(tree)

    def avg_tree_depth(self) -> float:
        sum = 0
        for t in self.trees:
            sum += t.depth
        return sum / len(self.trees)

    def max_tree_depth(self) -> int:
        max = -1
        for t in self.trees:
            if t.depth > max:
                max = t.depth
        return max

    def most_common_top_feature(self) -> str:
        d = {}
        for t in self.trees:
            top = t.top_feature
            if top in d:
                d[top] = 1 + d[top]
            else:
                d[top] = 1
        max_feat = ""
        max_count = 0
        for feat in d:
            if d[feat] > max_count:
                max_count = d[feat]
                max_feat = feat
        return max_feat

    def train(self, df: pd.DataFrame, class_col: str = "class",
              missing_val: str = "?"):
        for i in range(self.tree_count):
            tree = self.trees[i]
            seed1 = self.bag_seed_generator.integers(self.min_int,
                                                     high=self.max_int,
                                                     endpoint=True)
            seed2 = self.feat_bag_seed_generator.integers(self.min_int,
                                                          high=self.max_int,
                                                          endpoint=True)
            bag = get_subsample(df, self.bag_r, random_state=seed1)
            tree.train(bag, tree.root, class_col=class_col,
                       missing_val=missing_val, random_state=seed2)

    def classify(self, df: pd.DataFrame, id_col: str = "id",
                 class_col: str = "class", missing_attr_val="?"):
        preds = []
        for tree in self.trees:
            preds.append(
                tree.classify(df, tree.root, id_col=id_col, class_col=class_col,
                              missing_attr_val=missing_attr_val))
        ids = set(df[id_col])
        out = df[[id_col]].copy(deep=True)
        out[class_col] = ""
        out = out.set_index(id_col)

        def max_label(cs):
            max_count = 0
            max_lab = ""
            for k in cs.keys():
                if cs[k] > max_count:
                    max_count = cs[k]
                    max_lab = k
            return max_lab

        for id in ids:
            counts = {}
            for pred_df in preds:
                v = pred_df.loc[id][0]
                if v in counts:
                    counts[v] = counts[v] + 1
                else:
                    counts[v] = 1
            maj_label = max_label(counts)
            out.loc[id] = maj_label

        return out


if __name__ == "__main__":
    from utils import entropy

    pth = "../data/agaricus-lepiota-training.csv"
    df1 = pd.read_csv(pth)
    df_train = df1.drop(columns="id")
    feat_r = 0.01
    metric = entropy
    alpha = 0.01
    bag_r = 0.4
    tree_count = 10
    seed_for_bag_seed_generator = None  # random
    seed_for_feat_bag_seed_generator = None  # random
    rf = RandomForest(feat_r, metric, alpha, bag_r, tree_count,
                      seed_for_bag_seed_generator,
                      seed_for_feat_bag_seed_generator)
    rf.train(df_train)
    classifications = rf.classify(df1)
    print(classifications.head())

    pth2 = "../data/agaricus-lepiota-testing.csv"
    dftest = pd.read_csv(pth2)
    classifications_test = rf.classify(dftest)
    print(classifications_test.head())
