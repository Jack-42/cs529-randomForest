import pandas as pd
import numpy as np

from TreeNode import TreeNode
from utils import get_best_attribute, get_splits
from utils import get_chi2_statistic, get_chi2_critical, only_missing

"""
@author: Jack Ringer, Mike Adams
Date: 2/3/2023
Description:
Contains methods for training a decision tree using the ID3 algorithm.
"""


class DecisionTree:
    def __init__(self, feature_r: float, metric_fn, alpha: float):
        """
        Create a decision tree. Will be initialized w/ empty root node
        :param feature_r: float, use only a subset of features of
            size total_feats * feature_ratio (min 1 feature used) at each split
        :param metric_fn: function->float, metric used to calculate
            information gain
        :param alpha: float, (1-confidence level) for chi-square critical value
        """
        if feature_r > 1.0 or feature_r <= 0:
            raise ValueError("feature_r must be in range (0, 1]")
        self.root = TreeNode()
        self.feature_r = feature_r
        self.metric_fn = metric_fn
        self.alpha = alpha
        self.depth = 0

    def train(self, df: pd.DataFrame, cur_node: TreeNode,
              class_col: str = "class", missing_val="?",
              random_state: int = None, lvl=0,
              max_lvls=1000):
        """
        Method used to train DecisionTree
        :param df: Dataframe, assumes all cols (excluding class_col) are valid
            attributes - make sure to exclude "id" cols
        :param cur_node: TreeNode, the current node in our tree
        :param class_col: str, the column containing target attribute vals
        :param missing_val: str, the attribute value representing missing data
        :param random_state: int, seed for feature bagging
        :param lvl: int, level of tree
        :param max_lvls: int, max level of tree
        :return: TreeNode
        """
        if lvl > self.depth:
            self.depth = lvl
        n_classes = len(set(df[class_col]))
        if n_classes == 1:
            # examples are homogeneous (all + or all -)
            label = df[class_col].iloc[0]
            cur_node.target = label
            return cur_node
        attrs = df.columns.drop([class_col])

        all_missing = len(attrs) == 1 and only_missing(df, list(attrs)[0],
                                                       missing_val)
        if len(attrs) == 0 or all_missing or lvl >= max_lvls:
            # attributes empty, label = most common class label left
            cur_node.target = df[class_col].mode().iloc[0]
            return cur_node

        if random_state is None:
            gen = np.random.default_rng(random_state)
            random_state = gen.integers(0, (2 ** 32) - 1, endpoint=True)

        a = get_best_attribute(df, metric_fn=self.metric_fn,
                               missing_attr_val=missing_val,
                               feature_ratio=self.feature_r,
                               random_state=random_state)
        a_vals = set(df[a])
        if missing_val in a_vals:
            a_vals.remove(missing_val)
        if len(a_vals) == 0:
            # default branch
            nxt_node = TreeNode()
            cur_node.addBranch("default", nxt_node)
            nxt_node.target = df[class_col].mode().loc[0]
            return cur_node

        splits_miss_maj, splits_miss_branch = get_splits(df, a,
                                                         missing_attr_val=missing_val)
        chi2 = get_chi2_statistic(df, pd.Series(splits_miss_branch.values()),
                                  missing_attr_val=missing_val)
        chi2_critical = get_chi2_critical(self.alpha, n_classes,
                                          len(a_vals))
        if chi2 < chi2_critical:
            # split doesn't provide enough useful info
            cur_node.target = df[class_col].mode().loc[0]
            return cur_node

        cur_node.attribute = a

        # default branch
        nxt_node = TreeNode()
        cur_node.addBranch("default", nxt_node)
        nxt_node.target = df[class_col].mode().loc[0]

        random_state_new = random_state
        for vi in a_vals:
            nxt_node = TreeNode()
            cur_node.addBranch(vi, nxt_node)
            examples_vi = splits_miss_maj[vi]
            if len(examples_vi) == 0:
                # examples is empty
                nxt_node.target = df[class_col].mode().loc[0]
            else:
                random_state_new += 1
                if random_state_new == (2 ** 32) - 1:
                    random_state_new = 1
                self.train(examples_vi, nxt_node, lvl=lvl + 1,
                           class_col=class_col, missing_val=missing_val,
                           random_state=random_state_new, max_lvls=max_lvls)
        return cur_node

    def classify(self, df: pd.DataFrame, cur_node: TreeNode,
                 id_col: str = "id", class_col: str = "class",
                 missing_attr_val="?", out=None) -> pd.DataFrame:
        # Keep id column when calling
        if out is None:
            out = df[[id_col]].copy(deep=True)
            out[class_col] = ""
            out = out.set_index(id_col)
        if cur_node.isLeaf():
            out.loc[df[id_col]] = cur_node.target
            return out
        attr = cur_node.attribute
        splits_miss_maj, splits_miss_branch = get_splits(df, attr,
                                                         missing_attr_val=missing_attr_val)

        for val in splits_miss_maj.keys():
            df_val = splits_miss_maj[val]
            if len(df_val) == 0:
                # no examples to classify
                continue
            next_node = cur_node.next(val)
            if next_node is None:
                # branch missing, find default
                next_node = cur_node.next("default")
            out = self.classify(df_val, next_node, id_col=id_col,
                                class_col=class_col, out=out)
        return out


if __name__ == "__main__":
    from utils import entropy

    pth = "../data/agaricus-lepiota-training.csv"
    df1 = pd.read_csv(pth)
    df_train = df1.drop(columns="id")  # this is important!
    metric = entropy
    feat_r = 0.25
    dtree = DecisionTree(feat_r, metric, 0.1)
    dtree.train(df_train, dtree.root)
    print("done train")
    classifications = dtree.classify(df1, dtree.root)
    print(dtree.depth)
    print(classifications.head())
