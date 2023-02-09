import pandas as pd

from TreeNode import TreeNode
from utils import get_best_attribute
from utils import get_chi2_statistic, get_chi2_critical

"""
@author: Jack Ringer, Mike Adams
Date: 2/3/2023
Description:
Contains methods for training a decision tree using the ID3 algorithm.
"""


class DecisionTree:
    def __init__(self, n_features: int, metric_fn, alpha: float):
        """
        Create a decision tree for the RandomForest.
        Will be initialized w/ empty root node
        :param n_features: int, the number of features we allow the tree to
            consider at each split
        :param metric_fn: function->float, metric used to calculate
            information gain
        :param alpha: float, (1 - confidence level) for chi-squared critical
            value. e.g., alpha=0.05 would be a confidence level of 95%
        """
        self.root = TreeNode()
        self.n_features = n_features
        self.metric_fn = metric_fn
        self.alpha = alpha

    def train(self, df: pd.DataFrame, cur_node: TreeNode,
              class_col: str = "class", lvl=0, max_lvls=3):
        """
        Method used to train DecisionTree
        :param df: Dataframe, assumes all cols (excluding class_col) are valid
            attributes - make sure to exclude "id" cols
        :param cur_node: TreeNode, the current node in our tree
        :param class_col: str, the column containing target attribute vals
        :param lvl: (TEMPORARY) int, level of tree
        :param max_lvls: (TEMPORARY) int, max level of tree
        :return: TreeNode
        """
        n_classes = len(set(df[class_col]))
        if n_classes == 1:
            # examples are homogeneous (all + or all -)
            label = df[class_col].iloc[0]
            cur_node.target = label
            return cur_node
        attrs = df.columns.drop([class_col])
        if len(attrs) == 0 or lvl >= max_lvls:
            # attributes empty, label = most common class label left
            cur_node.target = df[class_col].mode().loc[0]
            return cur_node

        a = get_best_attribute(df, metric_fn=self.metric_fn,
                               n_features=self.n_features)
        a_vals = set(df[a])
        splits = {}
        for vi in a_vals:
            examples_vi = df[df[a] == vi].drop(columns=[a])
            splits[vi] = examples_vi
        chi2 = get_chi2_statistic(df, pd.Series(splits.values()))
        chi2_critical = get_chi2_critical(self.alpha, n_classes,
                                          len(a_vals))
        if chi2 < chi2_critical:
            # split doesn't provide enough useful info
            cur_node.target = df[class_col].mode().loc[0]
            return cur_node

        cur_node.attribute = a
        for vi in a_vals:
            nxt_node = TreeNode()
            cur_node.addBranch(vi, nxt_node)
            examples_vi = splits[vi]
            if len(examples_vi) == 0:
                # examples is empty
                nxt_node.target = df[class_col].mode().loc[0]
            else:
                self.train(examples_vi, nxt_node, lvl=lvl + 1)
        return cur_node

    def classify(self, df: pd.DataFrame, cur_node: TreeNode,
                 id_col: str = "id", class_col: str = "class",
                 out=None) -> pd.DataFrame:
        # Keep id column when calling
        if out is None:
            out = df[[id_col, class_col]].copy(deep=True)
            out = out.set_index(id_col)
        if cur_node.isLeaf():
            out.loc[df[id_col]] = cur_node.target
            return out
        attr = cur_node.attribute
        for val in set(df[attr]):
            df_val = df[df[attr] == val]
            if len(df_val) == 0:
                # no examples to classify
                continue
            out = self.classify(df_val, cur_node.next(val), id_col=id_col,
                                class_col=class_col, out=out)
        return out


if __name__ == "__main__":
    from utils import entropy

    pth = "../data/agaricus-lepiota-training.csv"
    df1 = pd.read_csv(pth)
    df_train = df1.drop(columns="id")  # this is important!
    metric = entropy
    total_features = len(df_train.columns) - 1  # - 1 for class col
    print(total_features)
    n_feats = total_features // 3
    print(n_feats)
    dtree = DecisionTree(n_features=n_feats, metric_fn=metric, alpha=0.05)
    dtree.train(df_train, dtree.root, max_lvls=1000)
    print("done train")
    print(dtree.classify(df1, dtree.root).head())
