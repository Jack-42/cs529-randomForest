import pandas as pd

from TreeNode import TreeNode
from utils import get_best_attribute, get_splits
from utils import get_chi2_statistic, get_chi2_critical

"""
@author: Jack Ringer, Mike Adams
Date: 2/3/2023
Description:
Contains methods for training a decision tree using the ID3 algorithm.
"""


class DecisionTree:
    def __init__(self, metric_fn, alpha: float):
        """
        Create a decision tree. Will be initialized w/ empty root node
        :param metric_fn: function->float, metric used to calculate
            information gain
        """
        self.root = TreeNode()
        self.metric_fn = metric_fn
        self.alpha = alpha

    def train(self, df: pd.DataFrame, cur_node: TreeNode,
              class_col: str = "class", missing_attr_val="?", lvl=0, max_lvls=1000):
        """
        Method used to train DecisionTree
        :param df: Dataframe, assumes all cols (excluding class_col) are valid
            attributes - make sure to exclude "id" cols
        :param cur_node: TreeNode, the current node in our tree
        :param class_col: str, the column containing target attribute vals
        :param missing_attr_val: str, the attribute value representing missing data
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

        a = get_best_attribute(df, metric_fn=self.metric_fn, missing_attr_val=missing_attr_val)
        a_vals = set(df[a])
        if missing_attr_val in a_vals:
            a_vals.remove(missing_attr_val)
        splitsWithMissingAsMajority, splitsWithMissingAsBranch = get_splits(df, a, missing_attr_val=missing_attr_val)
        chi2 = get_chi2_statistic(df, pd.Series(splitsWithMissingAsBranch.values()), missing_attr_val=missing_attr_val)
        chi2_critical = get_chi2_critical(self.alpha, n_classes,
                                          len(a_vals))
        if chi2 < chi2_critical:
            # split doesn't provide enough useful info
            cur_node.target = (df[df[a] != missing_attr_val])[class_col].mode().loc[0]
            return cur_node

        cur_node.attribute = a
        for vi in a_vals:
            nxt_node = TreeNode()
            cur_node.addBranch(vi, nxt_node)
            examples_vi = splitsWithMissingAsMajority[vi]
            if len(examples_vi) == 0:
                # examples is empty
                nxt_node.target = (df[df[a] != missing_attr_val])[class_col].mode().loc[0]
            else:
                self.train(examples_vi, nxt_node, lvl=lvl + 1)
        return cur_node

    def classify(self, df: pd.DataFrame, cur_node: TreeNode,
                 id_col: str = "id", class_col: str = "class", 
                 missing_attr_val: str = "?", out=None) -> pd.DataFrame:
        # Keep id column when calling
        if out is None:
            out = df[[id_col]].copy(deep=True)
            out = out.set_index(id_col)
            out[class_col] = ""
        if cur_node.isLeaf():
            out.loc[df[id_col]] = cur_node.target
            return out
        attr = cur_node.attribute
        splitsWithMissingAsMajority, splitsWithMissingAsBranch = get_splits(df, attr, missing_attr_val=missing_attr_val)
        for val in splitsWithMissingAsMajority.keys():
            df_val = splitsWithMissingAsMajority[val]
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
    dtree = DecisionTree(metric, 0.01)
    dtree.train(df_train, dtree.root)
    print("done train")
    classifications = dtree.classify(df1, dtree.root)
    print(classifications.head())