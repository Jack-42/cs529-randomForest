import pandas as pd
from utils import get_best_attribute
from TreeNode import TreeNode

"""
@author: Jack Ringer, Mike Adams
Date: 2/3/2023
Description:
Contains methods for training a decision tree using the ID3 algorithm.
"""


class DecisionTree:
    def __init__(self, metric_fn):
        """
        Create a decision tree. Will be initialized w/ empty root node
        :param metric_fn: function->float, metric used to calculate
            information gain
        """
        self.root = TreeNode()
        self.metric_fn = metric_fn

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
        if len(set(df[class_col])) == 1:
            # examples are homogeneous (all + or all -)
            label = df[class_col].iloc[0]
            cur_node.target = label
        attrs = df.columns.drop([class_col])
        if len(attrs) == 0 or lvl >= max_lvls:
            # attributes empty, label = most common class label left
            cur_node.target = df[class_col].mode().loc[0]
            return cur_node

        a = get_best_attribute(df, metric_fn=self.metric_fn)
        cur_node.attribute = a
        a_vals = set(df[a])
        for vi in a_vals:
            nxt_node = TreeNode()
            cur_node.addBranch(vi, nxt_node)
            examples_vi = df[df[a] == vi].drop(columns=[a])
            if len(examples_vi) == 0:
                # examples is empty
                nxt_node.target = df[class_col].mode().loc[0]
            else:
                self.train(examples_vi, nxt_node, lvl=lvl + 1)
        return cur_node

    def classify(self, df: pd.DataFrame, cur_node: TreeNode, 
                 id_col: str = "id", class_col: str = "class", out=None) -> pd.DataFrame:
        # Keep id column when calling
        if out == None:
            out = df[id_col].copy(deep=True)
        if cur_node.isLeaf():
            for id in df[id_col]:
                out[out[id_col] == id][class_col] = cur_node.target
            return out
        attr = cur_node.attribute
        for val in set(df[attr]):
            df_val = df[df[attr] == val]
            if len(df_val) == 0:
                # no examples to classify
                continue
            out = self.classify(df_val, cur_node.next(val), id_col=id_col, class_col=class_col, out=out)
        return out

if __name__ == "__main__":
    from utils import entropy

    pth = "../data/agaricus-lepiota-training.csv"
    df1 = pd.read_csv(pth)
    df1 = df1.drop(columns="id")  # this is important!
    metric = entropy
    dtree = DecisionTree(metric)
    dtree.train(df1, dtree.root)
    print(dtree.root.branches)  # best attribute is odor, keys are vals
