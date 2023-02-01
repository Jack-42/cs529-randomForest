import pandas as pd
import numpy as np


def get_proportions(s: pd.Series) -> pd.Series:
    v_counts = s.value_counts()
    s_len = len(s)
    return v_counts / s_len


def entropy(s: pd.Series) -> float:
    proportions = get_proportions(s)
    ent_vals = proportions * np.log2(proportions)
    ent = -np.sum(ent_vals)
    return ent


def gini(s: pd.Series) -> float:
    return 1.0 - np.sum(get_proportions(s) ** 2)


def misclassification_error(s: pd.Series) -> float:
    return 1.0 - np.max(get_proportions(s))


def information_gain(df: pd.DataFrame, attribute: str, metric_fn) -> float:
    def helper(s, s_v):
        return (len(s_v) / len(s)) * metric_fn(s_v['class'])

    s_impurity = metric_fn(df['class'])
    vals_a = set(df[attribute])
    gain_sum = 0
    for v in vals_a:
        sv = df[df[attribute] == v]
        gain_sum += helper(df, sv)
    return s_impurity - gain_sum


if __name__ == "__main__":
    pth = "../data/agaricus-lepiota-training.csv"
    df = pd.read_csv(pth)
    metric = entropy
    gsa = information_gain(df, 'gill-size', metric)
    print(gsa)
