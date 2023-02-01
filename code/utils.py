import pandas as pd
import numpy as np


def get_proportions(s: pd.Series) -> pd.Series:
    v_counts = s.value_counts()
    s_len = len(s)
    return v_counts / s_len


def entropy(s: pd.Series) -> float:
    proportions = get_proportions(s)
    ent_vals = proportions.map(lambda x: x * np.log2(x))
    ent = -np.sum(ent_vals)
    return ent


def information_gain(df: pd.DataFrame, attribute: str, metric_fn) -> float:
    def helper(s, s_v):
        return (len(s) / len(s_v)) * metric_fn(s_v['class'])

    s_impurity = metric_fn(df['class'])
    vals_a = set(df[attribute])
    gain_sum = 0
    for v in vals_a:
        sv = df[df[attribute] == v]
        gain_sum += helper(df, sv)
    return s_impurity - gain_sum
