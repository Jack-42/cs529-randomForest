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


def information_gain(df: pd.DataFrame, attribute: str, metric_fn,
                     class_col: str = "class") -> float:
    def helper(s, s_v):
        return (len(s_v) / len(s)) * metric_fn(s_v[class_col])

    s_impurity = metric_fn(df[class_col])
    vals_a = set(df[attribute])
    gain_sum = 0
    for v in vals_a:
        sv = df[df[attribute] == v]
        gain_sum += helper(df, sv)
    return s_impurity - gain_sum


def get_best_attribute(df: pd.DataFrame, metric_fn,
                       class_col: str = "class") -> str:
    """
    Given a df, return the attribute which gave the highest information gain
    :param df: pandas Dataframe, attributes are columns
    :param metric_fn: function that returns a float
    :param class_col: str, the column where the class label is
    :return: str, the column which gave the highest info gain
    """
    attrs = df.columns.drop([class_col])
    info_gains = attrs.map(lambda a: information_gain(df, a, metric_fn))
    max_idx = np.argmax(info_gains)
    best_atr = attrs[max_idx]
    return best_atr

def chi2_statistic(df_parent: pd.DataFrame, df_child: pd.DataFrame, class_col: str = "class") -> float:
    parent_vals = set(df_parent[class_col])
    parent_counts = []
    child_counts = []
    # manually computing to handle case where child is missing class
    for val in parent_vals:
        parent_counts.append(float(len(df_parent[df_parent[class_col] == val])))
        child_counts.append(float(len(df_child[df_child[class_col] == val])))
    np_p_counts = np.array(parent_counts)
    np_c_counts = np.array(child_counts)
    np_expected_c_counts = (np_p_counts / np.sum(np_p_counts)) * np_c_counts
    return np.sum(((np_expected_c_counts - np_c_counts) ** 2) / np_expected_c_counts)

if __name__ == "__main__":
    pth = "../data/agaricus-lepiota-training.csv"
    df1 = pd.read_csv(pth)
    metric = entropy
    df1 = df1.drop(columns="id")
    print(get_best_attribute(df1, metric))
