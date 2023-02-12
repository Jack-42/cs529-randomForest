import pandas as pd
import numpy as np

from scipy import stats as scistats


def get_proportions(s: pd.Series) -> pd.Series:
    v_counts = s.value_counts()
    return v_counts / len(s)


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
                     class_col: str = "class",
                     missing_attr_val: str = "?") -> float:
    def helper(s, s_v):
        return (len(s_v) / len(s)) * metric_fn(s_v[class_col])

    # ignore rows with ?'s in column
    dfWithoutMissing = df[df[attribute] != missing_attr_val]

    s_impurity = metric_fn(dfWithoutMissing[class_col])
    vals_a = set(dfWithoutMissing[attribute])
    gain_sum = 0.0
    for v in vals_a:
        sv = dfWithoutMissing[dfWithoutMissing[attribute] == v]
        gain_sum += helper(df, sv)
    return s_impurity - gain_sum


def get_best_attribute(df: pd.DataFrame, metric_fn,
                       class_col: str = "class",
                       missing_attr_val: str = "?",
                       feature_ratio: float = None,
                       random_state: int = None) -> str:
    """
    Given a df, return the attribute which gave the highest information gain
    :param df: pandas Dataframe, attributes are columns
    :param metric_fn: function that returns a float
    :param class_col: str, the column where the class label is
    :param missing_attr_val: str, the attribute value representing missing data
    :param feature_ratio: (optional) float, use only a subset of features of
        size total_feats * feature_ratio (min 1 feature used)
    :param random_state: (optional) int, random seed for reproducibility
    :return: str, the column which gave the highest info gain
    """
    attrs = df.columns.drop([class_col])
    if feature_ratio is not None:
        n_feats = max(1, int(len(attrs) * feature_ratio))
        attrs = attrs.to_series().sample(n=n_feats, random_state=random_state)
    info_gains = attrs.map(lambda a: information_gain(df, a, metric_fn,
                                                      missing_attr_val=missing_attr_val))
    max_idx = np.argmax(info_gains)
    best_atr = attrs[max_idx]
    if only_missing(df, best_atr, missing_attr_val) and len(attrs) != 1:
        # selected attribute has only meaningless values, choose another
        info_gains[max_idx] = 0.0
        new_max_idx = np.argmax(info_gains)
        best_atr = attrs[new_max_idx]
    return best_atr


def get_subsample(df: pd.DataFrame, ratio: float, random_state: int = None):
    """
    Get subsample from dataset
    :param df: pd.Dataframe, dataset being sampled
    :param ratio: float, fraction of data to use in sample
    :param random_state: (optional) int, random seed for reproducibility
    :return: pd.Dataframe, the subsample
    """
    return df.sample(frac=ratio, replace=True, random_state=random_state)


def get_splits(df: pd.DataFrame, attribute: str, missing_attr_val: str = "?"):
    splitsWithMissingAsMajority = {}
    splitsWithMissingAsBranch = {}

    dfWithoutMissing = df[df[attribute] != missing_attr_val]
    dfMissing = df[df[attribute] == missing_attr_val]

    most_common = dfWithoutMissing[attribute].mode().loc[0]

    if len(dfMissing) > 0:
        dfMostCommon = pd.concat(
            [dfWithoutMissing[dfWithoutMissing[attribute] == most_common],
             dfMissing], ignore_index=True, sort=False)
    else:
        dfMostCommon = dfWithoutMissing[
            dfWithoutMissing[attribute] == most_common]

    splitsWithMissingAsMajority[most_common] = dfMostCommon.drop(
        columns=[attribute])

    a_vals = set(df[attribute])
    for v in a_vals:
        if v == most_common or v == missing_attr_val:
            splitsWithMissingAsBranch[v] = df[df[attribute] == v].drop(
                columns=[attribute])
            continue

        splitsWithMissingAsBranch[v] = df[df[attribute] == v].drop(
            columns=[attribute])
        splitsWithMissingAsMajority[v] = dfWithoutMissing[
            dfWithoutMissing[attribute] == v].drop(columns=[attribute])

    return splitsWithMissingAsMajority, splitsWithMissingAsBranch


def chi2_statistic_child(df_parent: pd.DataFrame, df_child: pd.DataFrame,
                         class_col: str = "class", missing_attr_val: str = "?"):
    parent_vals = set(df_parent[class_col])
    parent_counts = []
    child_counts = []
    # manually computing to handle case where child is missing class
    for val in parent_vals:
        if val == missing_attr_val:
            continue
        parent_counts.append(float(len(df_parent[df_parent[class_col] == val])))
        child_counts.append(float(len(df_child[df_child[class_col] == val])))
    np_p_counts = np.array(parent_counts)
    np_c_counts = np.array(child_counts)
    np_expected_c_counts = (np_p_counts / np.sum(np_p_counts)) * np.sum(
        np_c_counts)
    return np.sum(
        ((np_expected_c_counts - np_c_counts) ** 2) / np_expected_c_counts)


def get_chi2_statistic(df: pd.DataFrame, splits: pd.Series,
                       class_col: str = "class", missing_attr_val: str = "?"):
    chi2_vals = splits.map(lambda s: chi2_statistic_child(df, s, class_col,
                                                          missing_attr_val=missing_attr_val))
    chi2 = np.sum(chi2_vals)
    return chi2


def get_chi2_critical(alpha: float, num_classes: int,
                      num_attr_vals: int) -> float:
    q = 1.0 - alpha
    dof = (num_classes - 1) * (num_attr_vals - 1)
    return scistats.chi2.ppf(q, dof)


def only_missing(df_h, attr_h, missing_val: str = "?"):
    attr_vals = set(df_h[attr_h])
    for val_h in attr_vals:
        if val_h != missing_val:
            return False
    return True


if __name__ == "__main__":
    pth = "../data/agaricus-lepiota-training.csv"
    df1 = pd.read_csv(pth)
    metric = entropy
    df1 = df1.drop(columns="id")
    print("Best attr test: ", get_best_attribute(df1, metric))
    print("chi crit at alpha = 0.1: ", get_chi2_critical(0.1, 10, 15))
    print("chi crit at alpha = 0.01: ", get_chi2_critical(0.01, 10, 15))
    df_parent = pd.read_csv(
        "C:\\Users\\Jack\\Documents\\School\\Spring 2023\\CS529\\cs529-randomForest\\data\\test.csv")
    df_strong = pd.read_csv(
        "C:\\Users\\Jack\\Documents\\School\\Spring 2023\\CS529\\cs529-randomForest\\data\\test_child.csv")
    df_weak = pd.read_csv(
        "C:\\Users\\Jack\\Documents\\School\\Spring 2023\\CS529\\cs529-randomForest\\data\\test_child2.csv")
    x = get_chi2_statistic(df_parent, pd.Series([df_weak, df_strong]))
    print("chi crit at alpha = 0.05: ", get_chi2_critical(0.05, 2, 2))
    print(x)
    abs_best = get_best_attribute(df1, metric)
    small_best = get_best_attribute(df1, metric, feature_ratio=0.5)
    print(abs_best)
    print(small_best)
