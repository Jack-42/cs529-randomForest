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
                       class_col: str = "class", n_features: int = None) -> str:
    """
    Given a df, return the attribute which gave the highest information gain
    :param df: pandas Dataframe, attributes are columns
    :param metric_fn: function that returns a float
    :param class_col: str, the column where the class label is
    :param n_features: (optional) int, use a random subset of this size
        instead of all features available. If None all features used
    :return: str, the column which gave the highest info gain
    """
    attrs = df.columns.drop([class_col])
    if n_features is not None:
        attrs = attrs.to_series().sample(n=n_features)
    info_gains = attrs.map(lambda a: information_gain(df, a, metric_fn))
    max_idx = np.argmax(info_gains)
    best_atr = attrs[max_idx]
    return best_atr


def chi2_statistic_child(df_parent: pd.DataFrame, df_child: pd.DataFrame,
                         class_col: str = "class"):
    parent_vals = set(df_parent[class_col])
    parent_counts = []
    child_counts = []
    # manually computing to handle case where child is missing class
    for val in parent_vals:
        parent_counts.append(float(len(df_parent[df_parent[class_col] == val])))
        child_counts.append(float(len(df_child[df_child[class_col] == val])))
    np_p_counts = np.array(parent_counts)
    np_c_counts = np.array(child_counts)
    np_expected_c_counts = (np_p_counts / np.sum(np_p_counts)) * np.sum(
        np_c_counts)
    return np.sum(
        ((np_expected_c_counts - np_c_counts) ** 2) / np_expected_c_counts)


def get_chi2_statistic(df: pd.DataFrame, splits: pd.Series,
                       class_col: str = "class"):
    chi2_vals = splits.map(lambda s: chi2_statistic_child(df, s, class_col))
    chi2 = np.sum(chi2_vals)
    return chi2


def get_chi2_critical(alpha: float, num_classes: int,
                      num_attr_vals: int) -> float:
    q = 1.0 - alpha
    dof = (num_classes - 1) * (num_attr_vals - 1)
    return scistats.chi2.ppf(q, dof)


def get_subsample(df: pd.DataFrame, ratio: float, random_state: int = None):
    """
    Get subsample from dataset
    :param df: pd.Dataframe, dataset being sampled
    :param ratio: float, fraction of data to use in sample
    :param random_state: (optional) int, random seed for reproducibility
    :return: pd.Dataframe, the subsample
    """
    return df.sample(frac=ratio, replace=True, random_state=random_state)


if __name__ == "__main__":
    df_parent = pd.read_csv(
        "C:\\Users\\Jack\\Documents\\School\\Spring 2023\\CS529\\cs529-randomForest\\data\\test.csv")
    df_strong = pd.read_csv(
        "C:\\Users\\Jack\\Documents\\School\\Spring 2023\\CS529\\cs529-randomForest\\data\\test_child.csv")
    df_weak = pd.read_csv(
        "C:\\Users\\Jack\\Documents\\School\\Spring 2023\\CS529\\cs529-randomForest\\data\\test_child2.csv")
    x = get_chi2_statistic(df_parent, pd.Series([df_weak, df_strong]))
    print("chi crit at alpha = 0.05: ", get_chi2_critical(0.05, 2, 2))
    print(x)
    print(df_weak)
    print(get_subsample(df_weak, 1.0, random_state=1))
    pth = "../data/agaricus-lepiota-training.csv"
    df1 = pd.read_csv(pth)
    df_train = df1.drop(columns="id")  # this is important!
    print(get_best_attribute(df_train, entropy))
    print(get_best_attribute(df_train, entropy, n_features=5))
