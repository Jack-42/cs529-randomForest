import pandas as pd


class Results:
    def __init__(self):
        self.cols = ["feature_r", "metric_fn", "alpha", "bag_r", "tree_count",
                     "split_seed", "max_depth", "avg_depth", "accuracy",
                     "seed_for_bag_seed_generator",
                     'seed_for_feat_bag_seed_generator']
        self.results_df = pd.DataFrame(columns=self.cols)

    def add_entry(self, params: list):
        """
        Add entry to results
        :param params: list, expected ordering is the same as self.cols
        :return: None
        """
        params_entry = {}
        for i in range(len(params)):
            params_entry[self.cols[i]] = params[i]
        params_entry = pd.DataFrame(params_entry)
        self.results_df = pd.concat(self.results_df, params_entry)
