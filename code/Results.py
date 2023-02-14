import pandas as pd

class Results:
    def __init__(self):
        self.cols = ["feature_r", "metric_fn", "alpha", "bag_r", "tree_count",
                     "split_seed", "max_depth", "avg_depth", "accuracy", "top_feature",
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
        params_entry = pd.DataFrame(params_entry, index=[len(self.results_df)])
        self.results_df = pd.concat([self.results_df, params_entry])

if __name__ == "__main__":
    demo_res = Results()
    print(demo_res.results_df)
    demo_res.add_entry([0.1, "entropy", 0.95, 0.3, 14, 1, 5, 4, 0.9, "stalk-root", 1, 2])
    print(demo_res.results_df)
    demo_res.add_entry([0.11, "gini", 0.5, 0.2, 10, 4, 2, 8, 0.7, "stalk-root", 3, 4])
    print(demo_res.results_df)