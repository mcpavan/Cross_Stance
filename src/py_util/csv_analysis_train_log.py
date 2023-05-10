import pandas as pd

config_base_path = "../../config"
eval_base_path = "../../out/ustancebr/eval"
data_path = f"{eval_base_path}/.results/data"
eval_file_path = f"{data_path}/log_data.csv"

grp_cols_source = [
    "data_folder",
    "config_file_name",
    "source_topic",
    "destination_topic",
]

out_metrics = [
    "test_fmacro",
    "test_pmacro",
    "test_rmacro",
]

def get_max_value_row(df, max_var):
    idx = df[max_var].idxmax()

    return df.loc[idx]

df = pd.read_csv(eval_file_path)

best_valid = df.groupby(grp_cols_source) \
               .apply(lambda x: get_max_value_row(x, "valid_fmacro")) \
               [out_metrics + ["version", "epoch"]]
best_valid.to_csv(f"{data_path}/best_valid_metrics_full_log_train.csv")

best_test = df.groupby(grp_cols_source) \
               .apply(lambda x: get_max_value_row(x, "test_fmacro")) \
               [out_metrics + ["version", "epoch"]]
best_test.to_csv(f"{data_path}/best_test_metrics_full_log_train.csv")
