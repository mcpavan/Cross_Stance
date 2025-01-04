import pandas as pd

dataset = "govbr" # "ustancebr" or "semeval" or "govbr" or "govbr_semeval" or "election" or "ufrgs"
config_base_path = "../../config"
eval_base_path = f"../../out/{dataset}/eval"
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

if dataset in ["semeval", "election", "ufrgs"]:
    out_metrics += [
        "test_f0",
        "test_f1",
        "test_f2",
        "test_p0",
        "test_p1",
        "test_p2",
        "test_r0",
        "test_r1",
        "test_r2",
    ]

def get_max_value_row(df, max_var):
    idx = df[max_var].idxmax()

    return df.loc[idx]

def get_min_value_row(df, min_var):
    idx = df[min_var].idxmin()

    return df.loc[idx]

df = pd.read_csv(eval_file_path)

# Invert the classes predicted
if dataset in ["ustancebr", "govbr", "govbr_semeval"]:#, "election"]:
    df_inv = df.copy()
    df_inv["test_t0"] = 100
    df_inv.eval("test_f0 = (test_t0/test_p0)-test_t0", inplace=True)
    df_inv.eval("test_f1 = (test_t0/test_r0)-test_t0", inplace=True)
    df_inv.eval("test_t1 = (test_f0*test_r1)/(1-test_r1)", inplace=True)

    df_inv["test_t0_inv"] = df_inv["test_f1"]
    df_inv["test_f0_inv"] = df_inv["test_t1"]
    df_inv["test_f1_inv"] = df_inv["test_t0"]
    df_inv["test_t1_inv"] = df_inv["test_f0"]

    df_inv.eval("test_p0 = test_t0_inv / (test_t0_inv + test_f0_inv)", inplace=True)
    df_inv.eval("test_p1 = test_t1_inv / (test_t1_inv + test_f1_inv)", inplace=True)
    df_inv.eval("test_pmacro = (test_p0 + test_p1) / 2", inplace=True)
    
    df_inv.eval("test_r0 = test_t0_inv / (test_t0_inv + test_f1_inv)", inplace=True)
    df_inv.eval("test_r1 = test_t1_inv / (test_t1_inv + test_f0_inv)", inplace=True)
    df_inv.eval("test_rmacro = (test_r0 + test_r1) / 2", inplace=True)

    df_inv.eval("test_f0 = 2 * (test_p0 * test_r0) / (test_p0 + test_r0)", inplace=True)
    df_inv.eval("test_f1 = 2 * (test_p1 * test_r1) / (test_p1 + test_r1)", inplace=True)
    df_inv.eval("test_fmacro = (test_f0 + test_f1) / 2", inplace=True)

    df_inv.drop(
        columns = [
            "test_t0",
            "test_f0",
            "test_f1",
            "test_t1",
            "test_t0_inv",
            "test_f0_inv",
            "test_f1_inv",
            "test_t1_inv"
        ],
        inplace=True
    )

    df_inv["version"] = df_inv["version"].apply(lambda x: f"{x}inv")
    df["version"] = df["version"].astype(str)
    df = pd.concat([df, df_inv], axis=0, ignore_index=True)

best_valid = df.groupby(grp_cols_source) \
               .apply(lambda x: get_max_value_row(x, "valid_fmacro")) \
               [out_metrics + ["version", "epoch"]]
best_valid.to_csv(f"{data_path}/best_valid_metrics_full_log_train.csv")

best_test = df.groupby(grp_cols_source) \
               .apply(lambda x: get_max_value_row(x, "test_fmacro")) \
               [out_metrics + ["version", "epoch"]]
best_test.to_csv(f"{data_path}/best_test_metrics_full_log_train.csv")

worst_test = df.groupby(grp_cols_source) \
               .apply(lambda x: get_min_value_row(x, "test_fmacro")) \
               [out_metrics + ["version", "epoch"]]
worst_test.to_csv(f"{data_path}/worst_test_metrics_full_log_train.csv")
