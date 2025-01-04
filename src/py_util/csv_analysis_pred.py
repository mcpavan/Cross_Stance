import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from train_model import load_config_file

show_img = False
save_plots = False
pretrained_model_name_map = {
    "neuralmind/bert-base-portuguese-cased": "BERTimbau",
    "pablocosta/bertabaporu-base-uncased": "BERTabaporu",
    "../../data/LLMs/ggml-alpaca-7b-q4.bin": "Alpaca-4bit",
}

dataset = "brmoral" # "ustancebr" or "semeval" or "govbr_semeval" or "brmoral" or "mtwitter"
config_folder = "govbr_brmoral"
config_base_path = f"../../config"
eval_base_path = f"../../out/{dataset}/eval"
data_path = f"{eval_base_path}/.results/data"
pred_file_path = f"{data_path}/pred_data.csv"

grp_cols_source = [
    "data_folder",
    "config_file_name",
    "source_topic",
    "destination_topic",
    "pretrained_model",
]
grp_cols_dest = [
    "data_folder",
    "config_file_name",
    "destination_topic",
    "pretrained_model",
]
out_metrics = [
    "test_fmacro",
    "test_pmacro",
    "test_rmacro",
]

if dataset in ["semeval", "election", "ufrgs", "govbr_semeval", "brmoral", "mtwitter"]:
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

def get_pretrained_model_name(info):
    data_folder = info["data_folder"]
    data_folder_path = data_folder
    if dataset == "semeval" and data_folder == "simple_domain_translated":
        data_folder_path = "simple_domain"
    elif dataset in ["brmoral", "mtwitter"]:
        data_folder_path = "simple_domain"

    config_file_name = info["config_file_name"]
    version = info["version"]
    if isinstance(version, str):
        version = version.replace("inv", "")
    
    config_file_path = f"{config_base_path}/{config_folder}/{data_folder_path}/{config_file_name}_v{int(version)}.txt"
    config_dict = load_config_file(config_file_path=config_file_path)
    model_name = "-"

    if "bert" in config_dict or "bert" in config_dict["name"]:
        model_name = config_dict.get("bert_pretrained_model", "bert-base-uncased")

    if "pretrained_model_name" in config_dict:
        model_name = config_dict.get("pretrained_model_name", "bert-base-uncased")

    return pretrained_model_name_map.get(model_name, model_name)

df = pd.read_csv(pred_file_path)
df["pretrained_model"] = df.apply(get_pretrained_model_name, axis=1)
# df["valid_fmacro"] = df[["test_fmacro", "valid_fmacro"]].apply(lambda x: x["valid_fmacro"] if x["valid_fmacro"] == x["valid_fmacro"] else x["test_fmacro"], axis=1)
df["valid_fmacro"] = df["test_fmacro"]

best_valid = df.groupby(grp_cols_source) \
               .apply(lambda x: get_max_value_row(x, "valid_fmacro")) \
               [out_metrics+["version"]] #\
            #    .unstack("destination_topic")
best_valid.to_csv(f"{data_path}/best_test_metrics_full_test_pred.csv")