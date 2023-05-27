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

config_base_path = "../../config"
eval_base_path = "../../out/ustancebr/eval"
data_path = f"{eval_base_path}/.results/data"
img_base_path = f"{eval_base_path}/.results/img"
eval_file_path = f"{data_path}/eval_data.csv"

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

def get_max_value_row(df, max_var):
    idx = df[max_var].idxmax()

    return df.loc[idx]

def get_pretrained_model_name(info):
    data_folder = info["data_folder"]
    config_file_name = info["config_file_name"]
    version = info["version"]

    config_file_path = f"{config_base_path}/{data_folder}/{config_file_name}_v{version}.txt"

    config_dict = load_config_file(config_file_path=config_file_path)
    model_name = "-"

    if "bert" in config_dict or "bert" in config_dict["name"]:
        model_name = config_dict.get("bert_pretrained_model", "bert-base-uncased")

    if "pretrained_model_name" in config_dict:
        model_name = config_dict.get("pretrained_model_name", "bert-base-uncased")

    return pretrained_model_name_map.get(model_name, model_name)

df = pd.read_csv(eval_file_path)
df["pretrained_model"] = df.apply(get_pretrained_model_name, axis=1)
df["valid_fmacro"] = df["valid_fmacro"].fillna(0)

best_valid = df.groupby(grp_cols_source) \
               .apply(lambda x: get_max_value_row(x, "valid_fmacro")) \
               [["test_fmacro", "test_pmacro", "test_rmacro", "version"]] #\
            #    .unstack("destination_topic")
best_valid.to_csv(f"{data_path}/best_valid_metrics_full_test.csv")

if save_plots:
    sd_df = df.query("source_topic == destination_topic")
    plt.figure(figsize=(16,9))
    sns.boxplot(
        data=sd_df,
        y="destination_topic",
        x="test_fmacro",
        hue="pretrained_model",
        order=["bo", "lu", "co", "cl", "gl", "ig"],
        linewidth=.5,
        saturation=1,
        whis=1.5,
        fliersize=2.5,
        palette="Set3",
    )
    plt.title("Simple Domain")
    plt.savefig(f"{img_base_path}/SimpleDomain.png")
    if show_img:
        plt.show()

    predefined_pairs = "(source_topic=='bo' and destination_topic=='lu')" + \
        "or (source_topic=='lu' and destination_topic=='bo')" + \
        "or (source_topic=='co' and destination_topic=='cl')" + \
        "or (source_topic=='cl' and destination_topic=='co')" + \
        "or (source_topic=='gl' and destination_topic=='ig')" + \
        "or (source_topic=='ig' and destination_topic=='gl')"

    ct_df = df.query(predefined_pairs)
    plt.figure(figsize=(16,9))
    sns.boxplot(
        data=df.query(predefined_pairs),
        y="destination_topic",
        x="test_fmacro",
        hue="pretrained_model",
        order=["bo", "lu", "co", "cl", "gl", "ig"],
        linewidth=.5,
        saturation=1,
        whis=1.5,
        fliersize=2.5,
        palette="Set3",
    )
    plt.title("Cross Target")
    plt.savefig(f"{img_base_path}/CrossTarget.png")
    if show_img:
        plt.show()


    h1to_df = df.query("source_topic!=destination_topic")
    plt.figure(figsize=(16,9))
    sns.boxplot(
        data=h1to_df,
        y="destination_topic",
        x="test_fmacro",
        hue="pretrained_model",
        order=["bo", "lu", "co", "cl", "gl", "ig"],
        linewidth=.5,
        saturation=1,
        whis=1.5,
        fliersize=2.5,
        palette="Set3",
    )
    plt.title("Hold1TopicOut")
    plt.savefig(f"{img_base_path}/Hold1TopicOut.png")
    if show_img:
        plt.show()
