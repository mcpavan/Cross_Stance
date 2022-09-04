import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

eval_base_path = "../../out/ustancebr/eval"
eval_file_path = f"{eval_base_path}/eval_data.csv"

df = pd.read_csv(eval_file_path)
df["bert"] = df["version"].apply(lambda x: "bertimbau" if x <= 32 else "berttweet")

# def get_max_value_row(df, max_var):
#     idx = df[max_var].idxmax()

#     return df.loc[idx]

# grp_cols = [
#     "data_folder",
#     "config_file_name",
#     "source_topic",
#     "destination_topic",
#     "bert",
# ]
# best_valid = df.groupby(grp_cols).apply(lambda x: get_max_value_row(x, "valid_fmacro"))["test_fmacro"]
# best_valid.to_csv("test1.csv")

plt.figure(figsize=(16,9))
sns.boxplot(
    data=df.query("source_topic==destination_topic"),
    y="destination_topic",
    x="test_fmacro",
    hue="bert",
    order=["bo", "lu", "co", "cl", "gl", "ig"],
    linewidth=.5,
    saturation=1,
    whis=1.5,
    fliersize=2.5,
    palette="Set3",
)
plt.title("Simple Domain")
plt.savefig(f"{eval_base_path}/SimpleDomain.png")
plt.show()

predefined_pairs = "(source_topic=='bo' and destination_topic=='lu')" + \
    "or (source_topic=='lu' and destination_topic=='bo')" + \
    "or (source_topic=='co' and destination_topic=='cl')" + \
    "or (source_topic=='cl' and destination_topic=='co')" + \
    "or (source_topic=='gl' and destination_topic=='ig')" + \
    "or (source_topic=='ig' and destination_topic=='gl')"

plt.figure(figsize=(16,9))
sns.boxplot(
    data=df.query(predefined_pairs),
    y="destination_topic",
    x="test_fmacro",
    hue="bert",
    order=["bo", "lu", "co", "cl", "gl", "ig"],
    linewidth=.5,
    saturation=1,
    whis=1.5,
    fliersize=2.5,
    palette="Set3",
)
plt.title("Cross Target")
plt.savefig(f"{eval_base_path}/CrossTarget.png")
plt.show()

plt.figure(figsize=(16,9))
sns.boxplot(
    data=df.query("source_topic!=destination_topic"),
    y="destination_topic",
    x="test_fmacro",
    hue="bert",
    order=["bo", "lu", "co", "cl", "gl", "ig"],
    linewidth=.5,
    saturation=1,
    whis=1.5,
    fliersize=2.5,
    palette="Set3",
)
plt.title("Hold1TopicOut")
plt.savefig(f"{eval_base_path}/Hold1TopicOut.png")
plt.show()
