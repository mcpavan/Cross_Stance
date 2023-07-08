import pandas as pd
from glob import glob
from tqdm import tqdm
from itertools import product

dataset = "semeval" # "ustancebr" or "semeval"
eval_base_path = f"../../out/{dataset}/eval"
out_path = f"{eval_base_path}/.results/data/eval_data.csv"
errors_out_path = f"{eval_base_path}/.results/data/errors_eval_data.csv"

if dataset == "ustancebr":
    data_list = ["valid", "test"]
    metric_list = ["f", "p", "r"]
    class_list = ["macro", "0", "1"]

elif dataset == "semeval":
    data_list = ["valid", "test"]
    metric_list = ["f", "p", "r"]
    class_list = ["macro", "0", "1", "2"]

eval_results_dict = {
    "data_folder": [],
    "source_topic": [],
    "destination_topic": [],
    "config_file_name": [],
    "version": [],
    "epoch": [],
}

for data_, metric_, class_ in product(data_list, metric_list, class_list):
    eval_results_dict[f"{data_}_{metric_}{class_}"] = []

for eval_file_path in tqdm(glob(f"{eval_base_path}/**/*.txt", recursive=True)):
    eval_file_path = eval_file_path.replace("\\", "/")
    data_folder = eval_file_path.replace(f"{eval_base_path}/", "").split("/")[0]
    saved_file_name = eval_file_path.replace(f"{eval_base_path}/", "").split("/")[-1]
    source_topic = saved_file_name.split("_")[0]
    destination_topic = saved_file_name.split("_")[1]
    config_file_name = "_".join(saved_file_name.split("_")[2:-1])
    version = float(saved_file_name.split("_")[-1].replace(".txt","")[1:])

    with open(eval_file_path, "r") as f_:
        prefix = ""
        for line in f_.readlines():
            if line.startswith('Evaluating on "VALIDATION" data'):
                prefix = "valid"
            elif line.startswith('Evaluating on "TEST" data'):
                prefix = "test"
            elif not line.startswith("saved to") \
                and prefix != "" \
                and not line.startswith("Output Path:") \
                and not line.startswith("Got '<Response"):
                line_spl = line.split()
                
                for i in range(int(len(line_spl)/2)):#3):
                    key = f'{prefix}_{line_spl[i*2].replace("_", "").replace(":", "")}'
                    value = float(line_spl[(i*2)+1])

                    eval_results_dict[key] += [value]
            
    eval_results_dict["data_folder"] += [data_folder]
    eval_results_dict["source_topic"] += [source_topic]
    eval_results_dict["destination_topic"] += [destination_topic]
    eval_results_dict["config_file_name"] += [config_file_name]
    eval_results_dict["version"] += [version]

    current_length = len(eval_results_dict["data_folder"])

    for key in eval_results_dict.keys():
        if len(eval_results_dict[key]) < current_length:
            eval_results_dict[key] += [None]        

df_results = pd.DataFrame(eval_results_dict)
# df_results = df_results.dropna(how="all")
df_results = df_results.dropna(subset=["test_fmacro", "valid_fmacro"], how="all")
df_results.to_csv(out_path, index=False)


errors_out_cols = [
    "config_file_name",
    "version",
    "source_topic",
    "destination_topic",
]
df_results \
    .query("test_p1 != test_p1") \
    [errors_out_cols] \
    .sort_values(errors_out_cols) \
    .to_csv(errors_out_path, index=False)