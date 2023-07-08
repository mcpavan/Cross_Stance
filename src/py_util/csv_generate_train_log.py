import pandas as pd
from glob import glob
from tqdm import tqdm
import re
from itertools import product
import os

dataset = "semeval" # "ustancebr" or "semeval"
log_base_path = f"../../out/{dataset}/log"
eval_base_path = f"../../out/{dataset}/eval"
out_path = f"{eval_base_path}/.results/data/log_data.csv"
errors_out_path = f"{eval_base_path}/.results/data/errors_log_data.csv"
os.makedirs("/".join(errors_out_path.split("/")[:-1]), exist_ok=True)

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

for log_file_path in tqdm(glob(f"{log_base_path}/**/*.txt", recursive=True)):
    log_file_path = log_file_path.replace("\\", "/")
    if log_file_path.endswith("old.txt"):
        continue
    data_folder = log_file_path.replace(f"{log_base_path}/", "").split("/")[0]
    saved_file_name = log_file_path.replace(f"{log_base_path}/", "").split("/")[-1]
    source_topic = saved_file_name.split("_")[0]
    # destination_topic = saved_file_name.split("_")[1]
    config_file_name = "_".join(saved_file_name.split("_")[1:-1])
    version = int(saved_file_name.split("_")[-1].replace(".txt","")[1:])

    with open(log_file_path, "r") as f_:
        prefix = ""
        for line in f_.readlines():
            if "Clustering" in line:
                line = line.split("Clustering")[0]
            
            epoch_line = re.search(r"] epoch ", line)
            metric_line = re.match(r"[p|f|r]{1}(_[macro|0|1]){1}", line)
            final_line = re.match(r"TRAINED for \d+ epochs", line)

            if epoch_line is not None or final_line is not None:
                if epoch_line is not None:
                    epoch = int(line.split("epoch")[-1].strip())
                else:
                    epoch = -1
                
                eval_results_dict["data_folder"] += [data_folder]
                eval_results_dict["source_topic"] += [source_topic]
                eval_results_dict["destination_topic"] += [source_topic]#destination_topic]
                eval_results_dict["config_file_name"] += [config_file_name]
                eval_results_dict["version"] += [version]
                eval_results_dict["epoch"] += [epoch]

            elif line.startswith('Evaluating on "VALIDATION" data'):
                prefix = "valid"
            elif line.startswith('Evaluating on "TEST" data'):
                prefix = "test"
            elif line.startswith('Evaluating on "TRAIN" data'):
                prefix = "train"
            elif metric_line is not None and prefix != "train":
                line_spl = line.split()
                
                for i in range(int(len(line_spl)/2)):#3):
                    key = f'{prefix}_{line_spl[i*2].replace("_", "").replace(":", "")}'
                    value = float(line_spl[(i*2)+1])

                    eval_results_dict[key] += [value]
            
    current_length = len(eval_results_dict["data_folder"])

    for key in eval_results_dict.keys():
        if len(eval_results_dict[key]) < current_length:
            eval_results_dict[key] += [None]        

df_results = pd.DataFrame(eval_results_dict)
df_results = df_results.dropna().query("data_folder in ['simple_domain', 'hold1topic_out']")
df_results.to_csv(out_path, index=False)


errors_out_cols = [
    "config_file_name",
    "version",
    "epoch",
    "source_topic",
    "destination_topic",
]
df_results \
    .query("test_p1 != test_p1") \
    [errors_out_cols] \
    .sort_values(errors_out_cols) \
    .to_csv(errors_out_path, index=False)