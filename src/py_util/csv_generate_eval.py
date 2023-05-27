import pandas as pd
from glob import glob
from tqdm import tqdm

eval_base_path = "../../out/ustancebr/eval"
out_path = f"{eval_base_path}/.results/data/eval_data.csv"
errors_out_path = f"{eval_base_path}/.results/data/errors_eval_data.csv"

eval_results_dict = {
    "data_folder": [],
    "source_topic": [],
    "destination_topic": [],
    "config_file_name": [],
    "version": [],
    "valid_fmacro": [],
    "valid_f0": [],
    "valid_f1": [],
    "valid_pmacro": [],
    "valid_p0": [],
    "valid_p1": [],
    "valid_rmacro": [],
    "valid_r0": [],
    "valid_r1": [],
    "test_fmacro": [],
    "test_f0": [],
    "test_f1": [],
    "test_pmacro": [],
    "test_p0": [],
    "test_p1": [],
    "test_rmacro": [],
    "test_r0": [],
    "test_r1": [],
}


for eval_file_path in tqdm(glob(f"{eval_base_path}/**/*.txt", recursive=True)):
    eval_file_path = eval_file_path.replace("\\", "/")
    data_folder = eval_file_path.replace(f"{eval_base_path}/", "").split("/")[0]
    saved_file_name = eval_file_path.replace(f"{eval_base_path}/", "").split("/")[-1]
    source_topic = saved_file_name.split("_")[0]
    destination_topic = saved_file_name.split("_")[1]
    config_file_name = "_".join(saved_file_name.split("_")[2:-1])
    version = int(saved_file_name.split("_")[-1].replace(".txt","")[1:])

    with open(eval_file_path, "r") as f_:
        prefix = ""
        for line in f_.readlines():
            if line.startswith('Evaluating on "VALIDATION" data'):
                prefix = "valid"
            elif line.startswith('Evaluating on "TEST" data'):
                prefix = "test"
            elif not line.startswith("saved to") and prefix != "":
                line_spl = line.split()
                
                for i in range(3):
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
df_results.dropna(how="all").to_csv(out_path, index=False)


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