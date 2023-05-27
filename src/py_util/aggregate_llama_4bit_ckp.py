import json
import pandas as pd
from glob import glob

base_path = "../../out/ustancebr/pred/Llama_4bit_bo_v0/llama_cpp_pred_checkpoints"
file_list = [
    (f"{base_path}/Llama_4bit_ustancebr_1501.ckp", 0),
    (f"{base_path}/Llama_4bit_ustancebr_2302.ckp", 1499),
    (f"{base_path}/Llama_4bit_ustancebr_2357.ckp", 2299),
]
ckp_list = []
for file, start_idx in file_list:
    with open(file, mode="r", encoding="utf-8") as f:
        part_pred = pd.DataFrame(json.load(f)).set_index("index")
        part_pred.index += start_idx

        ckp_list += [part_pred]

df_pred = pd.concat(ckp_list, axis=0, ignore_index=False)
df_pred = df_pred.reset_index().drop_duplicates("index").set_index("index")
df_pred["pred"] = df_pred["pred"].astype(float)

out_path = f"{base_path}/Llama_4bit_ustancebr_full.ckp"
with open(out_path, mode="w", encoding="utf-8") as f_out:
    json.dump(df_pred.reset_index().to_dict(orient="list"), f_out)
