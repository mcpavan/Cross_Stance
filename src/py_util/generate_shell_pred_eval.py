config_version_map = [
    # ("BertAAD", 101, 102),
    # ("BiCondBertLstm", 101, 101), #104),
    ("BertBiLSTMAttn", 1, 1),#16), #116),
    # ("BertBiLSTMJointAttn", 101, 101), #116),
    # ("BertCrossNet", 101, 101), #108),
    # ("BertJointCL", 101, 102),
    # ("BertTOAD", 101, 104),
    ("Llama_8bit", 0, 0),#16), #116),
]

# config_file_name = "Bert_BiLSTMAttn"
# file_prefix = config_file_name.replace("_", "")
model_dataset = "ustancebr" #"ustancebr" or "semeval" or "govbr" or "govbr_semeval" or "govbr_brmoral" or "ufrgs" or "election"
test_dataset = "govbr" #"ustancebr" or "semeval" or "govbr" or "govbr_semeval" or "ufrgs" or "election" or "brmoral" or "mtwitter"
data_folder = "simple_domain" #"hold1topic_out" or "simple_domain"
out_path_prefix = f"./sh_auto_pred_eval_{model_dataset}_{test_dataset}_sd"

columns_set1 = ["Text","Target","Polarity"]
columns_set2 = ["Tweet","Target","Stance"]
columns_set = columns_set1 #columns_set1 or columns_set2 or None

dataset_path = test_dataset
if test_dataset in ["ustancebr"]:
    dataset_path = "ustancebr/v2"
elif test_dataset in ["govbr_semeval"]:
    dataset_path = "semeval"

data_folder_ckp = data_folder
if data_folder == "simple_domain_15k":
    data_folder = "simple_domain"

data_folder_path = data_folder
if test_dataset in ["govbr_semeval"]:
    data_folder_path = "simple_domain_translated_oc"


if model_dataset in ["ustancebr", "govbr"]:
    model_target_list = ["bo", "lu", "co", "cl", "gl", "ig"]
    # model_target_list = ["ig"]
elif model_dataset in ["semeval", "govbr_semeval"]:
    model_target_list = ["a", "cc", "fm", "hc", "la"]
    if data_folder in ["hold1topic_out"]:
        model_target_list += ["dt"]
elif model_dataset in ["govbr_brmoral"]:
    # model_target_list = ["ab", "ca", "ct", "dp", "dr", "gc", "gm", "rq"] #full
    # model_target_list = ["ca", "dp", "dr", "gc", "gm", "rq"] #brmoral (exc semeval and ustancebr)
    model_target_list = ["ca", "dp", "dr", "rq"] #mtwitter (exc semeval and ustancebr)

if test_dataset in ["ustancebr", "govbr"]:
    test_target_list = ["bo", "lu", "co", "cl", "gl", "ig"]
elif test_dataset in ["semeval", "govbr_semeval"]:
    test_target_list = ["a", "cc", "fm", "dt", "hc", "la"]
elif test_dataset in ["brmoral"]:
    test_target_list = ["gm", "gc", "ab", "dp", "dr", "ca", "rq", "ct"]
elif test_dataset in ["mtwitter"]:
    test_target_list = ["ab", "co", "ma", "mp", "pm"]

test_set_name = "test"
if test_dataset in ["brmoral", "mtwitter"]:
    test_set_name = "full"

base_config = " -c ../../config/{model_dataset}/{data_folder}/{config_file_name}_v{k}.txt"
base_test_path = " -p ../../data/{dataset_path}/{data_folder_path}/final_{dst_tgt}_{test_set_name}.csv"
# base_vld_path = " -v ../../data/{dataset_path}/{data_folder_path}/final_{src_tgt}_valid.csv"
base_vld_path = ""
base_ckp_path = " -f ../../checkpoints/{model_dataset}/{data_folder_ckp}/{model_folder}/V{k}/ckp-{config_file_name}_{model_dataset}_{src_tgt}-BEST.tar"
base_override_columns = ""
if columns_set is not None:
    base_override_columns = f" -pt {columns_set[0]} -pg {columns_set[1]} -pl {columns_set[2]}"

if model_dataset == test_dataset:
    base_out =  " -o ../../out/{model_dataset}/pred/{data_folder_ckp}/{src_tgt}_{dst_tgt}_{config_file_name}_v{k}.txt"
    base_log =  " > ../../out/{model_dataset}/eval/{data_folder_ckp}/{src_tgt}_{dst_tgt}_{config_file_name}_v{k}.txt"
else:
    base_out =  " -o ../../out/{test_dataset}/pred/{model_dataset}_{data_folder_ckp}/{src_tgt}_{dst_tgt}_{config_file_name}_v{k}.txt"
    base_log =  " > ../../out/{test_dataset}/eval/{model_dataset}_{data_folder_ckp}/{src_tgt}_{dst_tgt}_{config_file_name}_v{k}.txt"


base_command = "python train_model.py -m pred_eval" + \
                base_config + \
                base_test_path + \
                base_vld_path + \
                base_ckp_path + \
                base_override_columns + \
                base_out + \
                base_log

if data_folder == "simple_domain":
    base_text = ""
    for src_tgt in model_target_list:
        base_text += "\n"
        for dst_tgt in test_target_list:
            base_text += "\n" + base_command.replace("{src_tgt}", src_tgt).replace("{dst_tgt}", dst_tgt)

elif data_folder == "hold1topic_out":
    base_text = ""
    for tgt in model_target_list:
        base_text += "\n" + base_command.replace("{src_tgt}", tgt).replace("{dst_tgt}", tgt)

for config_file_name, init_version, final_version in config_version_map:
    out_path = f"{out_path_prefix}_{config_file_name}.sh"
    
    with open(out_path, "w") as f_:
        print("\n", file=f_)

    for k in range(init_version, final_version+1):
        partial_text = base_text \
        .replace("{config_file_name}", config_file_name) \
        .replace("{model_folder}", config_file_name.lower().replace("bert", "")) \
        .replace("{k}", str(k)) \
        .replace("{data_folder}", data_folder) \
        .replace("{data_folder_path}", data_folder_path) \
        .replace("{data_folder_ckp}", data_folder_ckp) \
        .replace("{model_dataset}", model_dataset) \
        .replace("{test_dataset}", test_dataset) \
        .replace("{dataset_path}", dataset_path) \
        .replace("{test_set_name}", test_set_name)

        with open(out_path, "a") as f_:
            print(partial_text, "\n", file=f_)
        
    with open(out_path, "a") as f_:
        print("\n", file=f_, flush=True)
