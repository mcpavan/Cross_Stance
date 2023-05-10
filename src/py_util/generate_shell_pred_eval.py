config_version_map = [
    # ("BertAAD", 101, 102),
    # ("BiCondBertLstm", 101, 104),
    # ("BertBiLSTMAttn", 101, 116),
    # ("BertBiLSTMJointAttn", 101, 116),
    # ("BertCrossNet", 101, 108),
    # ("BertJointCL", 101, 102),
    ("BertTOAD", 101, 104),
]

# config_file_name = "Bert_BiLSTMAttn"
# file_prefix = config_file_name.replace("_", "")
data_folder = "hold1topic_out" #"hold1topic_out" or "simple_domain"
out_path_prefix = "./sh_auto_pred_eval_h1to"

target_list = ["bo", "lu", "co", "cl", "gl", "ig"]
base_config = " -c ../../config/{data_folder}/{config_file_name}_v{k}.txt"
base_test_path = " -p ../../data/ustancebr/v2/{data_folder}/final_{dst_tgt}_test.csv"
base_vld_path = " -v ../../data/ustancebr/v2/{data_folder}/final_{src_tgt}_valid.csv"
base_ckp_path = " -f ../../checkpoints/ustancebr/{model_folder}/V{k}/ckp-{config_file_name}_ustancebr_{src_tgt}-BEST.tar"
base_out =  " -o ../../out/ustancebr/pred/{data_folder}/{src_tgt}_{dst_tgt}_{config_file_name}_v{k}.txt"
base_log =  " > ../../out/ustancebr/eval/{data_folder}/{src_tgt}_{dst_tgt}_{config_file_name}_v{k}.txt"

base_command = "python train_model.py -m pred_eval" + \
                base_config + \
                base_test_path + \
                base_vld_path + \
                base_ckp_path + \
                base_out + \
                base_log

if data_folder == "simple_domain":
    base_text = ""
    for src_tgt in target_list:
        base_text += "\n"
        for dst_tgt in target_list:
            base_text += "\n" + base_command.replace("{src_tgt}", src_tgt).replace("{dst_tgt}", dst_tgt)

elif data_folder == "hold1topic_out":
    base_text = ""
    for tgt in target_list:
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

        with open(out_path, "a") as f_:
            print(partial_text, "\n", file=f_)
        
    with open(out_path, "a") as f_:
        print("\n", file=f_, flush=True)
