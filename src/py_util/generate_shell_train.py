
config_version_map = [
    ("BertAAD", 101, 102),
    # ("BiCondBertLstm", 101, 104),
    # ("BertBiLSTMAttn", 101, 116),
    # ("BertBiLSTMJointAttn", 101, 116),
    # ("BertCrossNet", 101, 108),
    # ("BertJointCL", 101, 102),
    # ("BertTOAD", 101, 108),
]

# config_file_name = "Bert_BiLSTMAttn"
data_folder = "hold1topic_out" #"hold1topic_out" or "simple_domain"
out_path = "./sh_auto_train_aad.sh"

with open(out_path, "w") as f_:
    print("", end="", file=f_)

target_model_list = ["bo", "lu", "co", "cl", "gl", "ig"]
base_command = "python train_model.py -m train -c ../../config/{data_folder}/{config_file_name}_v{k}.txt -t ../../data/ustancebr/v2/{data_folder}/final_{tgt}_train.csv -v ../../data/ustancebr/v2/{data_folder}/final_{tgt}_valid.csv -p ../../data/ustancebr/v2/{data_folder}/final_{tgt}_test.csv -n {tgt} -e 5 -s 1"
base_out = " > ../../out/ustancebr/log/{data_folder}/{tgt}_{config_file_name}_v{k}.txt"
for config_file_name, init_version, final_version in config_version_map:
    base_text = "\n"

    if config_file_name == "BertAAD":
        if data_folder == "simple_domain":
            tgt_train_list = ["lu", "bo", "cl", "co", "ig", "gl"]
        else: #if data_folder == "hold1topic_out":
            tgt_train_list = ["bo", "lu", "co", "cl", "gl", "ig"]
        
        for k, tgt in enumerate(target_model_list):
            base_text += base_command.replace("{tgt}", tgt) + " -g ../../data/ustancebr/v2/simple_domain/final_" + tgt_train_list[k] + "_train.csv" + base_out.replace("{tgt}", tgt) + "\n"

    elif config_file_name == "BertJointCL":
        for k, tgt in enumerate(target_model_list):
            base_text += base_command.replace("{tgt}", tgt) + " -a 100" + base_out.replace("{tgt}", tgt) + "\n"

    else:
        for k, tgt in enumerate(target_model_list):
            base_text += base_command.replace("{tgt}", tgt) + base_out.replace("{tgt}", tgt) + "\n"

    for k in range(init_version, final_version+1):
        partial_text = base_text \
        .replace("{config_file_name}", config_file_name) \
        .replace("{k}", str(k)) \
        .replace("{data_folder}", data_folder) \

        with open(out_path, "a") as f_:
            print(partial_text, "\n\n", file=f_)
    
with open(out_path, "a") as f_:
    print("\n", file=f_, flush=True)
    

