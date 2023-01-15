
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
for config_file_name, init_version, final_version in config_version_map:
    if config_file_name != "BertAAD":
        base_text = """
python train_model.py -m train -c ../../config/{data_folder}/{config_file_name}_v{k}.txt -t ../../data/ustancebr/v2/{data_folder}/final_bo_train.csv -v ../../data/ustancebr/v2/{data_folder}/final_bo_valid.csv -p ../../data/ustancebr/v2/{data_folder}/final_bo_test.csv -n bo -e 5 -s 1 > ../../out/ustancebr/log/{data_folder}/bo_{config_file_name}_v{k}.txt
python train_model.py -m train -c ../../config/{data_folder}/{config_file_name}_v{k}.txt -t ../../data/ustancebr/v2/{data_folder}/final_lu_train.csv -v ../../data/ustancebr/v2/{data_folder}/final_lu_valid.csv -p ../../data/ustancebr/v2/{data_folder}/final_lu_test.csv -n lu -e 5 -s 1 > ../../out/ustancebr/log/{data_folder}/lu_{config_file_name}_v{k}.txt
python train_model.py -m train -c ../../config/{data_folder}/{config_file_name}_v{k}.txt -t ../../data/ustancebr/v2/{data_folder}/final_co_train.csv -v ../../data/ustancebr/v2/{data_folder}/final_co_valid.csv -p ../../data/ustancebr/v2/{data_folder}/final_co_test.csv -n co -e 5 -s 1 > ../../out/ustancebr/log/{data_folder}/co_{config_file_name}_v{k}.txt
python train_model.py -m train -c ../../config/{data_folder}/{config_file_name}_v{k}.txt -t ../../data/ustancebr/v2/{data_folder}/final_cl_train.csv -v ../../data/ustancebr/v2/{data_folder}/final_cl_valid.csv -p ../../data/ustancebr/v2/{data_folder}/final_cl_test.csv -n cl -e 5 -s 1 > ../../out/ustancebr/log/{data_folder}/cl_{config_file_name}_v{k}.txt
python train_model.py -m train -c ../../config/{data_folder}/{config_file_name}_v{k}.txt -t ../../data/ustancebr/v2/{data_folder}/final_gl_train.csv -v ../../data/ustancebr/v2/{data_folder}/final_gl_valid.csv -p ../../data/ustancebr/v2/{data_folder}/final_gl_test.csv -n gl -e 5 -s 1 > ../../out/ustancebr/log/{data_folder}/gl_{config_file_name}_v{k}.txt
python train_model.py -m train -c ../../config/{data_folder}/{config_file_name}_v{k}.txt -t ../../data/ustancebr/v2/{data_folder}/final_ig_train.csv -v ../../data/ustancebr/v2/{data_folder}/final_ig_valid.csv -p ../../data/ustancebr/v2/{data_folder}/final_ig_test.csv -n ig -e 5 -s 1 > ../../out/ustancebr/log/{data_folder}/ig_{config_file_name}_v{k}.txt
"""
    else:
        if data_folder == "simple_domain":
            tgt_train_list = ["lu", "bo", "cl", "co", "ig", "gl"]
        else: #if data_folder == "hold1topic_out":
            tgt_train_list = ["bo", "lu", "co", "cl", "gl", "ig"]

        base_text = """
python train_model.py -m train -c ../../config/{data_folder}/{config_file_name}_v{k}.txt -t ../../data/ustancebr/v2/{data_folder}/final_bo_train.csv -v ../../data/ustancebr/v2/{data_folder}/final_bo_valid.csv -p ../../data/ustancebr/v2/{data_folder}/final_bo_test.csv -g ../../data/ustancebr/v2/simple_domain/final_""" + tgt_train_list[0] + """_train.csv -n bo -e 5 -s 1 > ../../out/ustancebr/log/{data_folder}/bo_{config_file_name}_v{k}.txt
python train_model.py -m train -c ../../config/{data_folder}/{config_file_name}_v{k}.txt -t ../../data/ustancebr/v2/{data_folder}/final_lu_train.csv -v ../../data/ustancebr/v2/{data_folder}/final_lu_valid.csv -p ../../data/ustancebr/v2/{data_folder}/final_lu_test.csv -g ../../data/ustancebr/v2/simple_domain/final_""" + tgt_train_list[1] + """_train.csv -n lu -e 5 -s 1 > ../../out/ustancebr/log/{data_folder}/lu_{config_file_name}_v{k}.txt
python train_model.py -m train -c ../../config/{data_folder}/{config_file_name}_v{k}.txt -t ../../data/ustancebr/v2/{data_folder}/final_co_train.csv -v ../../data/ustancebr/v2/{data_folder}/final_co_valid.csv -p ../../data/ustancebr/v2/{data_folder}/final_co_test.csv -g ../../data/ustancebr/v2/simple_domain/final_""" + tgt_train_list[2] + """_train.csv -n co -e 5 -s 1 > ../../out/ustancebr/log/{data_folder}/co_{config_file_name}_v{k}.txt
python train_model.py -m train -c ../../config/{data_folder}/{config_file_name}_v{k}.txt -t ../../data/ustancebr/v2/{data_folder}/final_cl_train.csv -v ../../data/ustancebr/v2/{data_folder}/final_cl_valid.csv -p ../../data/ustancebr/v2/{data_folder}/final_cl_test.csv -g ../../data/ustancebr/v2/simple_domain/final_""" + tgt_train_list[3] + """_train.csv -n cl -e 5 -s 1 > ../../out/ustancebr/log/{data_folder}/cl_{config_file_name}_v{k}.txt
python train_model.py -m train -c ../../config/{data_folder}/{config_file_name}_v{k}.txt -t ../../data/ustancebr/v2/{data_folder}/final_gl_train.csv -v ../../data/ustancebr/v2/{data_folder}/final_gl_valid.csv -p ../../data/ustancebr/v2/{data_folder}/final_gl_test.csv -g ../../data/ustancebr/v2/simple_domain/final_""" + tgt_train_list[4] + """_train.csv -n gl -e 5 -s 1 > ../../out/ustancebr/log/{data_folder}/gl_{config_file_name}_v{k}.txt
python train_model.py -m train -c ../../config/{data_folder}/{config_file_name}_v{k}.txt -t ../../data/ustancebr/v2/{data_folder}/final_ig_train.csv -v ../../data/ustancebr/v2/{data_folder}/final_ig_valid.csv -p ../../data/ustancebr/v2/{data_folder}/final_ig_test.csv -g ../../data/ustancebr/v2/simple_domain/final_""" + tgt_train_list[5] + """_train.csv -n ig -e 5 -s 1 > ../../out/ustancebr/log/{data_folder}/ig_{config_file_name}_v{k}.txt
"""

    for k in range(init_version, final_version+1):
        partial_text = base_text \
        .replace("{config_file_name}", config_file_name) \
        .replace("{k}", str(k)) \
        .replace("{data_folder}", data_folder) \

        with open(out_path, "a") as f_:
            print(partial_text, "\n\n", file=f_)
    
with open(out_path, "a") as f_:
    print("\n", file=f_, flush=True)
    

