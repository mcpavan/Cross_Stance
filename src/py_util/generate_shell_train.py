
config_version_map = [
    # ("BertAAD", 1, 2),
    # ("BiCondBertLstm", 1, 4),
    ("BertBiLSTMAttn", 1, 16),
    ("BertBiLSTMJointAttn", 1, 16),
    # ("BertCrossNet", 1, 8),
    # ("BertJointCL", 1, 2),
    # ("BertTOAD", 1, 4),
]

# config_file_name = "Bert_BiLSTMAttn"
data_folder = "stance_vs_notstance" #"hold1topic_out" or "simple_domain" or "simple_domain_filter" or "hold1topic_out_filter" or "hc_bilstmattn_v7" or "domaincorpus_weaksup"
dataset = "ustancebr" #"ustancebr" or "semeval" or "govbr" or "govbr_semeval" or "govbr_brmoral" or "govbr_mtwitter" or "election" or "ufrgs"

base_command = "python train_model.py -m train"
base_config = " -c ../../config/{dataset}/{data_folder}/{config_file_name}_v{k}.txt"

if dataset == "ustancebr":
    base_trn_path = " -t ../../data/ustancebr/v2/{data_folder}/final_{tgt}_train.csv"
    base_vld_path = " -v ../../data/ustancebr/v2/{data_folder}/final_{tgt}_valid.csv"
    base_tst_path = " -p ../../data/ustancebr/v2/{data_folder}/final_{tgt}_test.csv"
    
    if data_folder == "simple_domain_filter":
        target_model_list = [("lu", "bo"), ("co", "bo"), ("cl", "bo"), ("cl", "co")]
        base_tst_path = " -p ../../data/ustancebr/v2/{data_folder}/final_{testtgt}_test.csv"
    elif data_folder == "hold1topic_out_filter":
        target_model_list = ["bo"]
    else:
        target_model_list = ["bo", "lu", "co", "cl", "gl", "ig"]

elif dataset == "semeval":
    if data_folder == "simple_domain":
        target_model_list = ["a", "cc", "fm", "hc", "la"]
    elif data_folder == "hold1topic_out":
        target_model_list = ["a", "cc", "dt", "fm", "hc", "la"]
    base_trn_path = " -t ../../data/semeval/{data_folder}/final_{tgt}_train.csv"
    base_vld_path = " -v ../../data/semeval/{data_folder}/final_{tgt}_valid.csv"
    base_tst_path = " -p ../../data/semeval/{data_folder}/final_{tgt}_test.csv"

elif dataset == "govbr":
    target_model_list = ["bo", "lu", "co", "cl", "gl", "ig"] + ["ca", "dp", "dr", "gc", "gm", "rq"]
    base_trn_path = " -t ../../data/govbr/{data_folder}/final_{tgt}_train.csv"
    base_vld_path = " -v ../../data/govbr/{data_folder}/final_{tgt}_valid.csv"
    base_tst_path = " -p ../../data/ustancebr/v2/{data_folder}/final_{tgt}_test.csv"

elif dataset == "govbr_semeval":
    target_model_list = ["a", "cc", "fm", "la"]
    base_trn_path = " -t ../../data/govbr_semeval/{data_folder}/final_{tgt}_train.csv"
    base_vld_path = " -v ../../data/govbr_semeval/{data_folder}/final_{tgt}_valid.csv"
    base_tst_path = " -p ../../data/semeval/{data_folder}/final_{tgt}_test.csv"

elif dataset == "govbr_brmoral":
    # target_model_list = ["ab", "ca", "ct", "dp", "dr", "gc", "gm", "rq"]
    target_model_list = ["ca", "dp", "dr", "gc", "gm", "rq"]
    base_trn_path = " -t ../../data/govbr_brmoral/{data_folder}/final_{tgt}_train.csv"
    base_vld_path = " -v ../../data/govbr_brmoral/{data_folder}/final_{tgt}_valid.csv"
    base_tst_path = " -p ../../data/brmoral/{data_folder}/final_{tgt}_test.csv"

elif dataset == "govbr_mtwitter":
    target_model_list = ["ab", "co", "ma", "mp", "pm"]
    base_trn_path = " -t ../../data/govbr_mtwitter/{data_folder}/final_{tgt}_train.csv"
    base_vld_path = " -v ../../data/govbr_mtwitter/{data_folder}/final_{tgt}_valid.csv"
    base_tst_path = " -p ../../data/mtwitter/{data_folder}/final_{tgt}_test.csv"

elif dataset == "election":
    target_model_list = ["dt"]
    base_trn_path = " -t ../../data/election/{data_folder}/final_{tgt}_train.csv"
    base_vld_path = " -v ../../data/election/{data_folder}/final_{tgt}_valid.csv"
    base_tst_path = " -p ../../data/semeval/{data_folder}/final_{tgt}_test.csv"

elif dataset == "ufrgs":
    target_model_list = ["dt"]
    base_trn_path = " -t ../../data/ufrgs/{data_folder}/final_{tgt}_train.csv"
    base_vld_path = " -v ../../data/ufrgs/{data_folder}/final_{tgt}_valid.csv"
    base_tst_path = " -p ../../data/semeval/simple_domain/final_{tgt}_test.csv"
    base_config = " -c ../../config/{dataset}/{data_folder}/{config_file_name}_v{k}.txt"


base_others = " -n {tgt} -e 5 -s 1"
base_command = base_command + base_config + base_trn_path + base_vld_path + base_tst_path + base_others
base_out = " > ../../out/{dataset}/log/{data_folder}/{tgt}_{config_file_name}_v{k}.txt"

for config_file_name, init_version, final_version in config_version_map:
    out_path = f"./sh_auto_train_{dataset}_{data_folder}_{config_file_name}.sh"
    with open(out_path, "w") as f_:
        print("", end="", file=f_)

    base_text = "\n"

    if config_file_name == "BertAAD":

        if dataset == "ustancebr":
            tgt_trn_path = "../../data/ustancebr/v2/simple_domain/final_"
            if data_folder == "simple_domain": # manual crpss target
                tgt_train_list = ["lu", "bo", "cl", "co", "ig", "gl"]
            elif data_folder == "hold1topic_out":  # files already set up
                tgt_train_list = ["bo", "lu", "co", "cl", "gl", "ig"]
        
        elif dataset == "semeval":
            tgt_trn_path = "../../data/semeval/simple_domain/final_"
            if data_folder == "simple_domain": # manual crpss target
                tgt_train_list = ["a", "cc", "fm", "hc", "la"]
            elif data_folder == "hold1topic_out":  # files already set up
                tgt_train_list = ["a", "cc", "dt", "fm", "hc", "la"]
        

        for k, tgt in enumerate(target_model_list):
            testtgt = ""
            if isinstance(tgt, tuple):
                testtgt = tgt[1]
                tgt = "_".join(tgt)
            base_text += base_command.replace("{tgt}", tgt).replace("{testtgt}", testtgt) + " -g " + tgt_trn_path + tgt_train_list[k] + "_train.csv" + base_out.replace("{tgt}", tgt) + "\n"

    elif config_file_name == "BertJointCL" and dataset == "ustancebr":
        for k, tgt in enumerate(target_model_list):
            testtgt = ""
            if isinstance(tgt, tuple):
                testtgt = tgt[1]
                tgt = "_".join(tgt)
            base_text += base_command.replace("{tgt}", tgt).replace("{testtgt}", testtgt) + " -a 100" + base_out.replace("{tgt}", tgt) + "\n"
    
    elif dataset.startswith("govbr"):
        if dataset == "govbr":
            sample_size = {
                "bo": " -a 15000 -j 60000",
                "lu": " -a 15000 -j 20000",
                "co": " -a 15000",
                "cl": " -a 15000 -j 2500",
                "gl": " -a 15000 -j 4000",
                "ig": " -a 15000",
            }
            # for k, tgt in enumerate(target_model_list):
            #     testtgt = ""
            #     if isinstance(tgt, tuple):
            #         testtgt = tgt[1]
            #         tgt = "_".join(tgt)
            #     base_text += base_command.replace("{tgt}", tgt).replace("{testtgt}", testtgt) + sample_size[tgt] + base_out.replace("{tgt}", tgt) + "\n"

        elif dataset == "govbr_semeval":
            sample_size = {
                "a": " -a 15000 -j 30000",
            }
        elif dataset == "govbr_brmoral":
            sample_size = {
                "ct": " -a 15000",
                "dr": " -a 15000",
            }
        elif dataset == "govbr_brmoral":
            sample_size = {
                "ma": " -a 15000",
            }
        
        for k, tgt in enumerate(target_model_list):
            testtgt = ""
            if isinstance(tgt, tuple):
                testtgt = tgt[1]
                tgt = "_".join(tgt)
            base_text += base_command.replace("{tgt}", tgt).replace("{testtgt}", testtgt) + sample_size.get(tgt, "") + base_out.replace("{tgt}", tgt) + "\n"

    else:
        for k, tgt in enumerate(target_model_list):
            testtgt = ""
            if isinstance(tgt, tuple):
                testtgt = tgt[1]
                tgt = "_".join(tgt)
            base_text += base_command.replace("{tgt}", tgt).replace("{testtgt}", testtgt) + base_out.replace("{tgt}", tgt) + "\n"

    for k in range(init_version, final_version+1):
        partial_text = base_text \
        .replace("{config_file_name}", config_file_name) \
        .replace("{k}", str(k)) \
        .replace("{dataset}", dataset) \
        .replace("{data_folder}", data_folder) \

        with open(out_path, "a") as f_:
            print(partial_text, "\n\n", file=f_)
    
    with open(out_path, "a") as f_:
        print("\n", file=f_, flush=True)
    

