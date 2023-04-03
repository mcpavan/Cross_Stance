from itertools import product
import os

first_v_config = 101
folder = "hold1topic_out" # "simple_domain" or "hold1topic_out"
# model_name_out_file = "BertBiLSTMAttn" # BertBiLSTMAttn or BertBiLSTMJointAttn
# batch_size = 32
modelname2example = {
    "BertAAD": "./Bert_AAD_example.txt",
    "BiCondBertLstm": "./Bert_BiCondLstm_example.txt",
    "BertBiLSTMAttn": "./Bert_BiLstmAttn_example.txt",
    "BertBiLSTMJointAttn": "./Bert_BiLstmJointAttn_example.txt",
    "BertCrossNet": "./Bert_CrossNet_example.txt",
    "BertJointCL": "./Bert_JointCL_example.txt",
    "BertTOAD": "./Bert_TOAD_example.txt",
}


values_dict = {
    "BertAAD": {
        "bert_pretrained_model": [
            # "neuralmind/bert-base-portuguese-cased",
            "pablocosta/bertabaporu-base-uncased",
        ],
        "bert_layers": [
            # "-1",
            "-4,-3,-2,-1",
        ],
        "learning_rate": [
            1e-7,
        ],
        "discriminator_learning_rate": [
            1e-7,
        ],
        "discriminator_dim": [
            1024,
            3072,
        ]

    },
    "BiCondBertLstm": {
        "bert_pretrained_model": [
            # "neuralmind/bert-base-portuguese-cased",
            "pablocosta/bertabaporu-base-uncased",
        ],
        "bert_layers": [
            # "-1",
            "-4,-3,-2,-1",
        ],
        "lstm_layers": [
            "1",
            "2",
        ],
        "lstm_hidden_dim": [
            "16",
            "128",
        ],
        "learning_rate": [
            1e-7,
        ]
    },
    "BertBiLSTMAttn": {
        "bert_pretrained_model": [
            # "neuralmind/bert-base-portuguese-cased",
            "pablocosta/bertabaporu-base-uncased",
        ],
        "bert_layers": [
            # "-1",
            "-4,-3,-2,-1",
        ],
        "lstm_layers": [
            "1",
            "2",
        ],
        "lstm_hidden_dim": [
            "16",
            "128",
        ],
        "attention_density": [
            "32",
            "64",
        ],
        "attention_heads": [
            "1",
            "16",
        ],
    },
    "BertBiLSTMJointAttn": {
        "bert_pretrained_model": [
            # "neuralmind/bert-base-portuguese-cased",
            "pablocosta/bertabaporu-base-uncased",
        ],
        "bert_layers": [
            # "-1",
            "-4,-3,-2,-1",
        ],
        "lstm_layers": [
            "1",
            "2",
        ],
        "lstm_hidden_dim": [
            "16",
            "128",
        ],
        "attention_density": [
            "32",
            "64",
        ],
        "attention_heads": [
            "1",
            "16",
        ],
    },
    "BertCrossNet": {
        "bert_pretrained_model": [
            # "neuralmind/bert-base-portuguese-cased",
            "pablocosta/bertabaporu-base-uncased",
        ],
        "bert_layers": [
            # "-1",
            "-4,-3,-2,-1",
        ],
        "lstm_layers": [
            "1",
            "2",
        ],
        "lstm_hidden_dim": [
            "16",
            "128",
        ],
        "attention_density": [
            "100",
            "200",
        ],
        "learning_rate": [
            1e-7,
        ],
        # "batch_size": [
        #     64
        # ],
    },
    "BertJointCL": {
        "bert_pretrained_model": [
            # "neuralmind/bert-base-portuguese-cased",
            "pablocosta/bertabaporu-base-uncased",
        ],
        "bert_layers": [
            "-1", # only 1 layer allowed
            # "-4,-3,-2,-1",
        ],
        "gnn_dims": [
            "64,64",
            # "128,128",
            "192,192",
        ],
        "att_heads": [
            "12,12",
            # "6,6",
            "4,4",
        ],
        "learning_rate": [
            1e-7,
        ],
        "batch_size": [
            32
        ],
    },
    "BertTOAD": {
        "bert_pretrained_model": [
            # "neuralmind/bert-base-portuguese-cased",
            "pablocosta/bertabaporu-base-uncased",
        ],
        "bert_layers": [
            "-1",
            # "-4,-3,-2,-1",
        ],
        "lstm_layers": [
            "1",
            # "2",
        ],
        "lstm_hidden_dim": [
            128,
            226,
        ],
        "stance_classifier_dimension": [
            201,
            402,
            804,
        ],
        "topic_classifier_dimension": [
            140,
            280,
            420,
        ],
        "learning_rate": [
            1e-5,
        ],
        "batch_size": [
            32,
        ],
    },
}

def load_config_file(config_file_path):
    with open(config_file_path, 'r') as f:
        config = dict()
        for l in f.readlines():
            config[l.strip().split(":")[0]] = l.strip().split(":")[1]
    
    return config


out_path = "./{folder}/{model_name_out_file}_v{k}.txt"
ckp_path = "../../checkpoints/ustancebr/{name}/V{k}/"
for model_name_out_file, example_file in modelname2example.items():
    os.makedirs("/".join(out_path.split("/")[:-1]).replace("{folder}", folder), exist_ok=True)

    base_config_dict = load_config_file(example_file)

    k = first_v_config
    for combination in product(*list(values_dict[model_name_out_file].values())):

        if model_name_out_file == "BertJointCL":
            comb_dict = {}
            for key, value in zip(values_dict[model_name_out_file].keys(), combination):
                comb_dict[key] = value

            bert_out_dim = 768 # it comes from the pooler output not from the layer hidden states
            gnn_dim = int(comb_dict["gnn_dims"].split(",")[0])
            att_heads = int(comb_dict["att_heads"].split(",")[0])

            if gnn_dim * att_heads != bert_out_dim:
                continue
        
        new_config_dict = base_config_dict

        for key, value in zip(values_dict[model_name_out_file].keys(), combination):
            new_config_dict[key] = value
        
        new_config_dict["ckp_path"] = ckp_path \
            .replace(
                "{name}",
                model_name_out_file.lower().replace("bert", ""),
            ) \
            .replace("{k}", str(k))
        current_out_path = out_path \
            .replace("{folder}", folder) \
            .replace("{model_name_out_file}", model_name_out_file) \
            .replace("{k}", str(k))
        
        with open(current_out_path, "w") as f_:
            new_config_str = "\n".join(f"{k}:{v}" for k, v in new_config_dict.items())
            print(new_config_str, end="", file=f_, flush=True)
        
        k += 1
    