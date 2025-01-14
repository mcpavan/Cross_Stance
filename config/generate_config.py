from itertools import product
import os

first_v_config = 1
folder = "stance_vs_notstance" #"simple_domain" or "hold1topic_out" or "simple_domain_filter" or "hold1topic_out_filter" or "hc_bilstmattn_v7" or "domaincorpus_weaksup"
dataset = "ustancebr" # "ustancebr" or "semeval" or "govbr" or "govbr_semeval" or "govbr_brmoral" or "govbr_mtwitter" or "election" or "ufrgs"

modelname2example = {
    # "BertAAD": "./example/Bert_AAD_example.txt",
    # "BiCondBertLstm": "./example/Bert_BiCondLstm_example.txt",
    "BertBiLSTMAttn": "./example/Bert_BiLstmAttn_example.txt",
    "BertBiLSTMJointAttn": "./example/Bert_BiLstmJointAttn_example.txt",
    # "BertCrossNet": "./example/Bert_CrossNet_example.txt",
    # "BertJointCL": "./example/Bert_JointCL_example.txt",
    # "BertTOAD": "./example/Bert_TOAD_example.txt",
    # "Llama_4bit": "./Llama_4bit_example.txt",
    # "Llama_8bit": "./Llama_8bit_example.txt",
}

default_params_ustancebr = {
    "text_col":"Text",
    "topic_col":"Target",
    "label_col":"Polarity",
    "sample_weights": 1,
    "n_output_classes": 2,
}

default_params_ustancebr_stance_notstance = default_params_ustancebr.copy()
default_params_ustancebr_stance_notstance["label_col"] = "IsStance"

default_params_semeval = {
    "text_col":"Tweet",
    "topic_col":"Target",
    "label_col":"Stance",
    "sample_weights": 0,
    "n_output_classes": 3,
    "alpha_load_classes": 1,
}

default_params_govbr_semeval = {
    "text_col":"Tweet",
    "topic_col":"Target",
    "label_col":"Stance",
    "sample_weights": 1,
    "n_output_classes": 2,
    "alpha_load_classes": 1,
}

default_params_govbr_brmoral_mtwitter = {
    "text_col":"Text",
    "topic_col":"Target",
    "label_col":"Polarity",
    "sample_weights": 1,
    "n_output_classes": 2,
    "alpha_load_classes": 1,
}

default_params_ufrgs = default_params_semeval.copy()
default_params_ufrgs["sample_weights"] = 1


values_dict_ustancebr = {
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
        ],
        # "batch_size": [
        #     96
        # ],
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
        "batch_size": [
            "192",
        ]
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
        "batch_size": [
            "152",
        ]
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
        # "batch_size": [
        #     32
        # ],
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
            # 226
        ],
        "stance_classifier_dimension": [
            201,
            402,
            # 804,
        ],
        "topic_classifier_dimension": [
            140,
            280,
            # 420,
        ],
        "learning_rate": [
            1e-5,
        ],
        # "batch_size": [
        #     # 80,
        #     16,
        # ],
    },
    "Llama_4bit": {
        "pretrained_model_name": [
            "../../data/LLMs/ggml-alpaca-7b-q4.bin",
        ],
        "prompt": [
            {
                "prompt_template_file": "../../data/ustancebr/prompts/stance_prompt_alpaca_score10_0.md",
                "output_max_score": 10,
            }
        ],
        # "batch_size": [
        #     # 80,
        #     16,
        # ],
    },
    "Llama_8bit": {
        "model": [
            {
                "pretrained_model_name": "pablocosta/llama-7b",
                "hf_model_load_in_8bit":True,
                "hf_model_use_auth_token": "",
                "hf_tokenizer_use_auth_token": "",
            },
        ],
        "prompt": [
            {
                "prompt_template_file": "../../data/ustancebr/prompts/stance_prompt_alpaca_score10_0.md",
                "output_max_score": 10,
            }
        ],
        # "batch_size": [
        #     # 80,
        #     16,
        # ],
    },
}

values_dict_semeval = {
    "BertAAD": {
        "bert_pretrained_model": [
            "bert-base-uncased",
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
            "bert-base-uncased",
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
        ],
        "epochs":[
            20,
        ],
    },
    "BertBiLSTMAttn": {
        "bert_pretrained_model": [
            "bert-base-uncased",
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
        # "learning_rate": [
        #     1e-10,
        # ],
        "epochs":[
            20,
        ],
    },
    "BertBiLSTMJointAttn": {
        "bert_pretrained_model": [
            "bert-base-uncased",
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
        # "learning_rate": [
        #     1e-10,
        # ],
        "epochs":[
            20,
        ],
    },
    "BertCrossNet": {
        "bert_pretrained_model": [
            "bert-base-uncased",
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
        "epochs":[
            20,
        ],
        # "batch_size": [
        #     64
        # ],
    },
    "BertJointCL": {
        "bert_pretrained_model": [
            "bert-base-uncased",
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
        "epochs":[
            20,
        ],
        # "batch_size": [
        #     32
        # ],
    },
    "BertTOAD": {
        "bert_pretrained_model": [
            "bert-base-uncased",
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
            # 226
        ],
        "stance_classifier_dimension": [
            201,
            402,
            # 804,
        ],
        "topic_classifier_dimension": [
            140,
            280,
            # 420,
        ],
        "learning_rate": [
            1e-5,
        ],
        "epochs":[
            20,
        ],
        # "batch_size": [
        #     # 80,
        #     16,
        # ],
    },
    "Llama_4bit": {
        "pretrained_model_name": [
            "../../data/LLMs/ggml-alpaca-7b-q4.bin",
        ],
        "prompt": [
            {
                "prompt_template_file": "../../data/ustancebr/prompts/stance_prompt_alpaca_score10_0.md",
                "output_max_score": 10,
            }
        ],
        # "batch_size": [
        #     # 80,
        #     16,
        # ],
    },
    "Llama_8bit": {
        "model": [
            {
                "pretrained_model_name": "pablocosta/llama-7b",
                "hf_model_load_in_8bit":True,
                "hf_model_use_auth_token": "",
                "hf_tokenizer_use_auth_token": "",
            },
        ],
        "prompt": [
            {
                "prompt_template_file": "../../data/ustancebr/prompts/stance_prompt_alpaca_score10_0.md",
                "output_max_score": 10,
            }
        ],
        # "batch_size": [
        #     # 80,
        #     16,
        # ],
    },
}

values_dict_ustancebr_filter = {
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
        ],
        "epochs": [
            50
        ],
        # "batch_size": [
        #     96
        # ],
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
        "epochs": [
            50
        ],
        # "batch_size": [
        #     "192",
        # ]
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
        "epochs": [
            50
        ],
        # "batch_size": [
        #     "152",
        # ]
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
        "epochs": [
            50
        ],
        # "batch_size": [
        #     64
        # ],
    },
}

def load_config_file(config_file_path):
    with open(config_file_path, 'r') as f:
        config = dict()
        for l in f.readlines():
            config[l.strip().split(":")[0]] = l.strip().split(":")[1]
    
    return config


out_path = "./{dataset}/{folder}/{model_name_out_file}_v{k}.txt"
ckp_path = "../../checkpoints/{dataset}/{folder}/{name}/V{k}/"

if dataset in ["semeval", "election", "ufrgs"]:
    values_dict = values_dict_semeval
    if dataset in ["ufrgs", "govbr_semeval"]:
        default_params = default_params_ufrgs
    else:
        default_params = default_params_semeval
elif dataset in ["ustancebr", "govbr"]:
    values_dict = values_dict_ustancebr
    default_params = default_params_ustancebr

    if folder.endswith("_filter"):
        values_dict = values_dict_ustancebr_filter
    
    if folder in ["stance_vs_notstance"]:
        default_params = default_params_ustancebr_stance_notstance
elif dataset == "govbr_semeval":
    values_dict = values_dict_ustancebr_filter
    default_params = default_params_govbr_semeval
elif dataset in ["govbr_brmoral", "govbr_mtwitter"]:
    values_dict = values_dict_ustancebr_filter
    default_params = default_params_govbr_brmoral_mtwitter

for model_name_out_file, example_file in modelname2example.items():
    os.makedirs(
        "/".join(out_path.split("/")[:-1]) \
           .replace("{dataset}", dataset) \
           .replace("{folder}", folder),
        exist_ok=True
    )

    base_config_dict = load_config_file(example_file)
    base_config_dict["name"] = f"{model_name_out_file}_{dataset}_"
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
        # setting default params
        for key, value in default_params.items():
            new_config_dict[key] = value

        # setting combination specific params
        for key, value in zip(values_dict[model_name_out_file].keys(), combination):
            if model_name_out_file in ["Llama_8bit", "Llama_8bit"] and key in ["prompt", "model"]:
                for prompt_key, prompt_value in value.items():
                    new_config_dict[prompt_key] = prompt_value
            else:
                new_config_dict[key] = value
        
        new_config_dict["ckp_path"] = ckp_path \
            .replace("{dataset}", dataset) \
            .replace("{folder}", folder) \
            .replace(
                "{name}",
                model_name_out_file.lower().replace("bert", ""),
            ) \
            .replace("{k}", str(k))             
        current_out_path = out_path \
            .replace("{dataset}", dataset) \
            .replace("{folder}", folder) \
            .replace("{model_name_out_file}", model_name_out_file) \
            .replace("{k}", str(k))
        
        with open(current_out_path, "w") as f_:
            new_config_str = "\n".join(f"{k}:{v}" for k, v in new_config_dict.items())
            print(new_config_str, end="", file=f_, flush=True)
        
        k += 1
    