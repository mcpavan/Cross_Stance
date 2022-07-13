from itertools import product
import os

first_v_config = 1
folder = "simple_domain" #"hold1topic_out" or "simple_domain"
out_path = "./{folder}/Bert_BiLSTMAttn_v{k}.txt"
os.makedirs("/".join(out_path.split("/")[:-1]).replace("{folder}", folder), exist_ok=True)


base_text = """name:BertBiLSTMAttn_ustancebr_
bert:1
bert_pretrained_model:{bert_pretrained_model}
bert_layers:{bert_layers}
bert_layers_agg:concat
lstm_layers:{lstm_layers}
lstm_hidden_dim:{lstm_hidden_dim}
dropout:0.35
attention_density:{attention_density}
attention_heads:{attention_heads}
text_col:Text
topic_col:Target
label_col:Polarity
max_seq_len_text:60
max_seq_len_topic:5
pad_value:0
add_special_tokens:1
is_joint:0
sample_weights:1
batch_size:64
learning_rate:0.001
n_output_classes:2
ckp_path:../../checkpoints/ustancebr/{folder}/V{k}/
epochs:10"""

values_dict = {
    "bert_pretrained_model": [
        # "neuralmind/bert-base-portuguese-cased",
        "pablocosta/bert-tweet-br-base",
    ],
    "bert_layers": [
        "-1",
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
}

for k, combination in enumerate(product(*list(values_dict.values())), start=first_v_config):
    new_text = base_text

    for key, value in zip(values_dict.keys(), combination):
        new_text = new_text.replace("{"+key+"}", value)

    new_text = new_text.replace("{folder}", folder).replace("{k}", str(k))
    current_out_path = out_path.replace("{folder}", folder).replace("{k}", str(k))
    with open(current_out_path, "w") as f_:
        print(new_text, file=f_, flush=True)
    