import numpy as np
import pandas as pd
import time
import torch

from collections import Counter
from torch.utils.data import Dataset
from transformers import BertTokenizer

class BertStanceDataset(Dataset):
    
    def __init__(self, **kwargs):
        """
        Holds the stance dataset.

        :param data_file: path to csv data file
        :param pd_read_kwargs: kwargs to the read_csv
        :param text_col: name of the text column in the dataset
        :param topic_col: name of the topic column in the dataset
        :param label_col: name of the label column in the dataset
        :param max_seq_len_text: maximum size of text sequence
        :param max_seq_len_topic: maximum size of topic sequente
        :param pad_value: value to pad the sequences. Default=0
        :param add_special_tokens: Whether to add the special tokens to the sequences. Default=True
        :param bert_pretrained_model: name of the pretrained BERT Model. Default='bert-base-uncased'
        :param is_joint: whether to encode the topic and text jointly. Default=False`
        :param topic_first: whether to encode the topic first or after the text when is_joint=True. Ignored if is_joint=False. Default=True
        :param sample_weights: whether to generate sample weights to balance the
               dataset considering both topics and label. Default=False
        """

        self.df = pd.read_csv(
            kwargs["data_file"],
            **kwargs["pd_read_kwargs"]
        )

        self.text_col = kwargs["text_col"]
        self.topic_col = kwargs["topic_col"]
        self.label_col = kwargs["label_col"]
        self.tgt2vec, self.vec2tgt, self.tgt_cnt = create_tgt_lookup_tables(self.df[self.label_col])
        self.df["vec_target"] = [self.convert_lbl_to_vec(tgt, self.tgt2vec) for tgt in self.df[self.label_col]]
        self.n_labels = len(self.tgt_cnt)

        self.topic2vec, self.vec2topic, self.topic_cnt = create_tgt_lookup_tables(self.df[self.topic_col])
        self.df["vec_topic"] = [self.convert_lbl_to_vec(tgt, self.topic2vec) for tgt in self.df[self.topic_col]]

        self.max_seq_len_text = int(kwargs["max_seq_len_text"])
        self.max_seq_len_topic = int(kwargs["max_seq_len_topic"])

        self.pad_value = int(kwargs.get("pad_value", "0"))
        self.add_special_tokens = bool(int(kwargs.get("add_special_tokens", "1")))

        self.bert_pretrained_model = kwargs.get("bert_pretrained_model", "bert-base-uncased")
        self.is_joint = bool(int(kwargs.get("is_joint", "0")))
        self.topic_first = bool(int(kwargs.get("topic_first", "1")))
        self.sample_weights = bool(int(kwargs.get("sample_weights", "0")))
        self.weight_dict = (1/self.df[[self.topic_col, self.label_col]].value_counts(normalize=True)).to_dict()
    
        # self.df["bert_token_text"] = [[] for _ in range(len(self.df))]
        self.df["text_ids"] = [[] for _ in range(len(self.df))]
        self.df["text_token_type_ids"] = [[] for _ in range(len(self.df))]
        self.df["text_mask"] = [[] for _ in range(len(self.df))]
        self.df["text_len"] = 0
        if self.is_joint:
            self.df["is_topic_mask"] = [[] for _ in range(len(self.df))]
        
        # self.df["bert_token_topic"] = [[] for _ in range(len(self.df))]
        self.df["topic_ids"] = [[] for _ in range(len(self.df))]
        self.df["topic_token_type_ids"] = [[] for _ in range(len(self.df))]
        self.df["topic_mask"] = [[] for _ in range(len(self.df))]
        self.df["topic_len"] = 0

        self.df["weight"] = 0

        self.tokenizer = BertTokenizer.from_pretrained(
            self.bert_pretrained_model,
             do_lower_case=True
        )

        start_time = time.time()
        print(f"Processing BERT {self.bert_pretrained_model}...")

        for idx in self.df.index:
            self.df.at[idx, "weight"] = self.weight_dict[tuple(self.df.loc[idx, [self.topic_col, self.label_col]])]

            if self.is_joint:
                text = self.df.loc[idx, self.text_col]
                topic = self.df.loc[idx, self.topic_col]

                text_indices = self.tokenizer(text)
                topic_indices = self.tokenizer(topic, max_length=4)
                topic_len = np.sum(topic_indices != 0)
                text_len = np.sum(text_indices != 0)

                if self.topic_first:
                    first_sentence = topic
                    second_sentence = text
                    is_topic_mask = [1] * (topic_len + 2) + [1] * (text_len + 1)
                    is_topic_mask = pad_and_truncate(is_topic_mask, self.max_seq_len_text)
                else:
                    first_sentence = text
                    second_sentence = topic
                    is_topic_mask = [0] * (text_len + 2) + [1] * (topic_len + 1)
                    is_topic_mask = pad_and_truncate(is_topic_mask, self.max_seq_len_text)

                enc_out = self.tokenizer(
                    first_sentence,
                    text_pair = second_sentence,
                    add_special_tokens = self.add_special_tokens,
                    max_length = self.max_seq_len_text,
                    pad_to_max_length = True,
                    truncation=True,
                    return_tensors="pt"
                )


                # self.df.loc[idx, "bert_token_text"] = enc_out
                self.df.at[idx, "text_ids"] = enc_out["input_ids"][0]
                self.df.at[idx, "text_token_type_ids"] = enc_out["token_type_ids"][0]
                self.df.at[idx, "text_mask"] = enc_out["attention_mask"][0]
                self.df.at[idx, "text_len"] = enc_out["attention_mask"][0].sum()
                self.df.at[idx, "is_topic_mask"] = torch.from_numpy(is_topic_mask)
            else:
                enc_text = self.tokenizer(
                    self.df.loc[idx, self.text_col],
                    add_special_tokens = self.add_special_tokens,
                    max_length = self.max_seq_len_text,
                    pad_to_max_length = True,
                    truncation=True,
                    return_tensors="pt"
                )
                # self.df.loc[idx, "bert_token_text"] = enc_text
                self.df.at[idx, "text_ids"] = enc_text["input_ids"][0]
                self.df.at[idx, "text_token_type_ids"] = enc_text["token_type_ids"][0]
                self.df.at[idx, "text_mask"] = enc_text["attention_mask"][0]
                self.df.at[idx, "text_len"] = enc_text["attention_mask"][0].sum()

                enc_topic = self.tokenizer(
                    self.df.loc[idx, self.topic_col],
                    add_special_tokens = self.add_special_tokens,
                    max_length = self.max_seq_len_topic,
                    pad_to_max_length = True,
                    truncation=True,
                    return_tensors="pt"
                )
                # self.df.loc[idx, "bert_token_text"] = enc_topic
                self.df.at[idx, "topic_ids"] = enc_topic["input_ids"][0]
                self.df.at[idx, "topic_token_type_ids"] = enc_topic["token_type_ids"][0]
                self.df.at[idx, "topic_mask"] = enc_topic["attention_mask"][0]
                self.df.at[idx, "topic_len"] = enc_topic["attention_mask"][0].sum()
        
        total_time = time.time() - start_time
        print(f"...finished processing BERT.")
        print(f"Total time: {total_time:.2f} sec (~{total_time/len(self.df):.4f} sec/instance)")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        default_col_names = [
            "input_ids",
            "token_type_ids",
            "attention_mask",
            "input_length",
        ]

        text_cols = [
            "text_ids",
            "text_token_type_ids",
            "text_mask",
            "text_len",
        ]

        if not self.is_joint:
            text_item = self.df.loc[index, text_cols] \
                            .rename({txt_:default_ for txt_, default_ in zip(text_cols, default_col_names)}) \
                            .to_dict()

            topic_cols = [
                "topic_ids",
                "topic_token_type_ids",
                "topic_mask",
                "topic_len",
            ]
            topic_item = self.df.loc[index, topic_cols] \
                             .rename({tpc_:default_ for tpc_, default_ in zip(topic_cols, default_col_names)}) \
                             .to_dict()
            return_dict = {
                "text": text_item,
                "topic": topic_item,
                "label": self.df.loc[index, "vec_target"],
                "topic_label": self.df.loc[index, "vec_topic"],
                "index": index,
            }
        else:
            text_cols += ["is_topic_mask"]
            default_col_names += ["is_topic_mask"]
            text_item = self.df.loc[index, text_cols] \
                            .rename({txt_:default_ for txt_, default_ in zip(text_cols, default_col_names)}) \
                            .to_dict()
            
            return_dict = {
                "text": text_item,
                "label": self.df.loc[index, "vec_target"],
                "topic_label": self.df.loc[index, "vec_topic"],
                "index": index,
            }
        
        if self.sample_weights:
            return_dict["sample_weight"] = self.df.loc[index, "weight"]

        return return_dict
    
    def convert_lbl_to_vec(self, label, lbl2vec=None):
        """
        Convert a target to a vector
        :param labels: a string label
        :param lbl2vec: a dict containing the map of string labels to vectors
        :return: A vector representing the label
        """
        if not lbl2vec:
            lbl2vec = self.tgt2vec
        assert isinstance(label, str), "Target is not String"
        return lbl2vec.get(label)

    def convert_vec_to_lbl(self, vec, vec2lbl=None):
        """
        Convert a vector representing a target to an actual label
        :param vec: A vector representing a label
        :param vec2lbl: A dict containing a map of vectors to string labels
        :return: Label represented by the input vector
        """
        if not vec2lbl:
            vec2lbl = self.vec2tgt
        assert isinstance(vec, tuple) or isinstance(vec, float), "Target type is not tuple or float"
        if isinstance(vec, float)==1:
            vec = vec[0] * len(self.tgt_cnt.keys())
        return vec2lbl.get(vec) or list(self.vec2lbl.values())[0]
    
    def get_topic_list(self):
        return self.df[self.topic_col].unique().tolist()

    def get_num_topics(self):
        return len(self.get_topic_list())
    
def create_tgt_lookup_tables(targets):
    """
    Create lookup tables for targets
    :param text: List of targets for each instance
    :return: A tuple of dicts (tgt2vec, vec2tgt, tgt_cnt)
    """
    tgt_cnt = Counter(targets)
    sort_voc = sorted(tgt_cnt, key=tgt_cnt.get, reverse=True)
    vec2tgt = {}
    tgt2vec = {}

    if len(tgt_cnt)==2:
        for k, wrd in enumerate(sort_voc):
            vec2tgt[(k,)]=wrd
            tgt2vec[wrd]=(k,)
    else:
        for k, wrd in enumerate(sort_voc):
            vec = [0] * len(tgt_cnt)
            vec[k] = 1
            vec2tgt[tuple(vec)]=wrd
            tgt2vec[wrd]=tuple(vec)

    return tgt2vec, vec2tgt, tgt_cnt

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x