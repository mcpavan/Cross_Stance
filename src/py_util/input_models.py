import torch
import torch.nn as nn
from transformers import BertModel

# B: batch size
# T: max sequence length
# E: word embedding size
def get_BERT_Model(name=None, loader_params={}):
    if not name:
        name = 'bert-base-uncased'
    
    return BertModel.from_pretrained(name, **loader_params)

class BasicWordEmbedLayer(torch.nn.Module):
    def __init__(self, vecs, static_embeds=True, use_cuda=False):
        super(BasicWordEmbedLayer, self).__init__()
        vec_tensor = torch.tensor(vecs)
        self.static_embeds=static_embeds

        self.embeds = nn.Embedding.from_pretrained(vec_tensor, freeze=self.static_embeds)

        self.dim = vecs.shape[1]
        self.vocab_size = float(vecs.shape[0])
        self.use_cuda = use_cuda

    def forward(self, **kwargs):
        return self.embeds(kwargs['text']).type(torch.FloatTensor) # (B, T, E)

class BertLayer(torch.nn.Module):

    def __init__(self, use_cuda=False, pretrained_model_name=None, layers=None, layers_agg_type=None, loader_params={}):
        super(BertLayer, self).__init__()

        self.use_cuda = use_cuda
        self.static_embeds = True
        self.bert_layer = get_BERT_Model(name=pretrained_model_name, loader_params=loader_params)
        self.layers = layers or "-1"
        self.layers_agg_type = layers_agg_type or "concat"
        
        if self.layers_agg_type == "mean":
            self.n_layers = 1
        elif self.layers == "all":
            self.n_layers = self.bert_layer.config.num_hidden_layers
        else:
            self.n_layers = len(self.layers.split(","))
        self.dim = self.bert_layer.config.hidden_size * self.n_layers

        if self.use_cuda:
            self.bert_layer.to("cuda")

    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"].type(torch.LongTensor) # (B, max_length)
        token_type_ids = kwargs["token_type_ids"].type(torch.LongTensor) # (B, max_length)
        attention_mask = kwargs["attention_mask"].type(torch.LongTensor) # (B, max_length)

        if self.use_cuda:
            input_ids = input_ids.to("cuda")
            token_type_ids = token_type_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")
        
        bert_out = self.bert_layer(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask,
            output_hidden_states = True,
            return_dict = True
        )
        
        if self.layers == "-1":
            output = bert_out["last_hidden_state"]
        else:
            # embedding output + all layers hidden states for all tokens (13* (B, max_lenght, 768))
            all_hidden_states = bert_out["hidden_states"]
            selected_layers = self.get_hidden_state_layers(all_hidden_states, self.layers)
            output = self.aggregate_hidden_state_layers(selected_layers, self.layers_agg_type)

        return output

    def get_hidden_state_layers(self, hidden_states, layers):
        if layers == "all":
            return_list = list(hidden_states[1:])
        
        else:
            return_list = []
            layers_list = layers.split(",")
            for l_ in layers_list:
                return_list += [hidden_states[int(l_)]]

        return return_list

    def aggregate_hidden_state_layers(self, hidden_state_layers, agg_type):
        if agg_type == "mean":
            t_sum = torch.zeros_like(hidden_state_layers[0])

            for t_ in hidden_state_layers:
                t_sum = t_sum.add(t_)

            output_tensor = t_sum/len(hidden_state_layers)
        else: # agg_type == "concat"
            output_tensor = torch.cat(hidden_state_layers, -1)
        
        return output_tensor

