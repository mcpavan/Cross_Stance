from copy import deepcopy
import torch
import torch.nn as nn
import model_layers as ml

def get_model_class_and_params(model_type, params):
    model_type = model_type.lower()
    if model_type == "bicond":
        return BiCondLSTMModel(
            hidden_dim=int(params["lstm_hidden_dim"]),
            text_input_dim=params["text_input_dim"],
            topic_input_dim=params["topic_input_dim"],
            num_layers=int(params.get("lstm_layers", "1")),
            drop_prob=float(params.get("dropout", "0")),
            num_labels=params["num_labels"],
            use_cuda=params["use_cuda"]
        )
    
    elif model_type == "bilstmattn":
        return BiLSTMAttentionModel(
            lstm_text_input_dim=params["text_input_dim"],
            lstm_hidden_dim=int(params["lstm_hidden_dim"]),
            lstm_num_layers=int(params.get("lstm_layers", "1")),
            lstm_drop_prob=float(params.get("lstm_drop_prob", params.get("dropout", "0"))),
            attention_density=int(params["attention_density"]),
            attention_heads=int(params["attention_heads"]),
            attention_drop_prob=float(params.get("attention_drop_prob", params.get("dropout", "0"))),
            drop_prob=float(params.get("dropout", "0")),
            num_labels=params["num_labels"],
            use_cuda=params["use_cuda"]
        )
    
    elif model_type == "bilstmjointattn":
        return BiLSTMJointAttentionModel(
            lstm_text_input_dim=params["text_input_dim"],
            lstm_topic_input_dim=params["topic_input_dim"],
            lstm_hidden_dim=int(params["lstm_hidden_dim"]),
            lstm_num_layers=int(params.get("lstm_layers", "1")),
            lstm_drop_prob=float(params.get("lstm_drop_prob", params.get("dropout", "0"))),
            attention_density=int(params.get("attention_density", None)),
            attention_heads=int(params["attention_heads"]),
            attention_drop_prob=float(params.get("attention_drop_prob", params.get("dropout", "0"))),
            drop_prob=float(params.get("dropout", "0")),
            num_labels=params["num_labels"],
            use_cuda=params["use_cuda"]
        )
    
    elif model_type == "crossnet":
        return BiCondLSTMModel(
            hidden_dim=int(params["lstm_hidden_dim"]),
            attn_dim=int(params["attn_dim"]),
            text_input_dim=params["text_input_dim"],
            topic_input_dim=params["topic_input_dim"],
            num_layers=int(params.get("lstm_layers", "1")),
            drop_prob=float(params.get("dropout", "0")),
            num_labels=params["num_labels"],
            use_cuda=params["use_cuda"]
        )

class BiCondLSTMModel(torch.nn.Module):
    '''
    Bidirectional Coniditional Encoding LSTM (Augenstein et al, 2016, EMNLP)
    Single layer bidirectional LSTM where initial states are from the topic encoding.
    Topic is also with a bidirectional LSTM. Prediction done with a single layer FFNN with
    tanh then softmax, to use cross-entropy loss.
    '''

    def __init__(self, hidden_dim, text_input_dim, topic_input_dim, num_layers=1, drop_prob=0, num_labels=3, use_cuda=False):
        super(BiCondLSTMModel, self).__init__()
        
        self.use_cuda = use_cuda
        self.num_labels = num_labels
        self.output_dim = 1 if self.num_labels == 2 else self.num_labels

        self.bilstm = ml.BiCondLSTMLayer(
            hidden_dim = hidden_dim,
            text_input_dim = text_input_dim,
            topic_input_dim = topic_input_dim,
            num_layers = num_layers,
            use_cuda = use_cuda
        )

        self.dropout = nn.Dropout(p=drop_prob) #dropout on last layer
        self.pred_layer = ml.PredictionLayer(
            input_dim = 2 * num_layers * hidden_dim,
            output_dim = self.output_dim,
            use_cuda=use_cuda
        )

    def forward(self, text_embeddings, topic_embeddings, text_length, topic_length):
        text_embeddings = text_embeddings.transpose(0, 1) # (T, B, E)
        topic_embeddings = topic_embeddings.transpose(0, 1) # (C, B, E)

        _, combo_fb_hm, _, _ = self.bilstm(text_embeddings, topic_embeddings, text_length, topic_length) # (dir*Hidden*N_layers, B)

        #dropout
        combo_fb_hm = self.dropout(combo_fb_hm) # (B, dir*Hidden*N_layers)
        y_pred = self.pred_layer(combo_fb_hm) # (B, 2)
        return y_pred

class BiLSTMJointAttentionModel(torch.nn.Module):
    '''
    Text    -> Embedding    -> Bidirectional LSTM   - A
    Topic   -> Embedding    -> Bidirectional LSTM   - A
    A = Multihead Attention Mechanism -> Dense -> Softmax
    '''

    def __init__(self, lstm_text_input_dim=768, lstm_topic_input_dim=768, lstm_hidden_dim=20, lstm_num_layers=1, lstm_drop_prob=0,
                 attention_density=None, attention_heads=4, attention_drop_prob=0, drop_prob=0, num_labels=3, use_cuda=False):
        super(BiLSTMJointAttentionModel, self).__init__()
        
        self.use_cuda = use_cuda
        self.num_labels = num_labels
        self.output_dim = 1 if self.num_labels == 2 else self.num_labels

        self.bilstm = ml.BiLSTMJointAttentionLayer(
            lstm_topic_input_dim=lstm_topic_input_dim,
            lstm_text_input_dim=lstm_text_input_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            lstm_dropout=lstm_drop_prob,
            attention_density=attention_density,
            attention_heads=attention_heads,
            attention_dropout=attention_drop_prob,
            use_cuda=use_cuda,
        )

        self.dropout = nn.Dropout(p=drop_prob) #dropout on last layer
        self.pred_layer = ml.PredictionLayer(
            input_dim = None,
            output_dim = self.output_dim,
            use_cuda=use_cuda
        )

    def forward(self, text_embeddings, topic_embeddings, text_length, topic_length):
        text_embeddings = text_embeddings.transpose(0, 1) # (T, B, E)
        topic_embeddings = topic_embeddings.transpose(0, 1) # (C, B, E)

        bilstm_return_dict = self.bilstm(text_embeddings, topic_embeddings, text_length, topic_length)

        #dropout
        attention_dropout = self.dropout(bilstm_return_dict["attention_output"]) # (B, text_len*attn_den)

        y_pred = self.pred_layer(attention_dropout) # (B, 2)

        return y_pred

class BiLSTMAttentionModel(torch.nn.Module):
    '''
    Text -> Embedding -> Bidirectional LSTM -> Multihead Self-Attention Mechanism -> Dense -> Softmax
    '''

    def __init__(self, lstm_text_input_dim=768, lstm_hidden_dim=20, lstm_num_layers=1, lstm_drop_prob=0,
                 attention_density=16, attention_heads=4, attention_drop_prob=0, drop_prob=0, num_labels=3, use_cuda=False):
        super(BiLSTMAttentionModel, self).__init__()
        
        self.use_cuda = use_cuda
        self.num_labels = num_labels
        self.output_dim = 1 if self.num_labels == 2 else self.num_labels

        self.bilstm = ml.BiLSTMAttentionLayer(
            lstm_text_input_dim=lstm_text_input_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            lstm_dropout=lstm_drop_prob,
            attention_density=attention_density,
            attention_heads=attention_heads,
            attention_dropout=attention_drop_prob,
            use_cuda=use_cuda,
        )

        self.dropout = nn.Dropout(p=drop_prob) #dropout on last layer
        self.pred_layer = ml.PredictionLayer(
            input_dim = None,
            output_dim = self.output_dim,
            use_cuda=use_cuda
        )

    def forward(self, text_embeddings, text_length):
        text_embeddings = text_embeddings.transpose(0, 1) # (T, B, E)
        
        bilstm_return_dict = self.bilstm(text_embeddings, text_length)

        #dropout
        attention_dropout = self.dropout(bilstm_return_dict["attention_output"]) # (B, text_len*attn_den)

        y_pred = self.pred_layer(attention_dropout) # (B, 2)

        return y_pred

class EnsembleModel(torch.nn.Module):
    # '''
    # Text -> Embedding -> gating model (bilstmattn/bilstmjointattn/crossnet/bicond) -> CLF_Weights
    # CLF_Weights * clf_preds -> final ensemble pred
    # '''

    def __init__(self, gating_params=None, num_labels=3, use_cuda=False):
        super(EnsembleModel, self).__init__()
        
        self.use_cuda = use_cuda
        self.n_clfs = 2
        self.num_labels = num_labels
        self.output_dim = 1 if self.num_labels == 2 else self.num_labels
        self.is_ensemble = True
        new_gating_params_ = deepcopy(gating_params)
        new_gating_params_["use_cuda"] = self.use_cuda
        new_gating_params_["num_labels"] = self.n_clfs*self.output_dim
        del new_gating_params_["model_type"]

        self.gating = get_model_class_and_params(
            gating_params["model_type"],
            new_gating_params_
        )

    def forward(self, text_embeddings, text_length, topic_embeddings=None, topic_length=None, clf1_pred=None, clf2_pred=None):
        input_params = {
            "text_embeddings": text_embeddings, # (T, B, E)
            "text_length": text_length,
        }
        
        if topic_embeddings is not None:
            input_params["topic_embeddings"] = topic_embeddings # (C, B, E)
            input_params["topic_length"] = topic_length
        
        # print("input_params", input_params)
        gating_pred = self.gating.forward(**input_params) # (B, n_clfs*output_dim)
        if self.n_clfs*self.output_dim == 2:
            gating_pred = torch.cat([gating_pred, 1-gating_pred], dim=1) #(B, 2)

        gating_pred = gating_pred.reshape(-1, self.n_clfs, self.output_dim) #(B, n_clf, output_dim)
        
        # print("gating_pred", gating_pred)

        # print("clf1_pred", clf1_pred)
        # print("clf2_pred", clf2_pred)
        # print()
        clf_pred = torch.stack([clf1_pred, clf2_pred], dim=1).reshape(-1, self.n_clfs, self.output_dim) # (n_clfs, n_outputs)
        if self.use_cuda:
            clf_pred = clf_pred.to("cuda")
        # print("clf_pred", clf_pred)
        # print("clf_pred.device", clf_pred.device)
        # print("gating_pred.device", gating_pred.device)
        y_pred = clf_pred.multiply(gating_pred).sum(dim=1) #(n_outputs,)
        # print("y_pred", y_pred)
        return y_pred


class CrossNet(torch.nn.Module):
    '''
    Cross Net (Xu et al. 2018)
    Cross-Target Stance Classification with Self-Attention Networks
    BiCond + Aspect Attention Layer
    '''
    def __init__(self, hidden_dim, attn_dim, text_input_dim, topic_input_dim, num_layers=1, drop_prob=0, num_labels=3, use_cuda=False):
        super(CrossNet, self).__init__()

        self.use_cuda = use_cuda
        self.num_labels = num_labels
        self.output_dim = 1 if self.num_labels == 2 else self.num_labels
        self.text_input_dim = text_input_dim
        self.topic_input_dim = topic_input_dim
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim

        self.crossNet_layer = ml.CrossNetLayer(
            hidden_dim=self.hidden_dim,
            attn_dim=self.attn_dim,
            text_input_dim=self.text_input_dim,
            topic_input_dim=self.topic_input_dim,
            num_layers=num_layers,
            dropout_prob=drop_prob,
            use_cuda=self.use_cuda
        )

        self.dropout = nn.Dropout(p=drop_prob) #dropout on last layer
        self.pred_layer = ml.PredictionLayer(
            input_dim = 2 * self.hidden_dim,#2 * num_layers * self.hidden_dim,
            output_dim = self.output_dim,
            use_cuda=use_cuda
        )

    def forward(self, text_embeddings, topic_embeddings, text_length, topic_length):
        text_embeddings = text_embeddings.transpose(0, 1) # (T, B, E)
        topic_embeddings = topic_embeddings.transpose(0, 1) # (C, B, E)

        _, att_vec, _ = self.crossNet_layer(text_embeddings, topic_embeddings, text_length, topic_length)

        #dropout
        att_vec_drop = self.dropout(att_vec) # (B, H*N, dir * N_layers)

        y_pred = self.pred_layer(att_vec_drop) # (B, 2)

        return y_pred

class TOAD(torch.nn.Module):
    def __init__(self, hidden_dim, text_input_dim, topic_input_dim,
                       stance_dim, topic_dim, num_topics, proj_layer_dim=128,
                       num_layers=1, num_labels=3, drop_prob=0, use_cuda=False):
        super(TOAD, self).__init__()

        self.hidden_dim = hidden_dim
        self.text_input_dim = text_input_dim
        self.topic_input_dim = topic_input_dim
        self.stance_dim = stance_dim
        self.topic_dim = topic_dim
        self.num_topics = num_topics
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.output_dim = self.num_labels #1 if self.num_labels <= 2 else self.num_labels
        self.use_cuda = use_cuda
        self.proj_layer_dim = proj_layer_dim

        self.text_proj_layer = nn.Linear(
            in_features=self.text_input_dim,
            out_features=self.proj_layer_dim,
            bias=False,
        )

        self.topic_proj_layer = nn.Linear(
            in_features=self.topic_input_dim,
            out_features=self.proj_layer_dim,
            bias=False,
        )
        if self.use_cuda:
            self.text_proj_layer.to("cuda")
            self.topic_proj_layer.to("cuda")
        
        self.enc = ml.BiCondLSTMLayer(
            hidden_dim=self.hidden_dim,
            text_input_dim=self.proj_layer_dim,#self.text_input_dim,
            topic_input_dim=self.proj_layer_dim,#self.topic_input_dim,
            num_layers=self.num_layers,
            lstm_dropout=drop_prob,
            use_cuda=self.use_cuda,
        )

        self.att_layer = ml.TOADScaledDotProductAttentionLayer(
            input_dim=2*self.hidden_dim,
            use_cuda=self.use_cuda
        )
        
        self.in_dropout = nn.Dropout(p=drop_prob)
        self.out_dropout = nn.Dropout(p=drop_prob)

        self.text_recon_layer = ml.TOADReconstructionLayer(
            hidden_dim=self.hidden_dim,
            embed_dim=self.text_input_dim,
            use_cuda=self.use_cuda
        )

        self.topic_recon_layer = ml.TOADReconstructionLayer(
            hidden_dim=self.hidden_dim,
            embed_dim=self.topic_input_dim,
            use_cuda=self.use_cuda
        )

        self.trans_layer = ml.TOADTransformationLayer(
            input_size=2*self.hidden_dim,
            use_cuda=self.use_cuda
        )

        multiplier = 4
        self.stance_classifier = ml.TwoLayerFFNNLayer(
            input_dim=multiplier*self.hidden_dim,
            hidden_dim=self.stance_dim,
            output_dim=self.output_dim,
            activation_fn=nn.ReLU(),
            use_cuda=self.use_cuda
        )

        self.topic_classifier = ml.TwoLayerFFNNLayer(
            input_dim=2*self.hidden_dim,
            hidden_dim=topic_dim,
            output_dim=self.num_topics,
            activation_fn=nn.ReLU(),
            use_cuda=self.use_cuda
        )

    def forward(self, text_embeddings, topic_embeddings, text_length, topic_length, text_mask=None, topic_mask=None):
        # text: (B, T, E), topic: (B, C, E), text_l: (B), topic_l: (B), text_mask: (B, T), topic_mask: (B, C)

        proj_text_emb = self.text_proj_layer(text_embeddings)
        proj_topic_emb = self.topic_proj_layer(topic_embeddings)

        # apply dropout on the input
        # dropped_text = self.in_dropout(text_embeddings)
        dropped_text = self.in_dropout(proj_text_emb)

        dropped_text = dropped_text.transpose(0, 1) # (T, B, E)
        # topic_embeddings_t = topic_embeddings.transpose(0, 1) # (C, B, E)
        topic_embeddings_t = proj_topic_emb.transpose(0, 1) # (C, B, E)

        # encode the text
        text_output, _, last_top_hn, topic_output = self.enc(dropped_text, topic_embeddings_t, text_length, topic_length)

        text_output = text_output.transpose(0, 1)     #output represents the token level text encodings of size (B,T,2*H)
        topic_output = topic_output.transpose(0, 1)   #Token levek topic embeddings of size (B, C, 2*H)
        last_top_hn = last_top_hn.transpose(0, 1).reshape(-1, 2*self.hidden_dim)        #(B, 2*H)
        att_vecs = self.att_layer(text_output, last_top_hn)      #(B, 2H)

        # reconstruct the original text embeddings
        text_recon_embeds = self.text_recon_layer(text_output, text_mask) #(B, L, E)
        # reconstruct the original topic embeddings
        topic_recon_embeds = self.topic_recon_layer(topic_output, topic_mask)

        # transform the representation
        trans_reps = self.trans_layer(att_vecs) #(B, 2H)

        trans_reps = self.out_dropout(trans_reps)  # adding dropout
        last_top_hn = self.out_dropout(last_top_hn)

        # stance prediction
        # added topic input to stance classifier
        stance_input = torch.cat((trans_reps, last_top_hn), 1)      #(B, 4H)
        stance_preds = self.stance_classifier(stance_input)

        # topic prediction
        topic_preds = self.topic_classifier(trans_reps)
        topic_preds_ = self.topic_classifier(trans_reps.detach())

        pred_info = {
            'text': text_embeddings,
            'text_l': text_length,
            'topic': topic_embeddings,
            'topic_l': topic_length,
            'adv_pred': topic_preds,
            'adv_pred_': topic_preds_,
            'stance_pred': stance_preds,
            'text_recon_embeds': text_recon_embeds,
            'topic_recon_embeds': topic_recon_embeds,
        }
        return pred_info

class AAD(torch.nn.Module):
    def __init__(self, src_encoder, tgt_encoder, text_input_dim, discriminator_dim,
                 num_labels=3, drop_prob=0, use_cuda=False):
        super(AAD, self).__init__()

        self.text_input_dim = text_input_dim
        self.discriminator_dim = discriminator_dim
        self.num_labels = num_labels
        self.output_dim = 1 if self.num_labels == 2 else self.num_labels
        self.use_cuda = use_cuda

        self.src_encoder = src_encoder
        self.tgt_encoder = tgt_encoder

        self.classifier = ml.AADClassifier(
            input_dim=self.text_input_dim,
            output_dim=self.output_dim,
            use_cuda=self.use_cuda
        )

        self.discriminator = ml.AADDiscriminator(
            intermediate_size=self.discriminator_dim,
            use_cuda=self.use_cuda
        )

    def forward(self, text_embeddings, **kwargs):
        # text: (B, T, E)
        y_pred = self.classifier(text_embeddings)
        return y_pred

#JOINT CL
# https://github.com/HITSZ-HLT/JointCL/blob/4da3e0f7511366797506dcbb74e69ba455532e14/run_semeval.py#L164
from JointCL_model import BERT_SCL_Proto_Graph as JointCL
