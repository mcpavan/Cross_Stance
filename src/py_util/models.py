import torch
import torch.nn as nn
import model_layers as ml

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

        _, combo_fb_hm, _, _ = self.bilstm(text_embeddings, topic_embeddings, text_length, topic_length)

        #dropout
        combo_fb_hm = self.dropout(combo_fb_hm) # (B, H*N, dir * N_layers)

        y_pred = self.pred_layer(combo_fb_hm) # (B, 2)

        return y_pred

class BiLSTMJointAttentionModel(torch.nn.Module):
    '''
    Bidirectional Coniditional Encoding LSTM (Augenstein et al, 2016, EMNLP)
    Single layer bidirectional LSTM where initial states are from the topic encoding.
    Topic is also with a bidirectional LSTM. Prediction done with a single layer FFNN with
    tanh then softmax, to use cross-entropy loss.
    '''

    def __init__(self, lstm_text_input_dim=768, lstm_topic_input_dim=768, lstm_hidden_dim=20, lstm_num_layers=1, lstm_drop_prob=0,
                 attention_density=16, attention_heads=4, attention_drop_prob=0, drop_prob=0, num_labels=3, use_cuda=False):
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
