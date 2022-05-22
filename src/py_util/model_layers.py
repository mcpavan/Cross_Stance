import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

class TwoLayerFFNNLayer(torch.nn.Module):
    '''
    N-layer FFNN with specified nonlinear function
    must be followed with some kind of prediction layer for actual prediction
    '''

    def __init__(self, input_dim, hidden_dim, output_dim, activation_fn, use_cuda=False):
        super(TwoLayerFFNNLayer, self).__init__()

        self.use_cuda = use_cuda

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.model = torch.nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_fn,
            nn.Linear(hidden_dim, output_dim),
            activation_fn
        )

        if self.use_cuda:
            self.model.to("cuda")
    
    def forward(self, input):
        return self.model(input)

class PredictionLayer(torch.nn.Module):
    '''
    Predicition layer. linear projection followed by the specified functions
    ex: pass pred_fn=nn.Tanh()
    '''

    def __init__(self, input_dim, output_dim, use_cuda=False):
        super(PredictionLayer, self).__init__()

        self.use_cuda = use_cuda

        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if self.output_dim == 1:
            self.pred_fn = nn.Sigmoid()
        else:
            self.pred_fn = nn.Softmax()
        
        if self.input_dim:
            self.model = nn.Sequential(
                nn.Linear(input_dim, output_dim, bias=False),
                self.pred_fn
            )
        else:
            self.model = nn.Sequential(
                nn.LazyLinear(output_dim, bias=False),
                self.pred_fn
            )

        if self.use_cuda:
            self.model.to("cuda")
    
    def forward(self, input):
        return self.model(input)

class BiCondLSTMLayer(torch.nn.Module):
    '''
    Bidirection Conditional Encoding (Augenstein et al. 2016 EMNLP).
    Bidirectional LSTM with initial states from topic encoding.
    Topic encoding is also a bidirectional LSTM.
    '''

    def __init__(self, hidden_dim, text_input_dim, topic_input_dim, num_layers=1, lstm_dropout=0, use_cuda=False):
        super(BiCondLSTMLayer, self).__init__()

        self.use_cuda = use_cuda

        self.hidden_dim = hidden_dim
        self.text_input_dim = text_input_dim
        self.topic_input_dim = topic_input_dim
        self.num_layers = num_layers
        self.lstm_dropout = lstm_dropout

        self.topic_lstm = nn.LSTM(
            input_size=self.topic_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.lstm_dropout,
            bidirectional=True,
        )
        self.text_lstm = nn.LSTM(
            input_size=self.text_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.lstm_dropout,
            bidirectional=True,
        )

        if self.use_cuda:
            self.topic_lstm.to("cuda")
            self.text_lstm.to("cuda")

    def forward(self, txt_e, top_e, txt_l, top_l):
        ####################
        # txt_e = (Lx, B, E), top_e = (Lt, B, E), txt_l=(B), top_l=(B)
        ########################
        # Topic
        p_top_embeds = rnn.pack_padded_sequence(top_e, top_l, enforce_sorted=False)
        self.topic_lstm.flatten_parameters()

        #feed topic
        topic_output, last_top_hn_cn = self.topic_lstm(p_top_embeds) # (seq_ln, B, 2*H),((2, B, H), (2, B, H))
        last_top_hn = last_top_hn_cn[0] #LSTM
        padded_topic_output, _ = rnn.pad_packed_sequence(topic_output, total_length=top_e.shape[0])

        #Text
        p_text_embeds = rnn.pack_padded_sequence(txt_e, txt_l, enforce_sorted=False)
        self.text_lstm.flatten_parameters()

        #feed text conditioned on topic
        output, (txt_last_hn, _) = self.text_lstm(p_text_embeds, last_top_hn_cn) # (2, B, H)
        txt_fw_bw_hn = txt_last_hn.transpose(0, 1).reshape((-1, 2 * self.hidden_dim))
        padded_output, _ = rnn.pad_packed_sequence(output, total_length=txt_e.shape[0])
        
        return padded_output, txt_fw_bw_hn, last_top_hn, padded_topic_output

class BiLSTMJointAttentionLayer(torch.nn.Module):

    def __init__(self, lstm_topic_input_dim=768, lstm_text_input_dim=768, lstm_hidden_dim=20, lstm_num_layers=1, lstm_dropout=0,
                 attention_density=16, attention_heads=4, attention_dropout=0, use_cuda=False):
        super(BiLSTMJointAttentionLayer, self).__init__()

        self.use_cuda = use_cuda

        self.lstm_topic_input_dim = lstm_topic_input_dim
        self.lstm_text_input_dim = lstm_text_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout

        self.attention_density = attention_density
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout

        self.topic_lstm = nn.LSTM(
            input_size=self.lstm_topic_input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_num_layers,
            dropout=self.lstm_dropout,
            bidirectional=True,
        )
        self.text_lstm = nn.LSTM(
            input_size=self.lstm_text_input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_num_layers,
            dropout=self.lstm_dropout,
            bidirectional=True,
        )

        self.mha = torch.nn.MultiheadAttention(
            embed_dim=self.attention_density,
            num_heads=self.attention_heads,
            dropout=self.attention_dropout,
        )

        if self.use_cuda:
            self.topic_lstm.to("cuda")
            self.text_lstm.to("cuda")
            self.mha.to("cuda")

    def forward(self, txt_e, top_e, txt_l, top_l):
        ####################
        # txt_e = (Lx, B, E), top_e = (Lt, B, E), txt_l=(B), top_l=(B)
        ########################
        
        # Topic
        p_top_embeds = rnn.pack_padded_sequence(top_e, top_l, enforce_sorted=False)
        self.topic_lstm.flatten_parameters()
        # (text_ln, B, 2*H), ((2*N_layers, B, H), (2*N_layers, B, H))
        topic_output, (topic_last_hiddenstate, topic_last_cellstate) = self.topic_lstm(p_top_embeds)
        padded_topic_output, _ = rnn.pad_packed_sequence(topic_output, total_length=top_e.shape[0])

        #Text
        p_text_embeds = rnn.pack_padded_sequence(txt_e, txt_l, enforce_sorted=False)
        self.text_lstm.flatten_parameters()
        # (topic_ln, B, 2*H), ((2*N_layers, B, H), (2*N_layers, B, H))
        text_output, (text_last_hiddenstate, text_last_cellstate) = self.text_lstm(p_text_embeds)
        padded_text_output, _ = rnn.pad_packed_sequence(text_output, total_length=txt_e.shape[0])

        # (text_len, B, Attn_den), (B, text_len, topic_len)
        attention_output, attention_weights = self.mha(
            query=padded_text_output,
            key=padded_topic_output,
            value=padded_topic_output,
        )

        # (B, text_len * Attn_den)
        attention_output = attention_output.transpose(0, 1).reshape((len(txt_l), -1))

        return {
            "topic_lstm_output": topic_output,
            "text_lstm_output": text_output,
            "attention_output": attention_output,
            "attention_weights": attention_weights,
        }