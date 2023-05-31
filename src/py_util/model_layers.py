import math
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
            self.pred_fn = nn.Softmax(dim=1)
        
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
        txt_fw_bw_hn = txt_last_hn.transpose(0, 1).reshape((-1, 2 * self.hidden_dim * self.num_layers))
        padded_output, _ = rnn.pad_packed_sequence(output, total_length=txt_e.shape[0])
        
        return padded_output, txt_fw_bw_hn, last_top_hn, padded_topic_output

class BiLSTMJointAttentionLayer(torch.nn.Module):

    def __init__(self, lstm_topic_input_dim=768, lstm_text_input_dim=768, lstm_hidden_dim=20, lstm_num_layers=1, lstm_dropout=0,
                 attention_density=None, attention_heads=4, attention_dropout=0, use_cuda=False):
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

        if self.attention_density:
            self.dense_q = nn.LazyLinear(out_features=self.attention_density)
            self.q_dropout = nn.Dropout(self.attention_dropout)
            
            self.dense_k = nn.LazyLinear(out_features=self.attention_density)
            self.k_dropout = nn.Dropout(self.attention_dropout)
            
            self.dense_v = nn.LazyLinear(out_features=self.attention_density)
            self.v_dropout = nn.Dropout(self.attention_dropout)

            self.mha = torch.nn.MultiheadAttention(
                embed_dim=self.attention_density,
                num_heads=self.attention_heads,
                dropout=self.attention_dropout,
            )
        else:
            self.mha = torch.nn.MultiheadAttention(
                embed_dim=self.lstm_hidden_dim,
                num_heads=self.attention_heads,
                dropout=self.attention_dropout,
            )

        if self.use_cuda:
            self.topic_lstm.to("cuda")
            self.text_lstm.to("cuda")
            if self.attention_density:
                self.dense_q.to("cuda")
                self.dense_k.to("cuda")
                self.dense_v.to("cuda")
            self.mha.to("cuda")

    def forward(self, txt_e, top_e, txt_l, top_l):
        ####################
        # txt_e = (Lx, B, E), top_e = (Lt, B, E), txt_l=(B), top_l=(B)
        ########################
        
        # Topic
        p_top_embeds = rnn.pack_padded_sequence(top_e, top_l, enforce_sorted=False)
        self.topic_lstm.flatten_parameters()
        # (topic_ln, B, 2*H), ((2*N_layers, B, H), (2*N_layers, B, H))
        topic_output, (topic_last_hiddenstate, topic_last_cellstate) = self.topic_lstm(p_top_embeds)
        padded_topic_output, _ = rnn.pad_packed_sequence(topic_output, total_length=top_e.shape[0])

        #Text
        p_text_embeds = rnn.pack_padded_sequence(txt_e, txt_l, enforce_sorted=False)
        self.text_lstm.flatten_parameters()
        # (text_ln, B, 2*H), ((2*N_layers, B, H), (2*N_layers, B, H))
        text_output, (text_last_hiddenstate, text_last_cellstate) = self.text_lstm(p_text_embeds)
        padded_text_output, _ = rnn.pad_packed_sequence(text_output, total_length=txt_e.shape[0])

        if self.attention_density:
            # (text_ln, B, attention_density)
            q = self.q_dropout(self.dense_q(padded_text_output))
            k = self.k_dropout(self.dense_k(padded_topic_output))
            v = self.v_dropout(self.dense_v(padded_topic_output))

            # (text_len, B, Attn_den), (B, text_len, topic_len)
            attention_output, attention_weights = self.mha(
                query=q,
                key=k,
                value=v,
            )
        else:
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

class BiLSTMAttentionLayer(torch.nn.Module):

    def __init__(self, lstm_text_input_dim=768, lstm_hidden_dim=20, lstm_num_layers=1, lstm_dropout=0,
                 attention_density=16, attention_heads=4, attention_dropout=0, use_cuda=False):
        super(BiLSTMAttentionLayer, self).__init__()

        self.use_cuda = use_cuda

        self.lstm_text_input_dim = lstm_text_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout

        self.attention_density = attention_density
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout

        self.text_lstm = nn.LSTM(
            input_size=self.lstm_text_input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_num_layers,
            dropout=self.lstm_dropout,
            bidirectional=True,
        )

        if self.attention_density:
            self.dense_q = nn.LazyLinear(out_features=self.attention_density)
            self.q_dropout = nn.Dropout(self.attention_dropout)
            
            self.dense_k = nn.LazyLinear(out_features=self.attention_density)
            self.k_dropout = nn.Dropout(self.attention_dropout)
            
            self.dense_v = nn.LazyLinear(out_features=self.attention_density)
            self.v_dropout = nn.Dropout(self.attention_dropout)

            self.mha = torch.nn.MultiheadAttention(
                embed_dim=self.attention_density,
                num_heads=self.attention_heads,
                dropout=self.attention_dropout,
            )
        else:
            self.mha = torch.nn.MultiheadAttention(
                embed_dim=self.lstm_hidden_dim,
                num_heads=self.attention_heads,
                dropout=self.attention_dropout,
            )

        if self.use_cuda:
            self.text_lstm.to("cuda")
            if self.attention_density:
                self.dense_q.to("cuda")
                self.dense_k.to("cuda")
                self.dense_v.to("cuda")
            self.mha.to("cuda")

    def forward(self, txt_e, txt_l):
        ####################
        # txt_e = (Lx, B, E), txt_l=(B)
        ########################
        
        #Text
        p_text_embeds = rnn.pack_padded_sequence(txt_e, txt_l, enforce_sorted=False)
        self.text_lstm.flatten_parameters()
        # (text_ln, B, 2*H), ((2*N_layers, B, H), (2*N_layers, B, H))
        text_output, (text_last_hiddenstate, text_last_cellstate) = self.text_lstm(p_text_embeds)
        padded_text_output, _ = rnn.pad_packed_sequence(text_output, total_length=txt_e.shape[0])
        
        if self.attention_density:
            # (text_ln, B, attention_density)
            q = self.q_dropout(self.dense_q(padded_text_output))
            k = self.k_dropout(self.dense_k(padded_text_output))
            v = self.v_dropout(self.dense_v(padded_text_output))

            # (text_len, B, Attn_den), (B, text_len, topic_len)
            attention_output, attention_weights = self.mha(
                query=q,
                key=k,
                value=v,
            )
        else:
            # (text_len, B, Attn_den), (B, text_len, topic_len)
            attention_output, attention_weights = self.mha(
                query=padded_text_output,
                key=padded_text_output,
                value=padded_text_output,
            )
        
        # (B, text_len * Attn_den)
        attention_output = attention_output.transpose(0, 1).reshape((len(txt_l), -1))

        return {
            "text_lstm_output": text_output,
            "attention_output": attention_output,
            "attention_weights": attention_weights,
        }

class CrossNetLayer(torch.nn.Module):
    '''
    Cross Net (Xu et al. 2018)
    Cross-Target Stance Classification with Self-Attention Networks
    BiCond + Aspect Attention Layer
    '''
    def __init__(self, hidden_dim, attn_dim, text_input_dim, topic_input_dim, num_layers=1, dropout_prob=0, use_cuda=False):
        super(CrossNetLayer, self).__init__()
        self.use_cuda = use_cuda

        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.text_input_dim = text_input_dim
        self.topic_input_dim = topic_input_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        self.bicond = BiCondLSTMLayer(
            hidden_dim=self.hidden_dim,
            text_input_dim=self.text_input_dim,
            topic_input_dim=self.topic_input_dim,
            num_layers=self.num_layers,
            lstm_dropout=self.dropout_prob,
            use_cuda=self.use_cuda,
        )

        self.device = 'cuda' if self.use_cuda else 'cpu'

        #aspect attention
        self.W1 = torch.empty((2* self.hidden_dim, self.attn_dim), device=self.device)
        self.W1 = nn.Parameter(nn.init.xavier_normal_(self.W1))

        self.w2 = torch.empty((self.attn_dim, 1), device=self.device)
        self.w2 = nn.Parameter(nn.init.xavier_normal_(self.w2))

        self.b1 = torch.empty((self.attn_dim, 1), device=self.device)
        self.b1 = nn.Parameter(nn.init.xavier_normal_(self.b1)).squeeze(1)

        self.b2 = torch.rand([1],  device=self.device)
        self.b2 = nn.Parameter(self.b2)

        self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, txt_e, top_e, txt_l, top_l):
        ### bicond-lstm
        padded_output, _, last_top_hn, _ = self.bicond(txt_e, top_e, txt_l, top_l)

        padded_output = self.dropout(padded_output)
        # padded_output: (L, B, 2H), txt_fw_bw_hn: (B, 2H), last_top_hn: (2, B, H)
        output = padded_output.transpose(0, 1) #(B, L, 2H)

        ### self-attnetion
        temp_c = torch.sigmoid(torch.einsum('blh,hd->bld', output, self.W1) + self.b1) #(B, L, D)
        c = torch.einsum('bld,ds->bls', temp_c, self.w2).squeeze(-1) + self.b2 #(B, L)
        a = nn.functional.softmax(c, dim=1)

        att_vec = torch.einsum('blh,bl->bh', output, a) #(B, 2H)

        return output, att_vec, last_top_hn

class TOADScaledDotProductAttentionLayer(torch.nn.Module):
    '''
    Scaled Dot Product Attention Layer used in TOAD model
    '''
    def __init__(self, input_dim, use_cuda=False):
        super(TOADScaledDotProductAttentionLayer, self).__init__()
        self.input_dim = input_dim

        self.scale = math.sqrt(2 * self.input_dim)

    def forward(self, inputs, query):
        # inputs = (B, L, 2*H), query = (B, 2*H), last_hidden=(B, 2*H)
        sim = torch.einsum('blh,bh->bl', inputs, query) / self.scale  # (B, L)
        att_weights = nn.functional.softmax(sim, dim=1)  # (B, L)
        context_vec = torch.einsum('blh,bl->bh', inputs, att_weights)  # (B, 2*H)
        return context_vec

class TOADTransformationLayer(torch.nn.Module):
    '''
    Linear transformation layer used in TOAD model
    '''
    def __init__(self, input_size, use_cuda=False):
        super(TOADTransformationLayer, self).__init__()

        self.use_cuda = use_cuda

        self.dim = input_size

        self.W = torch.empty(
            (self.dim, self.dim),
            device='cuda' if self.use_cuda else 'cpu'
        )
        self.W = nn.Parameter(nn.init.xavier_normal_(self.W)) # (D, D)

    def forward(self, text):
        # text: (B, D)
        return torch.einsum('bd,dd->bd', text, self.W)

class TOADReconstructionLayer(torch.nn.Module):
    '''
    Embedding reconstruction layer used in TOAD model
    '''
    def __init__(self, hidden_dim, embed_dim, use_cuda=False):
        super(TOADReconstructionLayer, self).__init__()

        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.use_cuda = use_cuda
        device = 'cuda' if self.use_cuda else 'cpu'

        self.recon_W = torch.empty(
            (2 * self.hidden_dim, self.embed_dim),
            device=device
        )
        self.recon_w = nn.Parameter(nn.init.xavier_normal_(self.recon_W))

        self.recon_b = torch.empty(
            (self.embed_dim, 1),
            device=device
        )
        self.recon_b = nn.Parameter(nn.init.xavier_normal_(self.recon_b)).squeeze(1)
        
        self.tanh = nn.Tanh()

    def forward(self, text_output, text_mask):
        # text_output: (B, T, H), text_mask: (B, T)
        recon_embeds = self.tanh(
            torch.einsum('blh,he->ble', text_output, self.recon_w) + self.recon_b
        )  # (B,L,E)

        recon_embeds = torch.einsum('ble,bl->ble', recon_embeds, text_mask)
        
        return recon_embeds

class AADDiscriminator(nn.Module):
    """AAD Discriminator model for source domain."""

    def __init__(self, intermediate_size, use_cuda=False):
        """Init discriminator."""
        super(AADDiscriminator, self).__init__()
        self.use_cuda = use_cuda
        self.layer = nn.Sequential(
            nn.LazyLinear(intermediate_size),
            nn.LeakyReLU(),
            nn.Linear(intermediate_size, intermediate_size),
            nn.LeakyReLU(),
            nn.Linear(intermediate_size, 1),
            nn.Sigmoid()
        )
        if self.use_cuda:
            self.layer.to("cuda")

    def forward(self, x):
        """Forward the discriminator."""
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        out = self.layer(x)
        return out

class AADClassifier(nn.Module):
    """AAD Classifier model for stance prediction."""

    def __init__(self, input_dim, output_dim, drop_prob=0, use_cuda=False):
        super(AADClassifier, self).__init__()
        
        self.use_cuda = use_cuda
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.dropout_layer = nn.Dropout(p=drop_prob)
        self.classifier = PredictionLayer(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            use_cuda=self.use_cuda
        )
    
    def forward(self, x):
        x = self.dropout_layer(x)
        out = self.classifier(x)
        return out

# JointCL
