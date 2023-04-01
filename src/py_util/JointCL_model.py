#JOINT CL
# https://github.com/HITSZ-HLT/JointCL/blob/4da3e0f7511366797506dcbb74e69ba455532e14/run_semeval.py#L164
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from JointCL_layers import GraphAttentionLayer_weight
from model_layers import PredictionLayer

class GraphNN(nn.Module):
    def __init__(self, att_heads, bert_dim, dp, gnn_dims, use_cuda=False):
        super(GraphNN, self).__init__()
        self.use_cuda = use_cuda
        in_dim = bert_dim
        self.gnn_dims = [in_dim] + [int(dim) for dim in gnn_dims.strip().split(',')]

        self.gnn_layers = len(self.gnn_dims) - 1
        self.att_heads = [int(att_head) for att_head in att_heads.strip().split(',')]
        self.gnn_layer_stack = nn.ModuleList()

        for i in range(self.gnn_layers):
            in_dim = self.gnn_dims[i] * self.att_heads[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layer_stack.append(
                GraphAttentionLayer_weight(
                    att_head=self.att_heads[i],
                    in_dim=in_dim,
                    out_dim=self.gnn_dims[i + 1],
                    dp_gnn=dp,
                    use_cuda=self.use_cuda,
                )
            )

    def forward(self, node_feature, adj):

        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            node_feature, weight = gnn_layer(node_feature, adj)

        return node_feature, weight

class BERT_SCL_Proto_Graph(nn.Module):
    def __init__(self, att_heads, bert_dim, bert_layer, dp, dropout, gnn_dims, num_labels, use_cuda=False):
        '''
        att_heads
        bert_dim
        device
        dp
        dropout
        gnn_dims
        num_labels
        '''
        super(BERT_SCL_Proto_Graph, self).__init__()
        self.use_cuda = use_cuda
        
        self.bert = bert_layer.bert_layer
        self.bert_dim = bert_dim
        self.num_labels = num_labels
        # self.output_dim = 1 if self.num_labels < 3 else self.num_labels
        self.output_dim = self.num_labels
    	
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(bert_dim*2, self.output_dim)
        # self.dense = PredictionLayer(
        #     input_dim=bert_dim*2,
        #     output_dim=self.output_dim,
        #     use_cuda=self.use_cuda,
        # )
        self.gnn = GraphNN(
            att_heads=att_heads,
            bert_dim=bert_dim,
            dp=dp,
            gnn_dims=gnn_dims,
            use_cuda=self.use_cuda
        )

        if self.use_cuda:
            self.dropout = self.dropout.to("cuda")
            self.dense = self.dense.to("cuda")

    def forward(self, inputs):
        concat_bert_indices, concat_segments_indices, centroids = inputs
        batch_size = concat_bert_indices.shape[0]
        centroids = centroids[0]
        # _, pooled_output = self.bert(concat_bert_indices, token_type_ids=concat_segments_indices)
        bert_out = self.bert(
            concat_bert_indices,
            token_type_ids=concat_segments_indices,
            return_dict=True,
        )
        pooled_output = bert_out["pooler_output"]
        pooled_output = self.dropout(pooled_output)

        # adj
        # matrix = torch.ones([batch_size, centroids.shape[0]+1, centroids.shape[0]+1])
        # feature = torch.zeros([batch_size, centroids.shape[0]+1, self.bert_dim])
        # last_node_feature = torch.zeros([batch_size, self.bert_dim])

        matrix = torch.zeros([batch_size, centroids.shape[0]+1, centroids.shape[0]+1])
        matrix[:,-1:] = 1
        matrix[:,:,-1] = 1
        feature = torch.zeros([batch_size, centroids.shape[0]+1, self.bert_dim])
        last_node_feature = torch.zeros([batch_size, self.bert_dim])

        if self.use_cuda:
            matrix = matrix.to("cuda")
            feature = feature.to("cuda")
            last_node_feature = last_node_feature.to("cuda")

        for i in range(batch_size):
            feature[i][:-1] = centroids
            feature[i][-1] = pooled_output[i]

        node_feature, weight= self.gnn(feature, matrix)

        weight = weight[:,-1:]

        for i in range(batch_size):
            last_node_feature[i] = node_feature[i][-1]

        node_for_con = F.normalize(weight, dim=2)
        pooled_output = torch.cat([pooled_output, last_node_feature], dim=1)
        logits = self.dense(pooled_output)

        return logits, node_for_con

    def prototype_encode(self, inputs):

        concat_bert_indices, concat_segments_indices = inputs
        # _, pooled_output = self.bert(concat_bert_indices, token_type_ids=concat_segments_indices)
        bert_out = self.bert(
            concat_bert_indices,
            token_type_ids=concat_segments_indices,
            return_dict=True,
        )
        pooled_output = bert_out["pooler_output"]
        pooled_output = self.dropout(pooled_output)

        feature = pooled_output.unsqueeze(1)
        feature = F.normalize(feature, dim=2)

        return feature

