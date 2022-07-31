import torch
import torch.nn as nn
import math
import dgl
import torch.nn.functional as F
from config import Config
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
import dgl.nn.functional as fn
cfg=Config()

class Label_GCN_Model(nn.Module):
    def __init__(self, sentence_len,entityname_len,vocab_size, embedding_dim, hidden_dim, n_layers,output_size,use_label="heirarchy_label_graph.pth"):
        super(Label_GCN_Model, self).__init__()
        self.lamda = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.sentence_len=sentence_len
        self.hidden_dim = hidden_dim
        self.output_size=output_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=cfg.lstm_hidden_dim, num_layers=n_layers, batch_first=True,bidirectional=cfg.bidirectional)#input_size – The number of expected features in the input x
        self.dropout_lstm = nn.Dropout(cfg.dropout_lstm)
        self.linear = nn.Linear(cfg.directions*cfg.lstm_hidden_dim, output_size)
        self.sigmoid=nn.Sigmoid()
        self.use_label=use_label
        self.dropout_query=nn.Dropout(cfg.dropout_query)
        self.dropout_attention=nn.Dropout(cfg.dropout_attention)
        self.gcn = GraphConv(in_feats=768, out_feats=output_size)
        self.gat = GATConv(in_feats=768, out_feats=output_size, num_heads=1)
        if use_label == "cooccurence_cosin_label_graph.pth":
            g_list = dgl.load_graphs(filename="../cooccurence_cosin_label_graph.pth")
            g = g_list[0][0]
            print(g)
            self.g = g.to(cfg.device)
        if use_label == "heirarchy_label_graph.pth":
            g_list = dgl.load_graphs(filename="../heirarchy_label_graph.pth")
            g = g_list[0][0]
            print(g)
            self.g = g.to(cfg.device)
    def self_attention(self, x):  # 软性注意力机制（key=value=x）
        query=self.dropout_query(x)
        d_k = query.size(-1)  # d_k为query的维度
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  # 打分机制  scores:[batch, seq_len, seq_len]
        p_attn = F.softmax(scores, dim=-1)  # 对最后一个维度归一化得分
        context = torch.matmul(p_attn, x).sum(1)  # 对权重化的x求和，[batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
        return context, p_attn
    def forward(self, abstract,entityname):
        abstract=self.embedding(abstract)# batch_size,seq_len,embedding_dim
        output, (final_hidden_state, final_cell_state) = self.lstm(abstract)# batch_size,seq_len,lstm_hidden_dim*2
        output=self.dropout_lstm(output)
        output=output.reshape(-1,self.sentence_len,cfg.lstm_hidden_dim*cfg.directions)# batch_size,seq_len*hidden_dim*2
        output, p_attn=self.self_attention(output)#batch_size,hidden_dim*2
        output=self.dropout_attention(output)
        output=self.linear(output)
        if self.use_label == "cooccurence_cosin_label_graph.pth":
            feat = self.gcn(self.g, self.g.ndata["feat"],
                            edge_weight=self.g.edata["frequency_weight"] + self.lamda * self.g.edata["cosin_weight"])
            self.g.ndata["out"] = feat
            g_out = dgl.max_nodes(self.g, feat="out")
            output = torch.mul(output, g_out)
        if self.use_label == "heirarchy_label_graph.pth":
            feat = self.gat(self.g, self.g.ndata["feat"])
            feat = feat.reshape(-1, self.output_size)
            self.g.ndata["out"] = feat
            g_out = dgl.max_nodes(self.g, feat="out")
            output = torch.mul(output, g_out)
        output = self.sigmoid(output)
        return output

# class Label_GCN_Model(nn.Module):
#     def __init__(self, sentence_len,entityname_len,vocab_size, embedding_dim, hidden_dim, n_layers,output_size,use_label=False):
#         super(Label_GCN_Model, self).__init__()
#         self.lamda = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
#         self.embedding=nn.Embedding(vocab_size,embedding_dim)
#         self.sentence_len=sentence_len
#         self.entityname_len=entityname_len
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers
#         self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=cfg.lstm_hidden_dim, num_layers=n_layers, batch_first=True,bidirectional=cfg.bidirectional)#input_size – The number of expected features in the input x
#         self.Wcm=nn.Linear(sentence_len*cfg.lstm_hidden_dim*cfg.directions,hidden_dim)
#         self.Wcm_C=nn.Linear(cfg.lstm_hidden_dim*cfg.directions,hidden_dim)
#         self.W1=nn.Linear(embedding_dim,hidden_dim)
#         self.tanh=nn.Tanh()
#         self.Wa=nn.Linear(hidden_dim,1)
#         self.softmax=nn.Softmax(dim=-1)
#         self.gelu=nn.GELU()
#         self.Wr=nn.Linear(hidden_dim*3,hidden_dim)
#         self.Wg=nn.Linear(hidden_dim*3,hidden_dim)
#         self.dropout_lstm = nn.Dropout(cfg.dropout_lstm)
#         self.fc = nn.Linear(cfg.directions*hidden_dim*2, output_size)
#         self.sigmoid=nn.Sigmoid()
#         self.use_label=use_label
#         self.dropout_query=nn.Dropout(cfg.dropout_query)
#         self.dropout_attention=nn.Dropout(cfg.dropout_attention)
#         self.gcn=GraphConv(in_feats=768,out_feats=hidden_dim*2*cfg.directions)
#         if use_label==True:
#             g_list = dgl.load_graphs(filename="../cooccurence_cosin_label_graph.pth")
#             g=g_list[0][0]
#             print(g)
#             self.g = g.to(cfg.device)
#     #x,query：[batch, seq_len, hidden_dim*2]
#     def name_attention(self, m_proj, Ch):
#         A=m_proj*(self.Wa(Ch))
#         A_hat=self.softmax(A)# batch_size,hidden_dim
#         r_c=A_hat*Ch
#         r=self.gelu(self.Wr(torch.cat([r_c,m_proj,r_c-m_proj],dim=-1)))
#         g=self.sigmoid(self.Wg(torch.cat([r_c,m_proj,r_c-m_proj],dim=-1)))
#         o=g*r+(1-g)*m_proj
#         return o#batch_size,hidden_dim
#         # x,query：[batch, seq_len, hidden_dim*2]
#     def self_attention(self, x):  # 软性注意力机制（key=value=x）
#         query=self.dropout_query(x)
#         d_k = query.size(-1)  # d_k为query的维度
#         scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  # 打分机制  scores:[batch, seq_len, seq_len]
#         p_attn = F.softmax(scores, dim=-1)  # 对最后一个维度归一化得分
#         context = torch.matmul(p_attn, x).sum(1)  # 对权重化的x求和，[batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
#         return context, p_attn
#     def forward(self, abstract,entityname):
#         abstract=self.embedding(abstract)# batch_size,seq_len,embedding_dim
#         entityname=self.embedding(entityname)# batch_size,entityname_len,embedding_dim
#         M=torch.max(entityname, dim=1).values# batch_size,embedding_dim
#         m_proj=self.tanh(self.W1(M))# batch_size,hidden_dim
#         Ch_3, (final_hidden_state, final_cell_state) = self.lstm(abstract)# batch_size,seq_len,lstm_hidden_dim*2
#         Ch_3=self.dropout_lstm(Ch_3)
#         Ch=Ch_3.reshape(-1,self.sentence_len*cfg.lstm_hidden_dim*cfg.directions)# batch_size,seq_len*hidden_dim*2
#         Ch=self.Wcm(Ch)#batch_size,hidden_dim
#         o = self.name_attention(m_proj, Ch)
#         C, p_attn=self.self_attention(Ch_3)#batch_size,hidden_dim*2
#         # C = Ch_3.sum(1)
#         C=self.dropout_attention(C)
#         C = self.Wcm_C(C)
#         cat_context = torch.cat([o, C], dim=-1)
#         if self.use_label:
#             feat=self.gcn(self.g,self.g.ndata["feat"],edge_weight =self.g.edata["frequency_weight"]+self.lamda*self.g.edata["cosin_weight"])
#             self.g.ndata["out"]=feat
#             g_out=dgl.max_nodes(self.g,feat="out")
#             output=torch.mul(cat_context,g_out)
#             linear_output=self.fc(output)
#         else:
#             linear_output=self.fc(cat_context)
#         sigmoid_output=self.sigmoid(linear_output)
#         return sigmoid_output
