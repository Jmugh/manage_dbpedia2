import math

import torch
import torch.nn as nn
from transformers import BertModel
import dgl
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
import torch.nn.functional as F

from common import get_heirarchy_graph
from config import Config
cfg=Config()

class MLP(nn.Module):
    def __init__(self,n_classes,in_dim=768,hidden_dim1=cfg.hidden_dim1,hidden_dim2=cfg.hidden_dim2,use_label="cooccurence_cosin_label_graph.pth"):
        super(MLP, self).__init__()
        print("--------------------MLP-------------------------")
        # self.define_pn()
        _,src,dst=get_heirarchy_graph()
        self.g=dgl.graph((torch.tensor(src),torch.tensor(dst)))
        self.gcn=dgl.nn.pytorch.GraphConv(in_feats=232,out_feats=232)
        self.A=self.g.adjacency_matrix().to_dense().to(cfg.device)
        self.g=dgl.add_self_loop(self.g).to(cfg.device)
        self.in_dim=in_dim
        self.hidden_dim1=hidden_dim1
        self.hidden_dim2=hidden_dim2
        self.n_classes=n_classes
        self.bert=BertModel.from_pretrained(cfg.bert_path)
        for p in self.bert.parameters():
            p.requires_grad_(False)
        self.unfreeze_layers = ['layer.11','bert.pooler']
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in self.unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
        self.embedding=nn.Embedding(num_embeddings=232,embedding_dim=232).to(cfg.device)# [232,232]
        self.layer_norm=nn.LayerNorm(232)
        self.linear1=nn.Linear(in_features=self.in_dim,out_features=self.hidden_dim1)
        self.relu=nn.ReLU()
        self.linear2=nn.Linear(in_features=hidden_dim1,out_features=hidden_dim2)
        self.dropout = nn.Dropout(0.3)
        self.linear3 = nn.Linear(in_features=hidden_dim2, out_features=n_classes)
        self.label_ids=torch.tensor([i for i in range(232)]).to(cfg.device)
        self.cos=nn.CosineSimilarity(dim=-1)
        self.sigmoid=nn.Sigmoid()

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        output = output.last_hidden_state#[batch_size,sequence_len,hidden_dim]
        batch_tokens=output[:,0]
        # batch_tokens=torch.mean(output,dim=1)
        output=self.linear1(batch_tokens)
        # output = self.dropout(output)
        output=self.linear2(output)
        output=self.relu(output)
        output=self.linear3(output)#[batch_size,232]
        output=output.unsqueeze(dim=1)#[batch_size,1,232]
        label_embedding=self.embedding(self.label_ids)#[232,232]
        output=torch.mul(output,label_embedding)#batch_size,232,232
        output=torch.sum(output,dim=-1)
        output=output.view(-1,232)
        output=self.sigmoid(output)#  应该把这里去掉，直接用cos表示。但是cos在[-1,1]之间,
        gcn_label_embedding=torch.matmul(label_embedding,self.A)
        return output,label_embedding,gcn_label_embedding
    def define_pn(self):
        self.positive1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.positive2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.positive3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.positive4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.positive5 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.positive6 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.positive7 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.positive1.data.clamp_(0.65, 0.7)
        self.positive2.data.clamp_(0.7, 0.75)
        self.positive3.data.clamp_(0.75, 0.8)
        self.positive4.data.clamp_(0.8, 0.85)
        self.positive5.data.clamp_(0.85, 0.9)
        self.positive6.data.clamp_(0.9, 0.95)
        self.positive7.data.clamp_(0.95, 1.0)
    def cos_sim(self,entity,label_embedding):
        entity_sum = torch.sqrt(torch.sum(torch.square(entity), dim=-1)).unsqueeze(-1)  # []
        label_sum = torch.sqrt(torch.sum(torch.square(label_embedding), dim=-1)).unsqueeze(-1)
        entity = torch.div(entity, entity_sum) * math.pi
        label_embedding = torch.div(label_embedding, label_sum) * math.pi
        cos_output = self.cos(entity, label_embedding)
        return cos_output
if __name__ == '__main__':
    # Hetero_label()

    print()
# @deprecated  gcn_label_embedding=self.gcn(self.g,label_embedding)   不需要训练参数