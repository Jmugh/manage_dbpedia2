import math

import torch
import torch.nn as nn
from transformers import BertModel
import dgl
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
import torch.nn.functional as F
from config import Config
cfg=Config()

class MLP(nn.Module):
    def __init__(self,n_classes,in_dim=768,hidden_dim1=cfg.hidden_dim1,hidden_dim2=cfg.hidden_dim2,use_label="cooccurence_cosin_label_graph.pth"):
        super(MLP, self).__init__()
        print("--------------------MLP-------------------------")
        # self.lamda = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.mu = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.lamda.data = torch.tensor([0.5])
        # self.mu.data = torch.tensor([0.5])
        self.in_dim=in_dim
        self.hidden_dim1=hidden_dim1
        self.hidden_dim2=hidden_dim2
        self.n_classes=n_classes
        self.bert=BertModel.from_pretrained(cfg.bert_path)
        for p in self.bert.parameters():
            p.requires_grad_(False)
        self.embedding=nn.Embedding(num_embeddings=232,embedding_dim=232)
        self.linear1=nn.Linear(in_features=self.in_dim,out_features=self.hidden_dim1)
        self.relu=nn.ReLU()
        self.linear2=nn.Linear(in_features=hidden_dim1,out_features=hidden_dim2)
        self.dropout = nn.Dropout(0.3)
        self.linear3 = nn.Linear(in_features=hidden_dim2, out_features=n_classes)

        self.label_ids=torch.tensor([i for i in range(232)])
        self.sigmoid=nn.Sigmoid()

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        output = output.last_hidden_state#[batch_size,sequence_len,hidden_dim]
        # batch_cls=output[:,0]
        batch_tokens=torch.mean(output,dim=1)
        output=self.linear1(batch_tokens)
        output = self.dropout(output)
        output=self.linear2(output)
        output=self.relu(output)
        output=self.linear3(output)
        label_embedding=self.embedding(self.label_ids)
        output = output.cos(label_embedding)
        output=self.sigmoid(output)
        return output



if __name__ == '__main__':
    # Hetero_label()
    print()
