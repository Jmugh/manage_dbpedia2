import argparse
import time
import torch.nn as nn
import torch
import numpy as np
from common import loose_multilabel_evaluation
import dgl
from model import Classifier
from config import Config
parser = argparse.ArgumentParser()
parser.add_argument('--sample1', type=int, default=64, help='the input length for bert')
parser.add_argument('--sample2', type=int, default=128, help='the input length for bert')
parser.add_argument('--batch_size', type=int, default=64)
cfg=Config()
args = parser.parse_args()
cfg.sample1=args.sample1
cfg.sample2=args.sample2
cfg.batch_size=args.batch_size
g=dgl.load_graphs("./saved_graphs/hetero_graph.pth")[0][0]
train_doc_ids = torch.tensor([i for i in range(40000)])
train_label_ids = torch.tensor([i for i in range(232)])
# train_ids={"doc":train_doc_ids,"label":train_label_ids}
train_ids={"doc":train_doc_ids}

val_doc_ids = torch.tensor([i for i in range(40000,45000)])
val_label_ids = torch.tensor([i for i in range(232)])
# val_ids={"doc":val_doc_ids,"label":val_label_ids}
val_ids={"doc":val_doc_ids}

test_doc_ids = torch.tensor([i for i in range(45000,50000)])
test_label_ids = torch.tensor([i for i in range(232)])
# test_ids={"doc":test_doc_ids,"label":test_label_ids}
test_ids={"doc":test_doc_ids}
# sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
sampler = dgl.dataloading.MultiLayerNeighborSampler([cfg.sample1, cfg.sample2])
train_dataloader = dgl.dataloading.NodeDataLoader(g, train_ids, sampler, batch_size=cfg.batch_size)
val_dataloader = dgl.dataloading.NodeDataLoader(g, val_ids, sampler, batch_size=cfg.batch_size)
test_dataloader = dgl.dataloading.NodeDataLoader(g, test_ids, sampler, batch_size=cfg.batch_size)
# label_dataloader = dgl.dataloading.NodeDataLoader(g, train_label_ids, sampler, batch_size=3)
def eval(model,dataloader):
    model.eval()
    pred = []
    real = []
    for batch_src_nodes, batch_dst_nodes, batch_blocks in dataloader:  # blocks的个数和n_layers的个数相同  是层数  每个block对应一个层
        batch_inputs = {}
        batch_inputs["doc"] = g.nodes["doc"].data["feats"][batch_src_nodes["doc"]].to(cfg.device)
        batch_inputs["label"] = g.nodes["label"].data["feats"][batch_src_nodes["label"]].to(cfg.device)
        for i in range(len(batch_blocks)):
            batch_blocks[i] = batch_blocks[i].to(cfg.device)
        output_dict = model(batch_blocks, batch_inputs)
        label = g.nodes["doc"].data["label"][batch_dst_nodes["doc"]]
        real.extend(label.numpy().tolist())
        pred.extend(output_dict["doc"].cpu().detach().numpy().tolist())
    pred = np.where(np.asarray(pred) >= 0.5, 1, 0).tolist()
    result_reader, result_writer = loose_multilabel_evaluation(y_true=real, y_pred=pred)
    return result_reader, result_writer
'''
    epoch.....
'''
model=Classifier(in_dim=768, hidden_dim=256, out_dim=256,n_classes=232)
model=model.to(cfg.device)
criteron=nn.BCELoss()
optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=cfg.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
best_acc=0
best_result=""
for epoch in range(cfg.epochs):
    model.train()
    # model.lamda.requires_grad = True
    start=time.time()
    all_loss=[]
    pred = []
    real = []
    for batch_src_nodes, batch_dst_nodes, batch_blocks in train_dataloader:  # blocks的个数和n_layers的个数相同  是层数  每个block对应一个层
        batch_inputs={}
        batch_inputs["doc"]=g.nodes["doc"].data["feats"][batch_src_nodes["doc"]].to(cfg.device)
        batch_inputs["label"]=g.nodes["label"].data["feats"][batch_src_nodes["label"]].to(cfg.device)
        for i in range(len(batch_blocks)):
            batch_blocks[i]=batch_blocks[i].to(cfg.device)
        output_dict=model(batch_blocks,batch_inputs)
        label=g.nodes["doc"].data["label"][batch_dst_nodes["doc"]]
        real.extend(label.numpy().tolist())
        label=label.float().to(cfg.device)
        optimizer.zero_grad()
        loss = criteron(output_dict["doc"], label)
        loss.backward()
        optimizer.step()
        pred.extend(output_dict["doc"].cpu().detach().numpy().tolist())
        all_loss.append(loss.cpu().item())
    scheduler.step()
    pred=np.where(np.asarray(pred)>=0.5,1,0).tolist()
    train_result_reader,train_result_writer=loose_multilabel_evaluation(y_true=real,y_pred=pred)
    val_result_reader, val_result_writer = eval(model, val_dataloader)
    test_result_reader, test_result_writer = eval(model, test_dataloader)
    end = time.time()
    print("epoch:", epoch, " loss=", sum(all_loss) / len(all_loss), " time=", end - start)
    print("train:", train_result_reader)
    print(" val:", val_result_reader)
    print(" test:", test_result_reader)
    if float(val_result_writer.split("\t")[0].strip(" val:")) > best_acc:
        best_acc = float(val_result_writer.split("\t")[0].strip(" val:"))
        best_result = "\nepoch:" + str(
            epoch) + "\n" + "train:" + train_result_writer + "\n" + " val:" + val_result_writer + "\n" + " test:" + test_result_writer
    with open("log.txt", "a+", encoding="utf8") as wr:
        wr.write("epoch:" + str(epoch) + "\n")
        wr.write("train:" + train_result_writer + "\n")
        wr.write(" val:" + val_result_writer + "\n")
        wr.write(" test:" + test_result_writer + "\n")
with open("result.txt", "a+", encoding="utf8") as wr:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    timestr = '时间：{}'.format(now)
    wr.write("***********************************************\n")
    wr.write("* " + timestr + "\n")
    wr.write("* " + "sentence:" + str(cfg.sentence_len) + "\n")
    wr.write("* " + "batch_size:" + str(cfg.batch_size) + "\n")
    wr.write("* " + "best_acc_result" + best_result + "\n")
    wr.write("********************************************************\n\n\n")