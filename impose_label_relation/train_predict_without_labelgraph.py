from tokenize_data import get_paded_representation_idx_and_split
from common import Data_Loader
import torch
import torch.nn as nn
from config import Config
import numpy as np
from common import loose_multilabel_evaluation
import time
from label_gcn_model import Label_GCN_Model
cfg=Config()



def print_log():
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    timestr = '时间：{}'.format(now)
    print("**********************with label graph**********************************")
    print("*" + timestr )
    print("*参数:" )
    print("*abstract_word_num=" + str(cfg.sentence_len) )
    print("*hidden_dim=" + str(cfg.hidden_dim) )
    print("*lstm_hidden_dim=" + str(cfg.lstm_hidden_dim) )
    print("*embedding_dim=" + str(cfg.embedding_dim))
    print("*epochs=" + str(cfg.epoches) )
    print("*learning_rate=" + str(cfg.learning_rate) )
    print("*batch_size=" + str(cfg.batch_size) )
    print("*dropout_lstm=" + str(cfg.dropout_lstm))
    print("*dropout_query=" + str(cfg.dropout_query) )
    print("*attention_dropout=" + str(cfg.dropout_attention) )
    print("********************************************************\n\n\n\n")


def evaluate(model,dataloader):
    model.eval()
    preds = []
    y_true = []
    for batch_abstract, batch_entityname, batch_y in dataloader:
        batch_abstract = torch.tensor(batch_abstract)
        batch_entityname = torch.tensor(batch_entityname)
        batch_y = torch.from_numpy(batch_y).to(cfg.device)
        batch_abstract = batch_abstract.to(cfg.device)
        batch_entityname = batch_entityname.to(cfg.device)

        output = lstm_attention_model(batch_abstract, batch_entityname)
        preds.extend(output.detach().cpu().numpy())
        y_true.extend(batch_y.cpu().numpy())
    y_pred = np.where(np.asarray(preds) >= 0.5, 1, 0).tolist()
    result_reader, result_writer = loose_multilabel_evaluation(y_true=y_true, y_pred=y_pred)
    return result_reader, result_writer

train_abstract_word_index,train_entitynames_word_index,train_labels,val_abstract_word_index,val_entitynames_word_index, val_labels,test_abstract_word_index,test_entitynames_word_index, test_labels,vocab_size,vector_list=get_paded_representation_idx_and_split()

train_dataloader=Data_Loader(train_abstract_word_index,train_entitynames_word_index,train_labels)
val_dataloader=Data_Loader(val_abstract_word_index,val_entitynames_word_index,val_labels)
test_dataloader=Data_Loader(test_abstract_word_index,test_entitynames_word_index,test_labels)
output_size=len(train_labels[0])


# lstm_attention_model=Label_GCN_Model(sentence_len=cfg.sentence_len,entityname_len=cfg.entityname_word_num,vocab_size=vocab_size, embedding_dim=cfg.embedding_dim, hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers,output_size=output_size,use_label=True)

lstm_attention_model=Label_GCN_Model(
    sentence_len=cfg.sentence_len,
    entityname_len=cfg.entityname_word_num,
    vocab_size=vocab_size,
    embedding_dim=cfg.embedding_dim,
    hidden_dim=cfg.hidden_dim,
    n_layers=cfg.n_layers,
    output_size=output_size,
    # use_label="cooccurence_cosin_label_graph.pth")
    use_label="heirarchy_label_graph.pth")
lstm_attention_model.embedding.weight.data=torch.tensor(vector_list)
lstm_attention_model.embedding.weight.requires_grad=False
lstm_attention_model=lstm_attention_model.to(cfg.device)


criterion=nn.BCELoss()
optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, lstm_attention_model.parameters()),lr=cfg.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
best_result=""
best_acc=0
print_log()
for epoch in range(cfg.epoches):
    t0 = time.time()
    lstm_attention_model.train()
    all_loss=[]
    start=time.time()
    preds = []
    y_true = []
    for batch_abstract,batch_entityname,batch_y in train_dataloader:
        batch_abstract=torch.tensor(batch_abstract)
        batch_entityname=torch.tensor(batch_entityname)
        batch_y=torch.from_numpy(batch_y).to(cfg.device)
        batch_abstract = batch_abstract.to(cfg.device)
        batch_entityname = batch_entityname.to(cfg.device)

        output=lstm_attention_model(batch_abstract,batch_entityname)
        loss=criterion(output,batch_y.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        all_loss.append(loss.cpu().item())
        preds.extend(output.detach().cpu().numpy())
        y_true.extend(batch_y.cpu().numpy())
    scheduler.step()
    end=time.time()
    y_pred = np.where(np.asarray(preds) >= 0.5, 1, 0).tolist()
    train_result_reader, train_result_writer = loose_multilabel_evaluation(y_true=y_true, y_pred=y_pred)
    val_result_reader, val_result_writer = evaluate(lstm_attention_model, val_dataloader)
    test_result_reader, test_result_writer = evaluate(lstm_attention_model, test_dataloader)

    if float(val_result_writer.split("\t")[0]) >= best_acc:
        best_acc = float(val_result_writer.split("\t")[0])
        best_result = "epoch:" + str(
            epoch) + "\n" + "train:" + train_result_writer + "\n" + " val:" + val_result_writer + "\n" + " test:" + test_result_writer
    print("epoch:", epoch, " loss:", sum(all_loss), "time=", time.time() - t0)
    print("train:", train_result_reader)
    print("  val:", val_result_reader)
    print(" test:", test_result_reader)
    # with open("./log.txt", "a+", encoding="utf8") as writer:
    #     writer.write("epoch:" + str(epoch) + "\n")
    #     writer.write("train:" + train_result_writer + "\n")
    #     writer.write("  val:" + val_result_writer + "\n")
    #     writer.write(" test:" + test_result_writer + "\n")

with open("./result.txt", "a+", encoding="utf8") as wr:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    timestr = '时间：{}'.format(now)
    wr.write("**********************with label graph**********************************\n")
    wr.write("*" + timestr + "\n")
    wr.write("*参数:" + "\n")
    wr.write("*abstract_word_num=" + str(cfg.sentence_len) + "\n")
    wr.write("*hidden_dim=" + str(cfg.hidden_dim) + "\n")
    wr.write("*lstm_hidden_dim=" + str(cfg.lstm_hidden_dim) + "\n")
    wr.write("*embedding_dim=" + str(cfg.embedding_dim) + "\n")
    wr.write("*epochs=" + str(cfg.epoches) + "\n")
    wr.write("*learning_rate=" + str(cfg.learning_rate) + "\n")
    wr.write("*batch_size=" + str(cfg.batch_size) + "\n")
    wr.write("*dropout_lstm=" + str(cfg.dropout_lstm) + "\n")
    wr.write("*dropout_query=" + str(cfg.dropout_query) + "\n")
    wr.write("*dropout_attention=" + str(cfg.dropout_attention) + "\n")
    wr.write("*datasets_path=" + str(cfg.datasets_path) + "\n")
    wr.write("*best_acc_result:\n" + best_result + "\n")
    wr.write("********************************************************\n\n\n\n")

