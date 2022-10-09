from process_data import  split_train_test
from transformers import BertTokenizer, BertModel
from config import Config
from common import DataLoaderTokenizer
from model import MLP
import torch
import torch.nn as nn
import numpy as np
from common import loose_multilabel_evaluation,get_label_level
import time
cfg=Config()
train_x,train_entitynames,train_y,test_x,test_entitynames,test_y=split_train_test()
n_classes=len(test_y[0])
val_num=int(len(test_x)/2)
val_x=test_x[:val_num]
val_entitynames=test_entitynames[:val_num]
val_y=test_y[:val_num]
test_x=test_x[val_num:]
test_entitynames=test_entitynames[val_num:]
test_y=test_y[val_num:]
print("训练集:",len(train_x))
print("验证集:",len(val_x))
print("测试集:",len(test_x))
tokenizer=BertTokenizer.from_pretrained(cfg.bert_path)
#获取train  val  test数据集的input_ids,type_ids,attention_mask 然后获取对应的dataloader

def tokenize_data(tokenizer,x,entityname):
    print("start to tokenize data ...")
    input_ids=[]
    token_type_ids=[]
    attention_mask=[]
    for index,abstract in enumerate(x):
        output=tokenizer(abstract,entityname[index],return_tensors='pt', truncation=True, padding="max_length",max_length=cfg.sentence_len)
        input_ids.append(output["input_ids"])
        token_type_ids.append(output["token_type_ids"])
        attention_mask.append(output["attention_mask"])
        all_tokens = tokenizer.convert_ids_to_tokens(output["input_ids"].numpy()[0])
        # print(all_tokens)
    input_ids=torch.cat(input_ids,dim=0)
    token_type_ids=torch.cat(token_type_ids,dim=0)
    attention_mask=torch.cat(attention_mask,dim=0)
    return input_ids,token_type_ids,attention_mask

train_input_ids,train_token_type_ids,train_attention_mask=tokenize_data(tokenizer,train_x,train_entitynames)
val_input_ids,val_token_type_ids,val_attention_mask=tokenize_data(tokenizer,val_x,val_entitynames)
test_input_ids,test_token_type_ids,test_attention_mask=tokenize_data(tokenizer,test_x,test_entitynames)




model=MLP(n_classes=n_classes)
model=model.to(cfg.device)
criteron=nn.BCELoss()
mse=nn.MSELoss()
optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=cfg.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

train_dataloader=DataLoaderTokenizer(train_input_ids,train_token_type_ids,train_attention_mask,train_y,batch_size=cfg.batch_size)
val_dataloader=DataLoaderTokenizer(val_input_ids,val_token_type_ids,val_attention_mask,val_y,batch_size=cfg.batch_size)
test_dataloader=DataLoaderTokenizer(test_input_ids,test_token_type_ids,test_attention_mask,test_y,batch_size=cfg.batch_size)

def eval(model,dataloader):
    model.eval()
    # model.lamda.requires_grad=False
    pred = []
    real = []
    for batch_input_ids, batch_type_ids, batch_attention_mask, batch_y in dataloader:
        real.extend(batch_y)
        batch_input_ids = batch_input_ids.to(cfg.device)
        batch_type_ids = batch_type_ids.to(cfg.device)
        batch_attention_mask = batch_attention_mask.to(cfg.device)
        output,label_embedding,gcn_label_embedding = model(batch_input_ids, batch_type_ids, batch_attention_mask)
        pred.extend(output.cpu().detach().numpy().tolist())
    pred = np.where(np.asarray(pred) >= 0.5, 1, 0).tolist()
    result_reader, result_writer = loose_multilabel_evaluation(y_true=real, y_pred=pred)
    return result_reader,result_writer

print("start to train ..........")
bert=BertModel.from_pretrained(cfg.bert_path).to(cfg.device)

def train(param):
    best_acc=0
    best_result=""
    for epoch in range(cfg.epochs):
        model.train()
        # model.lamda.requires_grad = True
        start=time.time()
        all_loss=[]
        pred = []
        real = []
        # print("12->data:",model.lamda12.data,"  345->data:",model.lamda345.data,"  67->data:",model.lamda67.data)
        for batch_input_ids,batch_type_ids,batch_attention_mask,batch_y in train_dataloader:
            real.extend(batch_y)
            # ---------------------------------------------------
            # t_batch_y=[]
            # labels, levels = get_label_level()
            # for i in range(len(batch_y)):
            #     row=[]
            #     for j in range(232):
            #         if levels[j] == 1 or levels[j] == 2:
            #             row.extend(model.lamda12.data)
            #         if levels[j] == 3 or levels[j] == 4 or levels[j] == 5:
            #             row.extend(model.lamda345.data)
            #         if levels[j] == 6 or levels[j] == 7:
            #             row.extend(model.lamda67.data)
            #     t_batch_y.append(row)
            # batch_y=t_batch_y
            # ---------------------------------------------------
            batch_input_ids=batch_input_ids.to(cfg.device)
            batch_type_ids=batch_type_ids.to(cfg.device)
            batch_attention_mask=batch_attention_mask.to(cfg.device)
            batch_y=torch.tensor(batch_y).float().to(cfg.device)
            output,label_embedding,gcn_label_embedding=model(batch_input_ids,batch_type_ids,batch_attention_mask)
            optimizer.zero_grad()
            loss=criteron(output,batch_y)+param["lamda"]*mse(label_embedding,gcn_label_embedding)
            loss.backward()
            optimizer.step()
            pred.extend(output.cpu().detach().numpy().tolist())
            all_loss.append(loss.cpu().item())
        scheduler.step()
        pred=np.where(np.asarray(pred)>=0.5,1,0).tolist()
        train_result_reader,train_result_writer=loose_multilabel_evaluation(y_true=real,y_pred=pred)
        val_result_reader,val_result_writer=eval(model,val_dataloader)
        test_result_reader,test_result_writer=eval(model,test_dataloader)
        end=time.time()
        print("epoch:",epoch," loss=",sum(all_loss)/len(all_loss)," time=",end-start)
        print("train:",train_result_reader)
        print(" val:",val_result_reader)
        print(" test:",test_result_reader)
        if float(val_result_writer.split("\t")[0].strip(" val:")) > best_acc:
            best_acc = float(val_result_writer.split("\t")[0].strip(" val:"))
            best_result = "\nepoch:"+str(epoch)+"\n"+"train:"+train_result_writer+"\n"+" val:"+val_result_writer+"\n"+" test:"+test_result_writer
        with open("log.txt", "a+", encoding="utf8") as wr:
            wr.write("epoch:"+str(epoch)+"\n")
            wr.write("train:"+train_result_writer+"\n")
            wr.write(" val:"+val_result_writer+"\n")
            wr.write(" test:"+test_result_writer+"\n")
    with open("result.txt", "a+", encoding="utf8") as wr:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        timestr = '时间：{}'.format(now)
        wr.write("***********************************************\n")
        wr.write("* " + "lamda"+ str(param["lamda"]) + "\n")
        wr.write("* " + timestr + "\n")
        wr.write("* " + "sentence:"+str(cfg.sentence_len) + "\n")
        wr.write("* " + "batch_size:"+str(cfg.batch_size) + "\n")
        wr.write("* " + "best_acc_result" +best_result+ "\n")
        wr.write("********************************************************\n\n\n")
if __name__ == '__main__':
    param={}
    for lm in [0.01,0.05,0.1,0.2,0.4,0.6,0.8]:
        param["lamda"]=lm
        train(param)
