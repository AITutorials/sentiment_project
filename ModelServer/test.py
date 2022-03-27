# coding:utf-8
from sklearn.metrics import confusion_matrix
import requests
import pandas as pd
import numpy as np
url = "http://0.0.0.0:5003/v1/model_prediction/"
#data = {"sentence1":"Hey.","sentence2":"Hey!"}
#res = requests.post(url,json=data,timeout=200)
#print(res.text)


def request(sentence1, sentence2):
    data = {"sentence1": sentence1, "sentence2": sentence2}
    res = requests.post(url, json=data, timeout=200)
    return res.text


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def evaluate(test_file,save_file):
    df = pd.read_csv(test_file,sep='\t')
    datas = df[['sentence1','sentence2','Sentiment']].values.tolist()
    labels =df[['Sentiment']].values.tolist()
    preds = []
    for i,data in enumerate(datas):
        pred = request(data[0], data[1])
        #print(labels[i],pred,data[0],data[1])
        preds.append([pred])
    with open(save_file,'w') as f:
        for i in range(len(preds)):
            flag = 'True' if preds[i][0] == labels[i][0] else 'False'
            f.write(flag+'\t'+str(preds[i][0])+'\t'+str(labels[i][0])+'\t'+datas[i][0]+'\t'+ datas[i][1]+'\n')
    preds = np.array(preds)
    labels = np.array(labels)
    acc = simple_accuracy(preds,labels)
    confmat= confusion_matrix(y_true=labels,y_pred=preds)#输出混淆矩阵
    print("acc",acc)
    print(confmat)



if __name__ == "__main__":
    train_file = '/data/sentiment_project/data/train.tsv'
    dev_file = '/data/sentiment_project/data/dev.tsv'
    test_file = '/data/sentiment_project/data/test.tsv'
    train_pred_file = '/data/sentiment_project/data/xx/train_pred.txt'
    dev_pred_file = '/data/sentiment_project/data/xx/dev_pred.txt'
    test_pred_file  = '/data/sentiment_project/data/xx/test_pred.txt'
    evaluate(train_file,train_pred_file)
    evaluate(dev_file,dev_pred_file)
    evaluate(test_file,test_pred_file)


    
