# Flask框架固定工具
from flask import Flask
from flask import request
import os

app = Flask(__name__)

from config import model_path

# 从transformers中导入模型相关工具 
from transformers import BertForSequenceClassification,BertTokenizer, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import model_path
# 判断是否有gpu可以使用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# 将模型加载到设备
model.to(device)

# 设置模型为预测模式
model.eval()
# 标签列表
label_list = ['negative','positive','neutral'] 
# 文本对的最大长度 
MAX_LENGTH=100-2

# 定义服务请求路径和方式，使用POST请求
@app.route("/v1/model_prediction/",methods=["POST"])
def model_prediction():
    request_json = request.get_json()
    sentence1 = request_json['sentence1']
    sentence2 = request_json['sentence2']
    # 对文本对进行编码，并返回所有编码信息
    inputs = tokenizer.encode_plus(sentence1, sentence2, add_special_tokens=True, max_length=MAX_LENGTH,pad_to_max_length=True,return_tensors='pt')
    # 模型预测时 不用计算梯度
    with torch.no_grad():
        new_inputs = {
            "input_ids": inputs["input_ids"].to(device),
            "token_type_ids": inputs["token_type_ids"].to(device),
            "attention_mask": inputs["attention_mask"].to(device)}
        # 模型预测
        outputs = model(**new_inputs)
    # 获得预测标签
    predicted_labels = label_list[torch.argmax(outputs[0]).item()]
    score = max(F.softmax(outputs[0])[0])
    # 按照正常逻辑返回标签以及对应的得分列表
    return str([predicted_labels, score])    




if __name__ == '__main__':
    app.run('0.0.0.0','5003')




