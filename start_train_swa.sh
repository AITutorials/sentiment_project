# 微调运行脚本:
# 任务名代表任务类型：必须选择已有标准任务中的任务，这里我们选择MNLI-MM，这和我们刚刚选择MNLI类似，
# MNLI-MM早期也是GLUE之前的标准任务之一，后来融合到MNLI，但是我们使用的这个版本的transformers仍然支持MNLI-MM。
# 定义DATA_DIR: 微调数据所在路径, 这里我们使用yxb_data中的数据作为微调数据
export DATA_DIR="./data/"
# 定义SAVE_DIR: 模型的保存路径, 我们将模型保存在当前目录的bert_finetuning_test文件中
export SAVE_DIR="./checkpoints_swa/"

# 使用python运行微调脚本
# run_glue.py : 已为大家准备好
# --model_type: 选择需要微调的模型类型, 这里可以选择BERT, XLNET, XLM, roBERTa, distilBERT, ALBERT
# --model_name_or_path: 选择具体的模型或者变体, 这里是在英文语料上微调, 因此选择bert-base-uncased
# --task_name: 它将代表对应的任务类型, 如MRPC代表句子对二分类任务
# --do_train: 使用微调脚本进行训练
# --do_eval: 使用微调脚本进行验证
# --data_dir: 训练集及其验证集所在路径, 将自动寻找该路径下的train.tsv和dev.tsv作为训练集和验证集
# --max_seq_length: 输入句子的最大长度, 超过则截断, 不足则补齐
# --learning_rate: 学习率
# --num_train_epochs: 训练轮数
# --save_steps: 检测点保存步骤间隔
# --logging_steps: 日志打印步骤间隔
# --output_dir $SAVE_DIR: 训练后的模型保存路径
python optimizer_swa.py \
  --model_type BERT \
  --model_name_or_path bert-base-cased \
  --task_name MNLI-MM \
  --is_swa True \
  --do_train \
  --do_eval \
  --data_dir $DATA_DIR/ \
  --max_seq_length 70 \
  --weight_decay 0.01 \
  --learning_rate 2e-4 \
  --num_train_epochs 50 \
  --save_steps 1000 \
  --logging_steps 1000 \
  --overwrite_output_dir \
  --output_dir $SAVE_DIR
