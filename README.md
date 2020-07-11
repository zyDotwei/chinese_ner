# chinese_ner
一个中文简单的命名实体。有hmm、BiLSTM、BiLSTM-CRF，基于pytorch。

## 目录结构

```
.
├── data                                    # 数据文件夹
├── logging                                 # 文本日志文件夹
├── model                                   # 模型文件夹
├── run.py                                  # 主程序入口
├── utils                                   # utils
└── v_log                                   # tensorboardX日志文件夹
```

## 环境

* python 3.7
* pytorch 1.3
* seqeval
* tensorboardX
* loguru
* tqdm

## 中文数据集
数据集用的是论文[【ACL 2018Chinese NER using Lattice LSTM】](https://github.com/jiesutd/LatticeLSTM)中从新浪财经收集的简历数据。它的每一行由一个字及其对应的标注组成，虽然原文说采用BIOES（B表示实体开头，E表示实体结尾，I表示在实体内部，O表示非实体）进行标注，但github却提供了BMOES的标注方式，其实也没什么大影响。但这里为了使用seqeval进行评测，就又换了过来，即改成了BIOES。
```csv
高 B-NAME
勇 E-NAME
： O
男 O
， O
中 B-CONT
国 I-CONT
国 I-CONT
籍 E-CONT
， O
无 O
境 O
外 O
居 O
留 O
权 O
```

## 效果

|     模型      |  ACC  | F1 score |
| :-----------: | :---: | :------: |
| BiLSTM-BiLSTM | 0.957 |  0.901   |
|      HMM      | 0.915 |  9.841   |
|  BiLSTM-CRF   | 0.963 |  0.940   |

## 快速开始

```
# 训练并测试：

# BiLSTM-BiLSTM
python run.py --model bilstm

# HMM
python run.py --model hmm

# BiLSTM-CRF
python run.py --model bilstm_crf
```

