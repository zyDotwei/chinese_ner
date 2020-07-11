import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    def __init__(self):
        self.model_name = 'bilstm'
        # 环境配置
        self.use_cuda = True
        self.device = torch.device('cuda' if self.use_cuda and torch.cuda.is_available() else 'cpu')
        self.device_id = 0
        self.seed = 369

        # 数据配置
        self.data_dir = './data'
        self.do_lower_case = True
        self.label_list = []
        self.num_label = 0
        self.train_num_examples = 0
        self.dev_num_examples = 0
        self.test_num_examples = 0

        # logging
        self.logging_dir = './logging/' + self.model_name
        self.visual_log = './v_log/' + self.model_name

        # model
        self.max_seq_length = 256
        self.batch_size = 32
        self.hidden_size = 100
        self.dropout = 0.1
        self.emb_size = 200
        self.use_embedding_pretrained = True
        self.embedding_pretrained_name = 'embedding_Tencent.npz'
        self.embedding_pretrained = torch.tensor(
            np.load(os.path.join(self.data_dir, self.embedding_pretrained_name))
            ["embeddings"].astype('float32')) if self.use_embedding_pretrained else None
        self.vocab_size = 0
        self.ignore_index = -100

        # train and eval
        self.learning_rate = 5e-4
        self.weight_decay = 0
        self.num_epochs = 17
        self.early_stop = False
        self.require_improvement = 200
        self.batch_to_out = 50


class Model(nn.Module):
    def __init__(self, config):

        super(Model, self).__init__()
        self.n_class = config.num_label
        self.ignore_index = config.ignore_index

        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.emb_size,
                                          padding_idx=config.vocab_size-1)
            torch.nn.init.uniform_(self.embedding.weight, -0.10, 0.10)

        self.encoder = nn.LSTM(config.emb_size, config.hidden_size, dropout=config.dropout,
                               batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(config.hidden_size*2, config.hidden_size, dropout=config.dropout,
                               batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2*config.hidden_size, self.n_class)

    def forward(self,
                input_ids,
                labels=None):

        emb_out = self.embedding(input_ids)
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()
        encoder_out, _ = self.encoder(emb_out)
        decoder_out, _ = self.decoder(encoder_out)

        logits = self.linear(decoder_out)
        outputs = (logits, )

        if labels is not None:
            active_logits = logits.view(-1, self.n_class)
            active_labels = labels.view(-1)
            loss = F.cross_entropy(active_logits, active_labels, ignore_index=self.ignore_index)
            outputs = outputs + (loss, )
        return outputs
