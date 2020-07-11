import os
import numpy as np
import torch
import torch.nn as nn


class Config:
    def __init__(self):
        self.model_name = 'bilstm_crf'
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

        self.hidden_size = 100
        self.dropout = 0.1
        self.emb_size = 200
        self.use_embedding_pretrained = True
        self.embedding_pretrained_name = 'embedding_Tencent.npz'
        self.embedding_pretrained = torch.tensor(
            np.load(os.path.join(self.data_dir, self.embedding_pretrained_name))
            ["embeddings"].astype('float32')) if self.use_embedding_pretrained else None
        self.vocab_size = 0

        # train and eval
        self.batch_size = 1
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.num_epochs = 10
        self.early_stop = False
        self.require_improvement = 200
        self.batch_to_out = 200


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.tagset_size = config.num_label + 2
        self.hidden_dim = config.hidden_size
        self.start_tag_id = config.num_label
        self.end_tag_id = config.num_label + 1
        self.device = config.device

        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.emb_size,
                                          padding_idx=config.vocab_size - 1)
            torch.nn.init.uniform_(self.embedding.weight, -0.10, 0.10)

        self.encoder = nn.LSTM(config.emb_size, config.hidden_size, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(config.hidden_size * 2, config.hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2*config.hidden_size, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.start_tag_id, :] = -10000.
        self.transitions.data[:, self.end_tag_id] = -10000.
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim).to(self.device),
                torch.randn(2, 1, self.hidden_dim).to(self.device))

    def _get_lstm_features(self, input_ids):

        embeds = self.embedding(input_ids).view(1, input_ids.shape[1], -1)
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()
        encoder_out, _ = self.encoder(embeds, self.hidden)
        decoder_out, _ = self.decoder(encoder_out, self.hidden)
        decoder_out = decoder_out.view(input_ids.shape[1], -1)
        lstm_logits = self.linear(decoder_out)
        return lstm_logits

    def log_sum_exp(self, smat):
        # 每一列的最大数
        vmax = smat.max(dim=0, keepdim=True).values
        # return (smat - vmax).exp().sum(axis=0, keepdim=True).log() + vmax
        return torch.log(torch.sum(torch.exp(smat - vmax), axis=0, keepdim=True)) + vmax

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        alphas = torch.full((1, self.tagset_size), -10000.).to(self.device)
        # 初始化分值分布. START_TAG是log(1)=0, 其他都是很小的值 "-10000"
        alphas[0][self.start_tag_id] = 0.

        # Iterate through the sentence
        for feat in feats:
            # log_sum_exp()内三者相加会广播: 当前各状态的分值分布(列向量) + 发射分值(行向量) + 转移矩阵(方形矩阵)
            # 相加所得矩阵的物理意义见log_sum_exp()函数的注释; 然后按列求log_sum_exp得到行向量
            alphas = self.log_sum_exp(alphas.T + self.transitions + feat.unsqueeze(0))
        # 最后转到EOS，发射分值为0，转移分值为列向量 self.transitions[:, self.end_tag_id]
        score = self.log_sum_exp(alphas.T + 0 + self.transitions[:, self.end_tag_id].view(-1, 1))
        return score.flatten()

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(self.device)
        tags = torch.cat([torch.tensor([self.start_tag_id], dtype=torch.long).to(self.device), tags])
        for i, feat in enumerate(feats):
            # emit = X0,start + x1,label + ... + xn-2,label + (xn-1, end[0])
            # trans = 每一个状态的转移状态
            score += self.transitions[tags[i], tags[i+1]] + feat[tags[i + 1]]
        # 加上到END_TAG的转移
        score += self.transitions[tags[-1], self.end_tag_id]
        return score

    def _viterbi_decode(self, feats):
        backtrace = []  # 回溯路径;  backtrace[i][j] := 第i帧到达j状态的所有路径中, 得分最高的那条在i-1帧是神马状态
        alpha = torch.full((1, self.tagset_size), -10000.).to(self.device)
        alpha[0][self.start_tag_id] = 0

        for frame in feats:
            smat = alpha.T + frame.unsqueeze(0) + self.transitions
            backtrace.append(smat.argmax(0))  # 当前帧每个状态的最优"来源"
            alpha = smat.max(dim=0, keepdim=True).values
        # Transition to STOP_TAG
        smat = alpha.T + 0 + self.transitions[:, self.end_tag_id].view(-1, 1)
        best_tag_id = smat.flatten().argmax().item()
        best_score = smat.max(dim=0, keepdim=True).values.item()
        best_path = [best_tag_id]

        for bptrs_t in reversed(backtrace[1:]):  # 从[1:]开始，去掉开头的 START_TAG
            best_tag_id = bptrs_t[best_tag_id].item()
            best_path.append(best_tag_id)
        best_path.reverse()
        return best_score, best_path  # 返回最优路径分值 和 最优路径

    def forward(self, sentence_ids, tags_ids):
        tags_ids = tags_ids.view(-1)
        feats = self._get_lstm_features(sentence_ids)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags_ids)
        outputs = (forward_score - gold_score, )
        _, tag_seq = self._viterbi_decode(feats)
        outputs = (tag_seq, ) + outputs
        return outputs

    def predict(self, sentence_ids):
        lstm_feats = self._get_lstm_features(sentence_ids)
        _, tag_seq = self._viterbi_decode(lstm_feats)
        return tag_seq
