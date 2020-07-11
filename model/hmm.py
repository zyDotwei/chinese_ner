import torch
from loguru import logger
import time
from tqdm import tqdm


class Config:
    def __init__(self):
        self.model_name = 'hmm'
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


class Model(object):

    def __init__(self, hidden_status):
        """
        :param hidden_status: int, 隐状态数
        """
        self.hmm_N = hidden_status

        # 状态转移概率矩阵 A[i][j]表示从i状态转移到j状态的概率
        self.hmm_A = torch.zeros(self.hmm_N, self.hmm_N)
        # 初始状态概率  Pi[i]表示初始时刻为状态i的概率
        self.hmm_pi = torch.zeros(self.hmm_N)

    def _build_corpus_map(self, sentences_list):
        char2id = {}
        for sentence in sentences_list:
            for word in sentence:
                if word not in char2id:
                    char2id[word] = len(char2id)
        return char2id

    def _init_emission(self):
        self.hmm_M = len(self.word2id)
        # 观测概率矩阵, B[i][j]表示i状态下生成j观测的概率
        self.hmm_B = torch.zeros(self.hmm_N, self.hmm_M)

    def train(self, sentences_list, tags_list):
        """HMM的训练，即根据训练语料对模型参数进行估计,
           因为我们有观测序列以及其对应的状态序列，所以我们
           可以使用极大似然估计的方法来 估计 隐马尔可夫模型的参数
        参数:
            sentences_list: list，其中每个元素由字组成的列表，如 ['担','任','科','员']
            tags_list: list，其中每个元素是由对应的标注组成的列表，如 ['O','O','B-TITLE', 'E-TITLE']
        """
        start_time = time.time()
        assert len(sentences_list) == len(tags_list), "the lens of tag_lists is not eq to word_lists"

        logger.info('开始构建token字典...')
        self.word2id = self._build_corpus_map(sentences_list)
        self.tag2id = self._build_corpus_map(tags_list)
        self.id2tag = dict((id_, tag) for tag, id_ in self.tag2id.items())
        logger.info('训练语料总数:{}'.format(len(sentences_list)))
        logger.info('词典总数:{}'.format(len(self.word2id)))
        logger.info('标签总数:{}'.format(len(self.tag2id)))

        assert self.hmm_N == len(self.tag2id), "hidden_status is {}, but total tag is {}".\
            format(self.hmm_N, len(self.tag2id))
        self._init_emission()
        logger.info('构建词典完成{:>.4f}s'.format(time.time()-start_time))
        logger.info('开始构建转移概率矩阵...')
        # 估计转移概率矩阵
        for tags in tqdm(tags_list):
            seq_len = len(tags)
            for i in range(seq_len - 1):
                current_tagid = self.tag2id[tags[i]]
                next_tagid = self.tag2id[tags[i+1]]
                self.hmm_A[current_tagid][next_tagid] += 1.
        # 问题：如果某元素没有出现过，该位置为0，这在后续的计算中是不允许的
        # 解决方法：我们将等于0的概率加上很小的数
        self.hmm_A[self.hmm_A == 0.] = 1e-10
        self.hmm_A = self.hmm_A / self.hmm_A.sum(axis=1, keepdims=True)
        logger.info('完成转移概率矩阵构建. {:>.4f}s'.format(time.time() - start_time))
        logger.info('开始构建观测概率矩阵...')
        # 估计观测概率矩阵
        for tags, sentence in tqdm(zip(tags_list, sentences_list)):
            assert len(tags) == len(sentence), \
                "the lens of tag_list is not eq to word_list"
            for tag, word in zip(tags, sentence):
                tag_id = self.tag2id[tag]
                word_id = self.word2id[word]
                self.hmm_B[tag_id][word_id] += 1.
        self.hmm_B[self.hmm_B == 0.] = 1e-10
        self.hmm_B = self.hmm_B / self.hmm_B.sum(axis=1, keepdims=True)
        logger.info('完成观测概率矩阵构建. {:>.4f}s'.format(time.time() - start_time))
        logger.info('初始化初识状态概率...')
        # 估计初始状态概率
        for tags in tqdm(tags_list):
            init_tagid = self.tag2id[tags[0]]
            self.hmm_pi[init_tagid] += 1.
        self.hmm_pi[self.hmm_pi == 0.] = 1e-10
        self.hmm_pi = self.hmm_pi / self.hmm_pi.sum()
        logger.info('完成初始状态概率构建. {:>.4f}s'.format(time.time() - start_time))

    def predict(self, sentences_list):

        logger.info('启动HMM解码预测...')
        logger.info('预测句子总数:{}'.format(len(sentences_list)))
        pred_tag_lists = []
        for sentence in tqdm(sentences_list):
            pred_tag_list = self.decoding(sentence)
            pred_tag_lists.append(pred_tag_list)
        return pred_tag_lists

    def decoding(self, word_list):
        """
        使用维特比算法对给定观测序列求状态序列， 这里就是对字组成的序列,求其对应的标注。
        维特比算法实际是用动态规划解隐马尔可夫模型预测问题，即用动态规划求概率最大路径（最优路径）
        这时一条路径对应着一个状态序列
        """
        A = torch.log(self.hmm_A)
        B = torch.log(self.hmm_B)
        Pi = torch.log(self.hmm_pi)

        # 初始化 维比特矩阵viterbi 它的维度为[状态数, 序列长度]
        seq_len = len(word_list)
        viterbi = torch.zeros(self.hmm_N, seq_len)

        # 等解码的时候，我们用backpointer进行回溯，以求出最优路径
        backpointer = torch.zeros(self.hmm_N, seq_len).long()

        start_wordid = self.word2id.get(word_list[0], None)
        Bt = B.t()
        if start_wordid is None:
            # 如果字不再字典里，则假设状态的概率分布是均匀的
            bt = torch.log(torch.ones(self.hmm_N) / self.hmm_N)
        else:
            bt = Bt[start_wordid]
        viterbi[:, 0] = Pi + bt
        backpointer[:, 0] = -1

        for step in range(1, seq_len):
            wordid = self.word2id.get(word_list[step], None)
            # 处理字不在字典中的情况
            # bt是在t时刻字为wordid时，状态的概率分布
            if wordid is None:
                # 如果字不再字典里，则假设状态的概率分布是均匀的
                bt = torch.log(torch.ones(self.hmm_N) / self.hmm_N)
            else:
                bt = Bt[wordid]  # 否则从观测概率矩阵中取bt
            for tag_id in range(len(self.tag2id)):
                max_prob, max_id = torch.max(
                    viterbi[:, step - 1] + A[:, tag_id],
                    dim=0
                )
                viterbi[tag_id, step] = max_prob + bt[tag_id]
                backpointer[tag_id, step] = max_id

        # 终止， t=seq_len 即 viterbi[:, seq_len]中的最大概率，就是最优路径的概率
        best_path_prob, best_path_pointer = torch.max(
            viterbi[:, seq_len - 1], dim=0
        )

        # 回溯，求最优路径
        best_path_pointer = best_path_pointer.item()
        best_path = [best_path_pointer]
        for back_step in range(seq_len - 1, 0, -1):
            best_path_pointer = backpointer[best_path_pointer, back_step]
            best_path_pointer = best_path_pointer.item()
            best_path.append(best_path_pointer)

        # 将tag_id组成的序列转化为tag
        assert len(best_path) == len(word_list)
        tag_list = [self.id2tag[id_] for id_ in reversed(best_path)]

        return tag_list
