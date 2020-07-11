import os
import json
import copy
import numpy as np
from loguru import logger
import torch.utils.data as Data


class InputExample(object):
    """
    A single training/test example.

    Args:
        guid: Unique id for the example.
        text: list string. The untokenized text of the first sequence.
        label: string. tagging label.
    """

    def __init__(self, guid, text, label):
        self.guid = guid
        self.text = text
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        label_ids: Label corresponding to the input
    """

    def __init__(self, input_ids, label_ids):
        self.input_ids = input_ids
        self.label_ids = label_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor:

    def __init__(self,
                 data_dir,
                 do_lower_case=True):
        self.data_dir = data_dir
        self.do_lower_case = do_lower_case

    def get_train_examples(self):
        return self._create_examples(self._read_data(os.path.join(self.data_dir, "train.txt")), 'train')

    def get_dev_examples(self):
        return self._create_examples(self._read_data(os.path.join(self.data_dir, "dev.txt")), 'dev')

    def get_test_examples(self):
        return self._create_examples(self._read_data(os.path.join(self.data_dir, "test.txt")), 'test')

    def get_tagging(self):
        return self._read_dictionary(os.path.join(self.data_dir, 'tagging.txt'))

    def get_vocab(self):
        vocab = self._read_dictionary(os.path.join(self.data_dir, 'vocab.txt'))
        word2id = {word: idx for idx, word in enumerate(vocab)}
        return word2id

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i+1)
            text = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    def _read_data(self, input_file):
        """
        :param input_file:
        :return: list [text, category, intention, slot_label]
        """
        data_list = []
        with open(input_file, 'r', encoding='utf-8') as f:
            text_list = []
            tag_list = []
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    text, tag = line.split()
                    if self.do_lower_case:
                        text = str(text).lower()
                    text_list.append(text)
                    tag_list.append(tag)
                else:
                    data_list.append([text_list, tag_list])
                    text_list = []
                    tag_list = []

        return data_list

    def _read_dictionary(self, input_file):
        dict_list = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) > 0:
                    dict_list.append(line)
        return dict_list


def convert_examples_to_features(
    examples,
    word2id,
    label_list,
    max_seq_length=64,
    unk_token='<UNK>',
    pad_token='<PAD>',
    pad_token_label_id=-100,
):

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (index, example) in enumerate(examples):
        input_ids = []
        label_ids = []
        assert len(example.text) == len(example.label), \
            'example text lens not equal slot label lens.'
        for word, label in zip(example.text, example.label):
            word_tokens = word2id.get(word, word2id[unk_token])
            input_ids.append(word_tokens)
            label_ids.append(label_map[label])

        input_ids = input_ids[:max_seq_length]
        label_ids = label_ids[:max_seq_length]

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)

        input_ids += [word2id[pad_token]] * padding_length
        label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if index < 3:
            logger.debug("*** Example ***")
            logger.debug("guid: %s" % (example.guid))
            logger.debug("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.debug("label_ids: %s" % " ".join([str(x) for x in label_ids]))

        features.append(InputFeatures(input_ids, label_ids))

    return features


def convert_examples_to_features_crf(
    examples,
    word2id,
    label_list,
    unk_token='<UNK>'
):

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (index, example) in enumerate(examples):
        input_ids = []
        label_ids = []
        assert len(example.text) == len(example.label), \
            'example text lens not equal slot label lens.'
        for word, label in zip(example.text, example.label):
            word_tokens = word2id.get(word, word2id[unk_token])
            input_ids.append(word_tokens)
            label_ids.append(label_map[label])

        if index < 3:
            logger.debug("*** Example ***")
            logger.debug("guid: %s" % (example.guid))
            logger.debug("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.debug("label_ids: %s" % " ".join([str(x) for x in label_ids]))

        features.append(InputFeatures(input_ids, label_ids))

    return features


class BuildDataSet(Data.Dataset):
    """
    将经过convert_examples_to_features的数据 包装成 Dataset
    """
    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        feature = self.features[index]
        input_ids = np.array(feature.input_ids)
        label_ids = np.array(feature.label_ids)

        return input_ids, label_ids

    def __len__(self):
        return len(self.features)


class HMMDataProcessor:

    def __init__(self,
                 data_dir,
                 do_lower_case=True):
        self.data_dir = data_dir
        self.do_lower_case = do_lower_case

    def get_train_examples(self):
        return self._read_data(os.path.join(self.data_dir, "train.txt"))

    def get_dev_examples(self):
        return self._read_data(os.path.join(self.data_dir, "dev.txt"))

    def get_test_examples(self):
        return self._read_data(os.path.join(self.data_dir, "test.txt"))

    def get_tagging(self):
        return self._read_dictionary(os.path.join(self.data_dir, 'tagging.txt'))

    def _read_data(self, input_file):
        """
        :param input_file:
        :return: list [text, category, intention, slot_label]
        """
        sentences_list, tags_list = [], []
        with open(input_file, 'r', encoding='utf-8') as f:
            text_list = []
            tag_list = []
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    text, tag = line.split()
                    if self.do_lower_case:
                        text = str(text).lower()
                    text_list.append(text)
                    tag_list.append(tag)
                else:
                    sentences_list.append(text_list)
                    tags_list.append(tag_list)
                    text_list = []
                    tag_list = []

        return (sentences_list, tags_list)

    def _read_dictionary(self, input_file):
        dict_list = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) > 0:
                    dict_list.append(line)
        return dict_list


if __name__ == '__main__':
    processor = DataProcessor('../data')
    train_examples = processor._read_data(os.path.join(processor.data_dir, "train.txt"))
    print(train_examples[:5])
