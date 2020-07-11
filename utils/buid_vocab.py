import os


class BuildVocab:
    def __init__(self, path):
        self.path = path
        self.tag_dict = dict()
        self.tag_dict['O'] = len(self.tag_dict)

    def read_data(self, file_name):
        sentence_list = []
        sentence_tag_list = []
        with open(file_name, 'r', encoding='utf-8') as f:
            word_list = []
            word_tag_list = []
            for line in f.readlines():
                line = line.strip()
                if len(line) != 0:
                    word, tag = line.split(' ')
                    word_list.append(word)
                    word_tag_list.append(tag)
                else:
                    sentence_list.append(word_list)
                    sentence_tag_list.append(word_tag_list)
                    word_list = []
                    word_tag_list = []
        return sentence_list, sentence_tag_list

    def add_tag(self, tag_list):
        for sentence in tag_list:
            for tag in sentence:
                if tag not in self.tag_dict:
                    self.tag_dict[tag] = len(self.tag_dict)

    def write_vocab(self, file_name, vocab):
        with open(file_name, 'w', encoding='utf-8') as fw:
            for word in vocab.keys():
                fw.write(word+'\n')

    def run(self, files_list):

        for file in files_list:
            file_name = os.path.join(self.path, file)
            sentence_list, tag_list = self.read_data(file_name)
            print("{} : {}".format(file, len(sentence_list)))
            self.add_tag(tag_list)
        self.write_vocab(os.path.join(self.path, 'tagging.txt'), self.tag_dict)


if __name__ == '__main__':
    build_vocab = BuildVocab('../data')
    build_vocab.run(['train.txt', 'dev.txt', 'test.txt'])
