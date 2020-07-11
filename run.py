import argparse
from loguru import logger
from torch.utils.data import DataLoader
from importlib import import_module

from utils.utils import random_seed, set_logger, \
    config_to_json_string
from utils.data_processor import DataProcessor, \
    convert_examples_to_features, BuildDataSet, \
    convert_examples_to_features_crf, HMMDataProcessor
from utils.train_eval import model_train, model_test, model_metrics


def bilstm_train_eval(config,
                      import_model,
                      train_examples,
                      dev_examples=None,
                      test_examples=None
                      ):
    processor = DataProcessor(config.data_dir, config.do_lower_case)
    word2id = processor.get_vocab()
    config.vocab_size = len(word2id)
    train_features = convert_examples_to_features(examples=train_examples,
                                                  word2id=word2id,
                                                  label_list=config.label_list,
                                                  max_seq_length=config.max_seq_length
                                                  )
    train_dataset = BuildDataSet(train_features)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    if dev_examples:
        dev_features = convert_examples_to_features(examples=dev_examples,
                                                    word2id=word2id,
                                                    label_list=config.label_list,
                                                    max_seq_length=config.max_seq_length
                                                    )
        dev_dataset = BuildDataSet(dev_features)
        dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True)
    else:
        dev_loader = None

    if test_examples:
        test_features = convert_examples_to_features(examples=test_examples,
                                                     word2id=word2id,
                                                     label_list=config.label_list,
                                                     max_seq_length=config.max_seq_length
                                                     )
        test_dataset = BuildDataSet(test_features)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    else:
        test_loader = None

    logger.info("self config:\n {}".format(config_to_json_string(config)))

    model = import_model.Model(config).to(config.device)
    best_model = model_train(config, model, train_loader, dev_loader)
    model_test(config, best_model, test_loader)


def bilstm_crf_train_eval(config,
                          import_model,
                          train_examples,
                          dev_examples=None,
                          test_examples=None
                          ):
    processor = DataProcessor(config.data_dir, config.do_lower_case)
    word2id = processor.get_vocab()
    config.vocab_size = len(word2id)
    train_features = convert_examples_to_features_crf(examples=train_examples,
                                                      word2id=word2id,
                                                      label_list=config.label_list,
                                                      )
    train_dataset = BuildDataSet(train_features)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    if dev_examples:
        dev_features = convert_examples_to_features_crf(examples=dev_examples,
                                                        word2id=word2id,
                                                        label_list=config.label_list,
                                                        )
        dev_dataset = BuildDataSet(dev_features)
        dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True)
    else:
        dev_loader = None

    if test_examples:
        test_features = convert_examples_to_features_crf(examples=test_examples,
                                                         word2id=word2id,
                                                         label_list=config.label_list,
                                                         )
        test_dataset = BuildDataSet(test_features)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    else:
        test_loader = None

    logger.info("self config:\n {}".format(config_to_json_string(config)))

    model = import_model.Model(config).to(config.device)
    best_model = model_train(config, model, train_loader, dev_loader, to_crf=True)
    model_test(config, best_model, test_loader, to_crf=True)


def hmm_train_eval(config,
                   import_model,
                   train_examples,
                   dev_examples=None,
                   test_examples=None
                   ):
    train_sentences = train_examples[0]
    train_tags = train_examples[1]
    if dev_examples:
        train_sentences.extend(dev_examples[0])
        train_tags.extend(dev_examples[1])

    model = import_model.Model(config.num_label)
    model.train(train_sentences, train_tags)

    if test_examples:
        test_sentences = test_examples[0]
        test_tags = test_examples[1]
        predict_labels = model.predict(test_sentences)
        model_metrics(test_tags, predict_labels)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Chinese NER Task')
    parser.add_argument('--model', type=str, required=True,
                        help='choose a model: bilstm, bilstm_crf, hmm')
    args = parser.parse_args()

    model_name = args.model
    import_model = import_module('model.' + model_name)
    config = import_model.Config()
    random_seed(config.seed)
    set_logger(config.logging_dir)

    # load data
    if args.model == 'hmm':
        processor = HMMDataProcessor(config.data_dir, config.do_lower_case)
    else:
        processor = DataProcessor(config.data_dir, config.do_lower_case)

    train_examples = processor.get_train_examples()
    config.train_num_examples = len(train_examples)
    dev_examples = processor.get_dev_examples()
    config.dev_num_examples = len(dev_examples)
    test_examples = processor.get_test_examples()
    config.test_num_examples = len(test_examples)
    config.label_list = processor.get_tagging()
    config.num_label = len(config.label_list)

    if args.model == 'bilstm':
        bilstm_train_eval(config,
                          import_model=import_model,
                          train_examples=train_examples,
                          dev_examples=dev_examples,
                          test_examples=test_examples
                          )
    elif args.model == 'bilstm_crf':
        bilstm_crf_train_eval(config,
                              import_model=import_model,
                              train_examples=train_examples,
                              dev_examples=dev_examples,
                              test_examples=test_examples
                              )
    elif args.model == 'hmm':
        hmm_train_eval(config,
                       import_model=import_model,
                       train_examples=train_examples,
                       dev_examples=dev_examples,
                       test_examples=test_examples)




