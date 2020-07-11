import torch
from tensorboardX import SummaryWriter
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score
from seqeval.metrics import classification_report
from loguru import logger
import time
import copy
import os


def choice_slot_valid(slot_labels, slot_logits, slot_label_num):
    active_slot = slot_labels.view(-1) != -100
    active_logits = slot_logits.view(-1, slot_label_num)[active_slot]
    active_labels = slot_labels.view(-1)[active_slot]
    predic_slot = torch.max(active_logits.data, 1)[1].cpu()

    return (active_labels, predic_slot)


def model_train(config, model, train_iter, dev_iter=None, to_crf=False):

    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.learning_rate, weight_decay=config.weight_decay)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    total_true_labels, total_predict_labels = [], []
    tagging_map = {i: label for i, label in enumerate(config.label_list)}
    best_model = copy.deepcopy(model)
    train_loss = 0
    writer = SummaryWriter(log_dir=os.path.join(config.visual_log,
                                                time.strftime('%m-%d_%H-%M', time.localtime())))

    for epoch in range(config.num_epochs):
        logger.info('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (input_ids, labels_ids) in enumerate(train_iter):
            total_batch += 1
            model.train()

            input_tensor_ids = input_ids.clone().detach().type(torch.LongTensor).to(config.device)
            labels_tensor_ids = labels_ids.clone().detach().type(torch.LongTensor).to(config.device)
            outputs_logits, loss = model(input_tensor_ids, labels_tensor_ids)
            train_loss += loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()

            if not to_crf:
                active_labels, predict_labels = choice_slot_valid(labels_ids,
                                                               outputs_logits, config.num_label)
                total_true_labels.extend(list(active_labels.numpy()))
                total_predict_labels.extend(list(predict_labels.numpy()))
            else:
                total_true_labels.extend(list(labels_ids.view(-1).numpy()))
                total_predict_labels.extend(list(outputs_logits))

            if total_batch % config.batch_to_out == 0:
                # 每多少轮输出在训练集和验证集上的效果
                cur_true_label = [tagging_map[idx] for idx in total_true_labels]
                cur_predict_label = [tagging_map[idx] for idx in total_predict_labels]

                train_acc = accuracy_score(cur_true_label, cur_predict_label)
                train_f1 = f1_score(cur_true_label, cur_predict_label, average='macro')

                dev_acc, dev_f1, dev_loss = model_evaluate(config, model, dev_iter, mode='eval', to_crf=to_crf)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    improve = '*'
                    last_improve = total_batch
                    best_model = copy.deepcopy(model)
                else:
                    improve = ''
                time_dif = time.time() - start_time
                msg = 'Iter: {0:>6}, ' \
                      '\nTrain Loss: {1:>5.6f}, Train Acc: {2:>6.2%}, Train f1: {3:>6.2%}, ' \
                      'Val Loss: {4:>5.6f}, Val Acc: {5:>6.2%}, Val f1: {6:>6.2%}, ' \
                      'Time: {7} {8}'
                logger.info(msg.format(total_batch, train_loss/config.batch_to_out, train_acc, train_f1,
                                 dev_loss, dev_acc, dev_f1, time_dif, improve))

                writer.add_scalar("loss/train", train_loss/config.batch_to_out, total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                writer.add_scalar("f1score/train", train_f1, total_batch)
                writer.add_scalar("f1score/dev", dev_f1, total_batch)

                total_true_labels, total_predict_labels = [], []
                train_loss = 0

            if config.early_stop and total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                logger.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    return best_model


def model_evaluate(config, model, data_iter, mode='eval', to_crf=False):
    model.eval()
    loss_total = 0
    tagging_map = {i: label for i, label in enumerate(config.label_list)}
    total_true_labels, total_predict_labels = [], []

    with torch.no_grad():
        for input_ids, labels_ids in data_iter:

            input_tensor_ids = input_ids.clone().detach().type(torch.LongTensor).to(config.device)
            labels_tensor_ids = labels_ids.clone().detach().type(torch.LongTensor).to(config.device)
            outputs_logits, loss = model(input_tensor_ids, labels_tensor_ids)
            loss_total += loss.item()

            if not to_crf:
                active_labels, predict_labels = choice_slot_valid(labels_ids,
                                                                  outputs_logits, config.num_label)
                total_true_labels.extend(list(active_labels.numpy()))
                total_predict_labels.extend(list(predict_labels.numpy()))
            else:
                total_true_labels.extend(list(labels_ids.view(-1).numpy()))
                total_predict_labels.extend(list(outputs_logits))

    # 每多少轮输出在训练集和验证集上的效果
    cur_true_label = [tagging_map[idx] for idx in total_true_labels]
    cur_predict_label = [tagging_map[idx] for idx in total_predict_labels]

    acc = accuracy_score(cur_true_label, cur_predict_label)
    f1score = f1_score(cur_true_label, cur_predict_label, average='macro')

    if mode == 'test':
        report = classification_report(cur_true_label, cur_predict_label, digits=4)
        return acc, f1score, loss_total / len(data_iter), report
    else:
        return acc, f1score, loss_total / len(data_iter)


def model_test(config, model, test_iter, to_crf=False):
    start_time = time.time()
    test_acc, f1score, test_loss, test_report = model_evaluate(config, model, test_iter,
                                                               mode='test', to_crf=to_crf)
    msg = '\nTest Loss: {0:>5.4}, Test Acc: {1:>6.2%}, Test f1: {2:>6.2%}'
    logger.info(msg.format(test_loss, test_acc, f1score))
    logger.info("\nPrecision, Recall and F1-Score...")
    logger.info("\n{}".format(test_report))
    time_dif = time.time() - start_time
    logger.info("Time usage:{0:>.6}s".format(time_dif))


def model_metrics(true_labels, pre_labels):
    start_time = time.time()
    acc = accuracy_score(true_labels, pre_labels)
    f1score = f1_score(true_labels, pre_labels, average='macro')
    report = classification_report(true_labels, pre_labels, digits=4)
    msg = '\nTest Acc: {0:>6.2%}, Test f1: {1:>6.2%}'
    logger.info(msg.format(acc, f1score))
    logger.info("\nPrecision, Recall and F1-Score...")
    logger.info("\n{}".format(report))
    time_dif = time.time() - start_time
    logger.info("Time usage:{0:>.6}s".format(time_dif))
