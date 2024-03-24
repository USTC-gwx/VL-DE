import os
import time
import shutil
import json

import torch
import numpy
from transformers import BertTokenizer

import data
import opts
from vocab import Vocabulary, deserialize_vocab
from model import SGRAF
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_attn_scores

import logging
import tensorboard_logger as tb_logger


def logging_func(logfile, message):
    with open(logfile, "a") as f:
        f.write(message)
    f.close()

def main():
    opt = opts.parse_opt()
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # Load Vocabulary Wrapper
    # vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    # opt.vocab_size = len(vocab)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.vocab
    opt.vocab_size = len(vocab)
    opt.dep_path = './dep/'

    feature_vocab = None
    with open(opt.dep_path + 'feature2id.json', 'r') as f:
        feature_vocab = json.load(f)
    opt.feature_size = len(feature_vocab)

    # Load data loaders
    train_loader, val_loader = data.get_loaders(opt.data_name, vocab, tokenizer, opt.batch_size, opt.workers, opt)

    # Construct the model
    model = SGRAF(opt)

    # if not model.is_data_parallel:
    #     model.make_data_parallel()

    # Train the Model
    best_rsum = 0

    for epoch in range(opt.num_epochs):

        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)

        # evaluate on validation set
        with torch.no_grad():
            rsum, r1, r5, r10, r1i, r5i, r10i = validate(opt, val_loader, model, best_rsum)

        message = "Epoch:%d:Image to text: (%.1f,%.1f,%.1f)"%(epoch, r1, r5, r10)
        message += "Text to image: (%.1f, %.1f, %.1f)" % (r1i,r5i, r10i)
        message += "rsum: %.1f\n" % rsum

        log_file = os.path.join(opt.logger_name, "performance.txt")
        logging_func(log_file, message)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)

        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_name + '/')

        torch.cuda.empty_cache()

def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))
                    
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)


def validate(opt, val_loader, model, best_rsum):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, cap_lens = encode_data(model, val_loader, opt.log_step, logging.info)

    # clear duplicate 5*images and keep 1*images
    img_embs = numpy.array([img_embs[i] for i in range(0, len(img_embs), 5)])

    # record computation time of validation
    start = time.time()
    sims = shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, shard_size=100)
    end = time.time()
    print("calculate similarity time:", end-start)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr))

    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, cap_lens, sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri, meanr))

    # sum of recalls to be used for early stopping
    rsum = r1 + r5 + r10 + r1i + r5i + r10i
    if rsum > best_rsum:
        numpy.save('/output/sims.npy', sims)

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('rsum', rsum, step=model.Eiters)

    return rsum, r1, r5, r10, r1i, r5i, r10i


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + 'model_best.pth.tar')
            # if is_best:
            #     shutil.copyfile(prefix + 'model_best.pth.tar', prefix + filename)
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    """
    Sets the learning rate to the initial LR
    decayed by 10 after opt.lr_update epoch
    """
    if epoch >= opt.lr_update:
        a = 1
    else:
        a = 0
    lr = opt.learning_rate * (0.1 ** a)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    
    os.environ['CUDA_VISIBLE_DEVICES'] = "6"
    main()
