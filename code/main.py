from config import parse_args
FLAGS = parse_args(True)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.visible_device

# import torch
# torch.multiprocessing.set_sharing_strategy('file_system')

# from model import ModelMultiLabel
from model_nodistill import ModelMultiLabel
# from model_bert import ModelMultiLabel
# from model_bert_nodistill import ModelMultiLabel

from trainer import Trainer
# from trainer_bert import Trainer
# from trainer_bert_lab import Trainer

from dataset import ICDDataset
# from dataset_bert import ICDDataset

from vocab import WordVocab
from utils import LabelUtil, seed_torch

seed_torch(FLAGS.random_seed)


def main():
    vocab = WordVocab.load_vocab(FLAGS.vocab_path)
    print('vocab', len(vocab))

    label_util = LabelUtil(file_paths=[FLAGS.data_path_train, FLAGS.data_path_test, FLAGS.data_path_eval],
                           label_description_path=FLAGS.label_descriptions_path)

    dataset_train = ICDDataset(FLAGS, vocab, label_util, FLAGS.data_path_train, False)
    dataset_eval = ICDDataset(FLAGS, vocab, label_util, FLAGS.data_path_eval, True)
    dataset_test = ICDDataset(FLAGS, vocab, label_util, FLAGS.data_path_test, True)


    model = ModelMultiLabel(FLAGS, vocab, label_util).to(0)

    trainer = Trainer(FLAGS, dataset_train, dataset_eval, dataset_test, model)


    trainer.test(0, eval_or_test='eval')
    trainer.save_model(FLAGS.model_name + '.ep{:d}.pkl'.format(1))
    try:
        for epoch in range(1, FLAGS.n_epochs + 1):
            trainer.train(epoch, train_type=FLAGS.train_type, teacher_data_noise_length=FLAGS.teacher_data_noise_length)
            trainer.test(epoch, eval_or_test='eval')
            trainer.save_model(FLAGS.model_name + '.ep{:d}.pkl'.format(epoch))
            trainer.save_checkpoint(FLAGS.model_name + '.gpu{:s}.last.checkpoint.pkl'.format(FLAGS.visible_device))

    except KeyboardInterrupt:
        print('KeyboardInterrupt')
        trainer.save_checkpoint(FLAGS.model_name + '.gpu{:s}.last.checkpoint.pkl'.format(FLAGS.visible_device))
        print('last checkpoint saved')

main()


