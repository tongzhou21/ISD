import numpy as np
import torch

import os
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup as WarmupLinearSchedule

# from evaluate import Evaluate
import evaluation
import time
import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast, GradScaler


class Trainer:
    def __init__(self, FLAGS, dataset_train, dataset_eval, dataset_test, model):
        print('TRAINER: ' + os.path.basename(__file__))
        self.FLAGS = FLAGS

        self.dataset_train, self.dataset_eval, self.dataset_test = dataset_train, dataset_eval, dataset_test

        self.model = model.to(0)

        self.real_batch_size = FLAGS.real_batch_size

        # # bert
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in list(self.model.named_parameters()) if ('bert' not in n)],
        #      'weight_decay': FLAGS.weight_decay,
        #      'lr': FLAGS.lr}, # 非bert参数
        #     {'params': [p for n, p in list(self.model.named_parameters()) if ('bert' in n and not any (nd in n for nd in no_decay))],
        #      'weight_decay': FLAGS.weight_decay_bert,
        #      'lr': FLAGS.lr_bert}, # bert中可以weight_decay的部分
        #     {'params': [p for n, p in list(self.model.named_parameters()) if ('bert' in n and any(nd in n for nd in no_decay))],
        #      'weight_decay': 0.0,
        #      'lr': FLAGS.lr_bert},  # bert中不weight_decay的部分
        # ]
        # if self.FLAGS.freeze_bert:
        #     optimizer_grouped_parameters = [
        #         {'params': [p for n, p in list(self.model.named_parameters()) if ('bert' not in n)],
        #          'weight_decay': FLAGS.weight_decay,
        #          'lr': FLAGS.lr},  # 非bert参数
        #     ]
        #     print('***** FREEZE BERT PARAMETERS *****')
        #
        # self.optimizer = AdamW(optimizer_grouped_parameters)
        self.optimizer = AdamW(self.model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

        self.scheduler = WarmupLinearSchedule(self.optimizer,
                                              num_warmup_steps=self.dataset_train.__len__() / FLAGS.real_batch_size * 5,
                                              num_training_steps=self.dataset_train.__len__() / FLAGS.real_batch_size * FLAGS.n_epochs)
        if self.FLAGS.apex:
            # self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
            self.scaler = GradScaler()


        if FLAGS.last_checkpoint is not None:
            self.model, self.optimizer, self.scheduler = self.load_checkpoint(FLAGS.last_checkpoint, self.model, self.optimizer, self.scheduler)
            print('existed checkpoint:', FLAGS.last_checkpoint)
        if FLAGS.model_pkl_pretrain is not None:
            self.model = self.load_model(FLAGS.model_pkl_pretrain, self.model)
            print('existed pretrained model:', FLAGS.model_pkl_pretrain)


    def train(self, epoch, train_type='student+teacher+distill', teacher_data_noise_length=1):
        print('--' * 6, epoch, '--' * 6)

        print('train_type, teacher_data_noise_length', train_type, teacher_data_noise_length)
        self.dataset_train.teacher_data_noise_length=teacher_data_noise_length
        dataloader = DataLoader(self.dataset_train, batch_size=self.FLAGS.batch_size,
                                num_workers=self.FLAGS.num_workers, shuffle=False)

        data_iter = tqdm.tqdm(enumerate(dataloader), desc="%s: ep%d" % ('train', epoch),
                              total=len(dataloader), bar_format="{l_bar}{r_bar}")

        self.model.train()
        self.model.zero_grad()

        accumulation_steps = self.real_batch_size // self.FLAGS.batch_size
        dict_indicator = {'batch_count': 0}
        for idx_batch, data in data_iter:
            self.optimizer.zero_grad()
            data, origin_text = data
            # print('origin_text', origin_text[0][:20])
            data = {key: value.to(0) for key, value in data.items()}

            if self.FLAGS.apex:
                with autocast():
                    _, loss, indicator = self.model(data, train_type)
            else:
                _, loss, indicator = self.model(data, train_type)

            for key, value in indicator.items():
                if key not in dict_indicator:
                    dict_indicator[key] = 0
                    dict_indicator[key + '_count'] = 0
                dict_indicator[key] += value
                dict_indicator[key + '_count'] += 1.0

            if self.FLAGS.apex:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)

            if (idx_batch + 1) % accumulation_steps == 0:
                if self.FLAGS.apex:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                else:
                    self.optimizer.step()
                    self.scheduler.step()


        logs_train = 'train '
        for key, value in dict_indicator.items():
            if '_count' not in key:
                logs_train += '{:s}:{:.4f}, '.format(key,
                                                     dict_indicator[key] / dict_indicator[key + '_count'])
        print(time.strftime("%m%d%H%M", time.localtime()), end=' ')
        print(logs_train)



    def test(self, epoch, train_type='student+teacher+distill', eval_or_test='eval'):
        dataset_test = self.dataset_eval if eval_or_test == 'eval' else self.dataset_test
        dataloader = DataLoader(dataset_test, batch_size=self.FLAGS.batch_size,
                                num_workers=self.FLAGS.num_workers, shuffle=False)

        data_iter = tqdm.tqdm(enumerate(dataloader), desc="%s: ep%d" % (eval_or_test, epoch),
                              total=len(dataloader), bar_format="{l_bar}{r_bar}")

        self.model.eval()
        list_y, list_yhat, list_yhat_raw = [], [], []
        dict_indicator = {'batch_count': 0}
        with torch.no_grad():
            for idx_batch, data in data_iter:
                data, origin_text = data
                data = {key: value.to(0) for key, value in data.items()}

                prob, loss, indicator = self.model(data, train_type=train_type)
                # output =  torch.sigmoid(y_hat)
                output = prob.data.cpu().numpy()
                list_yhat_raw.append(output)
                output = np.round(output)

                list_y.append(data['y_mix'].data.cpu().numpy()) if 'y_mix' in data \
                    else list_y.append(data['y'].data.cpu().numpy())

                list_yhat.append(output)

                for key, value in indicator.items():
                    if key not in dict_indicator:
                        dict_indicator[key] = 0
                        dict_indicator[key + '_count'] = 0
                    dict_indicator[key] += value
                    dict_indicator[key + '_count'] += 1.0

            list_y = np.concatenate(list_y, axis=0)
            list_yhat = np.concatenate(list_yhat, axis=0)
            list_yhat_raw = np.concatenate(list_yhat_raw, axis=0)

            metrics = evaluation.all_metrics(list_yhat, list_y, k=8, yhat_raw=list_yhat_raw)
            # print('metrics', metrics)
            evaluation.print_metrics(metrics)
            # if metrics['f1_micro'] > max(self.list_indicator_for_model_save) and epoch != 0:
            #     self.save_model(self.FLAGS.model_name + '.best.pkl')
            #     print('save model pkl')
            self.save_model(self.FLAGS.model_name + '.latest.pkl')
            # self.list_indicator_for_model_save.append(metrics['f1_micro'])

        logs_train = eval_or_test + ' '
        for key, value in dict_indicator.items():
            if '_count' not in key:
                logs_train += '{:s}:{:.4f}, '.format(key, dict_indicator[key] / dict_indicator[key + '_count'])
        print(time.strftime("%m%d%H%M", time.localtime()), end=' ')
        print(logs_train)


    def save_model(self, model_pkl):
        torch.save({"model_state_dict": self.model.state_dict()}, model_pkl)
        return True

    def load_model(self, model_pkl, model):
        model.load_state_dict(torch.load(model_pkl)['model_state_dict'])
        return model

    def save_checkpoint(self, checkpoint):
        torch.save({"model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    }, checkpoint)
        return True

    def load_checkpoint(self, checkpoint, model, optimizer, scheduler):
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return model, optimizer, scheduler






