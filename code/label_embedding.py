from config import parse_args
# from config_bert import parse_args
FLAGS = parse_args(True)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.visible_device

import time
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModel
from utils import LabelUtil

import tqdm
import vocab
from vocab import WordVocab



class LabelEmbedding():
    def __init__(self, emb_dim, version):
        print('label_embedding', emb_dim)
        self.emb_dim = emb_dim
        # label_emb_pkl = 'model_save/label_emb_mimic_iii.{:d}.pkl'.format(emb_dim)
        label_emb_pkl = 'model_save/label_emb.{:s}.{:d}.pkl'.format(version, emb_dim)
        class_emb_pkl = 'model_save/class_emb.{:s}.{:d}.pkl'.format(version, emb_dim)

        self.vocab = WordVocab.load_vocab(FLAGS.vocab_path)
        self.label_util = LabelUtil(file_paths=[FLAGS.data_path_train, FLAGS.data_path_eval, FLAGS.data_path_test],
                                    label_description_path=FLAGS.label_descriptions_path)

        self.token_emb = nn.Embedding(len(self.vocab), emb_dim)
        try:
            self.token_emb.weight = torch.load('model_save/token_emb.full.skipgram.0907.{:d}.pkl'.format(emb_dim))
        except:
            self.token_emb.weight = torch.load('model_save/token_emb.full.skipgram.{:d}.pkl'.format(emb_dim))

        if self.emb_dim == 100:
            in_count, out_count = 0, 0
            with open('data_raw/processed_full_mimic2.embed', 'r') as f:
                for line in f:
                    list_data = line[:-1].split(' ')
                    word = list_data[0]
                    vec = np.array(list_data[1:]).astype('float32')
                    if word in self.vocab.stoi:
                        idx = self.vocab.stoi[word]
                        self.token_emb.weight.data[idx] = torch.tensor(vec)
                        in_count += 1
                    else:
                        out_count += 1
            print('pengfei embedding', in_count, out_count)

        self.label_emb = nn.Embedding(self.label_util.label_size, emb_dim)
        self.class_emb = nn.Embedding(self.label_util.class_size, emb_dim)
        with torch.no_grad():
            self.get_vector()
        torch.save(self.label_emb.weight, label_emb_pkl) # self.final.weight = torch.load('emb.pkl')
        torch.save(self.class_emb.weight, class_emb_pkl) # self.final.weight = torch.load('emb.pkl')

        print('label_emb_pkl save done')
        print('class_emb_pkl save done')


    def raw_text2tokens(self, text):
        list_word = text.split()
        tokens = list_word.copy()

        for i, token in enumerate(tokens):
            tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

        return tokens


    def get_vector(self):
        idx_iter = tqdm.tqdm(range(self.label_util.label_size), desc="get_vector" , total=self.label_util.label_size, bar_format="{l_bar}{r_bar}")

        # for idx, text in self.label_util.index2text.items():
        for idx, text in tqdm.tqdm(self.label_util.label_index2name.items(), total=self.label_util.label_size):
            tokens = self.raw_text2tokens(text)
            # if tokens == []:
            #     tokens = [0]
            #     print(idx, self.label_util.index2code[idx])
            tokens = torch.tensor(tokens).unsqueeze(0).to(0)

            # print('tokens', tokens.size(), tokens)
            hidden_state = self.token_emb(tokens).view(-1, self.emb_dim).mean(dim=0)
            # print('hidden_state', hidden_state.size())
            self.label_emb.weight[idx] = hidden_state

    def get_class_emb_weight(self):
        for class_idx, label_idxs, in self.label_util.class_index2label_indexs.items():
            list_vec = [self.label_emb.weight[label_idx].unsqueeze(0) for label_idx in label_idxs]
            vec = torch.cat(list_vec, dim=0).mean(dim=0)
            self.class_emb.weight[class_idx] = vec


class BertLabelEmbedding():
    def __init__(self, bert_file, bert_file_name, version):

        self.tokenizer = AutoTokenizer.from_pretrained(bert_file)
        self.bert = AutoModel.from_pretrained(bert_file).to(0)

        self.emb_dim = self.bert.embeddings.word_embeddings.weight.size(1)
        # label_emb_pkl = 'model_save/label_emb_mimic_iii.{:d}.pkl'.format(emb_dim)

        self.label_emb_pkl = 'model_save/label_emb.{:s}.{:s}.pkl'.format(bert_file_name, version)
        self.class_emb_pkl = 'model_save/class_emb.{:s}.{:s}.pkl'.format(bert_file_name, version)

        self.label_util = LabelUtil(file_paths=[FLAGS.data_path_train, FLAGS.data_path_eval, FLAGS.data_path_test],
                                    label_description_path=FLAGS.label_descriptions_path)

        self.bert.eval()
        self.label_emb = nn.Embedding(self.label_util.label_size, self.emb_dim).to(0)
        self.class_emb = nn.Embedding(self.label_util.class_size, self.emb_dim).to(0)

        with torch.no_grad():
            self.get_vector()

        torch.save(self.label_emb.weight, self.label_emb_pkl)  # self.final.weight = torch.load('emb.pkl')
        print('label_emb_pkl saved')

        torch.save(self.class_emb.weight, self.class_emb_pkl)  # self.final.weight = torch.load('emb.pkl')
        print('class_emb_pkl saved')

    def raw_text2tokens(self, text):
        inp = self.tokenizer(text)['input_ids']
        return inp

    def get_vector(self):
        # for idx, text in tqdm.tqdm(self.label_util.index2text.items(), total=self.label_util.label_size):
        for idx, text in tqdm.tqdm(self.label_util.label_index2name.items(), total=self.label_util.label_size):

            tokens = self.raw_text2tokens(text)
            tokens = torch.tensor(tokens).unsqueeze(0).to(0)
            with torch.no_grad():
                # hidden_state = self.bert_model(tokens)[0][:, 1:-1].view(-1, self.emb_dim).mean(dim=0)
                hidden_state = self.bert(tokens)[0].view(-1, self.emb_dim).mean(dim=0)

            self.label_emb.weight[idx] = hidden_state

    def get_class_emb_weight(self):
        for class_idx, label_idxs, in self.label_util.class_index2label_indexs.items():
            list_vec = [self.label_emb.weight[label_idx].unsqueeze(0) for label_idx in label_idxs]
            vec = torch.cat(list_vec, dim=0).mean(dim=0)
            self.class_emb.weight[class_idx] = vec

emb = LabelEmbedding(emb_dim=256, version=FLAGS.version)
# emb = BertLabelEmbedding(bert_file='/data/tongzhou/transformers_models/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12/',
#                          bert_file_name='bluebert', version=FLAGS.version)


