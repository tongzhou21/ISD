import torch
import gensim
import numpy as np


import http.client
import hashlib
import urllib
import random
import json
import time
# import os
import os

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# seed_torch()

dict_translation_map = {} # map
def back_translation(text, language='en'):
    if text in dict_translation_map:
        return dict_translation_map[text]
    query_ch = text
    list_en = translation(query=query_ch, src='zh', tgt=language)
    if len(list_en) == 0:
        return text
    time.sleep(1)
    query_en = list_en[0]
    list_ch = translation(query=query_en, src=language, tgt='zh')
    if len(list_ch) == 0:
        return text
    dict_translation_map[text] = list_ch[0]
    return list_ch[0]

def label_translation(list_data, src='en', tgt='zh'):
    query_ch = '\n'.join(list_data)
    list_en = translation(query=query_ch, src=src, tgt=tgt)
    time.sleep(1.01)
    return list_en


def back_translation_list(list_data, language='en'):
    query_ch = '\n'.join(list_data)
    list_en = translation(query=query_ch, src='zh', tgt=language)
    time.sleep(1)
    query_en = '\n'.join(list_en)
    list_ch = translation(query=query_en, src=language, tgt='zh')
    return list_ch


def translation(query, src='zh', tgt='en'):
    appid = '' # TODO: delete
    secretKey = ''

    httpClient = None
    myurl = '/api/trans/vip/translate'

    fromLang = src
    toLang = tgt
    salt = random.randint(32768, 65536)

    q = query

    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
        q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
        salt) + '&sign=' + sign

    list_tgt = []
    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)

        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)

        list_tgt = [pair['dst'] for pair in result['trans_result']]

    except Exception as e:
        pass
        # print(e)
    finally:
        if httpClient:
            httpClient.close()
    return list_tgt



def gensim_model2vec_matrix(vocab, emb_dim=100):
    gensim_model_path = 'model_save/gensim.w2v.{:d}.model'.format(emb_dim)
    np_matrix = np.random.rand(len(vocab), 100)
    model = gensim.models.Word2Vec.load(gensim_model_path)
    error_count = 0
    for idx, token in enumerate(vocab.itos):
        try:
            np_matrix[idx] = model.wv.get_vector(token)
        except:
            error_count += 1
    print('error_count', error_count, 'vocab_size', len(vocab))
    return torch.tensor(np_matrix).float()



def load_full_codes(data_path):
    index2code = {}
    code2index = {}

    with open(data_path, 'r') as f:
        for line in f:
            labels = line.replace('\n', '').replace('\r', '').strip().split(',')[-2]
            for code in labels.split(';'):
                if code not in code2index:
                    code2index[code] = len(code2index)
                    index2code[len(index2code)] = code

    with open(data_path.replace('train', 'test'), 'r') as f:
        for line in f:
            labels = line.replace('\n', '').replace('\r', '').strip().split(',')[-2]
            for code in labels.split(';'):
                if code not in code2index:
                    code2index[code] = len(code2index)
                    index2code[len(index2code)] = code
    print('full_code', len(code2index), len(index2code))
    return code2index, index2code



import re
def sort_key(s):
    #sort_strings_with_embedded_numbers
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)
    pieces[1::2] = map(int, pieces[1::2])
    return pieces

import re
def refine_text(raw_text):
    term_pattern = re.compile('[A-Za-z]+')
    raw_dsum = raw_text
    raw_dsum = re.sub(r'\[[^\]]+\]', ' ', raw_dsum)
    raw_dsum = re.sub(r'admission date', ' ', raw_dsum, flags=re.I)
    raw_dsum = re.sub(r'discharge date', ' ', raw_dsum, flags=re.I)
    raw_dsum = re.sub(r'date of birth', ' ', raw_dsum, flags=re.I)
    raw_dsum = re.sub(r'sex', ' ', raw_dsum, flags=re.I)
    raw_dsum = re.sub(r'service', ' ', raw_dsum, flags=re.I)
    raw_dsum = re.sub(r'dictated by .*$', ' ', raw_dsum, flags=re.I)
    raw_dsum = re.sub(r'completed by .*$', ' ', raw_dsum, flags=re.I)
    raw_dsum = re.sub(r'signed electronically by .*$', ' ', raw_dsum, flags=re.I)
    tokens = [token.lower() for token in re.findall(term_pattern, raw_dsum)]
    tokens = [token for token in tokens if len(token) > 1]

    text = raw_dsum.lower()

    list_remove = []
    for c in text:
        if c != ' ':
            if (ord(c) >= ord('a') and ord(c) <= ord('z')) or (ord(c) >= ord('0') and ord(c) <= ord('9')):
                pass
            else:
                list_remove.append(c)
    for c in list_remove:
        text = text.replace(c, ' ')
    text = ' '.join(text.split())

    return text



class LabelUtil():
    def __init__(self, file_paths=['data/mimic_ii_train.txt'],
                 label_description_path='data/ICD9_descriptions'):
        self.label_index2code, self.code2label_index = {}, {}
        self.label_index2name, self.name2label_index = {}, {}

        self.class_index2class_code, self.class_code2class_index = {}, {}
        self.class_index2label_indexs, self.label_index2class_index = {}, {}

        self.list_code = []
        self.code2name, self.name2code = {}, {}

        ## label description
        with open(label_description_path, 'r') as f:
            for line in f:
                code, name = line[:-1].split('\t')
                self.code2name[code] = name
                self.name2code[name] = code

        ## label in data
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                print(file_path)
                for line in f:
                    labels = line[:-1].strip().split('\t')[-1]
                    for code in labels.split('##'):
                        if code not in self.list_code:
                            self.list_code.append(code)

        self.list_code = list(set(self.list_code))
        self.list_code.sort(key=sort_key)
        # self.list_code = ['none'] + self.list_code

        for idx, code in enumerate(self.list_code):
            self.label_index2code[idx] = code
            self.code2label_index[code] = idx

            class_code = code.split('.')[0] # TODO: class_code oov

            if class_code not in self.class_code2class_index:
                self.class_code2class_index[class_code] = len(self.class_code2class_index)
                self.class_index2class_code[self.class_code2class_index[class_code]] = class_code

                self.class_index2label_indexs[self.class_code2class_index[class_code]] = []
            self.class_index2label_indexs[self.class_code2class_index[class_code]].append(idx)
            self.label_index2class_index[idx] = self.class_code2class_index[class_code]


        self.label_size, self.class_size = len(self.label_index2code), len(self.class_index2class_code)
        print('label:', self.label_size, 'class:', self.class_size)

        ooc_count = 0
        for code, index in self.code2label_index.items():
            if code not in self.code2name:
                name = 'OUT_OF_CODE_{:d}'.format(ooc_count)
                self.code2name[code] = name
                self.name2code[name] = code

            self.label_index2name[index] =self.code2name[code]
            self.name2label_index[self.code2name[code]] = index

        print('ooc_count', ooc_count)

