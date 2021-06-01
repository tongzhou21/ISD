import numpy as np

import random
random.seed(0)
import math
import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModel


class ICDDataset(Dataset):
    def __init__(self, FLAGS, vocab, label_util, data_path=None, flag_eval=False):
        self.FLAGS = FLAGS
        self.tokenizer = AutoTokenizer.from_pretrained(FLAGS.bert_file)
        self.bert_cut_length = self.FLAGS.bert_cut_length
        self.label_util = label_util
        self.flag_eval = flag_eval
        self.data_path = data_path
        self.vocab = vocab

        ## read data
        with open(self.data_path, 'r') as f:
            self.lines = [line[:-1].strip().split('\t') for line in f]

        self.lines = [line for line in self.lines if len(line) == 2]

        self.max_text_length = FLAGS.max_text_length
        self.max_label_count = FLAGS.max_label_count

        self.valid_item = [0]
        self.noise_rate = 0 if flag_eval else FLAGS.noise_rate
        self.text_argument_ratio = FLAGS.text_argument_ratio

        ## words to argument data
        list_words = [text.split() for text, labels in self.lines]
        self.list_words = [item for sublist in list_words for item in sublist]
        print('len(list_words)', len(self.list_words))

        print('dataset.__len__', self.__len__())
        self.unk_count, self.data_count = 0, 0
        self.teacher_data_noise_length = 0


    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        return self.item2data(item)

    def item2data(self, item):
        ## data
        origin_text, labels = self.lines[item] if item < len(self.lines) else \
            self.construct_label2label_data(item - len(self.lines))

        labels = '##'.join(labels.split('##')[:self.max_label_count])
        # text = origin_text if self.flag_test else self.text_argumentation(origin_text,
        #                                                                   ratio=self.text_argument_ratio)
        text = origin_text
        text = self.cut_text(text, self.max_text_length - 2)

        labels = [self.label_util.code2label_index[code] for code in labels.split('##')]
        classes = [self.label_util.label_index2class_index[label_index] for label_index in labels]

        ## target label
        label_count = len(labels)
        tgt = np.zeros(self.label_util.label_size)
        tgt[np.array(labels)] = 1
        tgt_all = labels + [-1] * (self.max_label_count - len(labels))
        tgt_weight = np.ones(self.label_util.label_size)
        # tgt_weight = np.zeros(self.label_util.label_size) + 1
        # tgt_weight[np.array(labels)] = (self.label_util.label_size - len(labels)) / len(labels)

        ## target class
        tgt_class = np.zeros(self.label_util.class_size)
        tgt_class[np.array(classes)] = 1
        tgt_all_class = classes + [-1] * (self.max_label_count - len(classes))

        tgt_weight_class = np.ones(self.label_util.class_size)
        # tgt_weight_class = np.zeros(self.label_util.class_size) + 1
        # tgt_weight_class[np.array(classes)] = (self.label_util.class_size - len(classes)) / len(classes)

        ## label complete task
        label_inp_missing, label_inp_padding_mask_missing, label_tgt_missing = self.construct_label_task_data_missing(labels[:self.max_label_count])
        label_inp_false, label_inp_padding_mask_false, label_tgt_false = self.construct_label_task_data_false(labels[:self.max_label_count])

        ## argument text teacher
        labels_text = [self.label_util.label_index2name[label_id] for label_id in labels]
        random.shuffle(labels_text)
        labels_text = self.convert_label_seq_to_plain_text(labels_text, noise_length=self.teacher_data_noise_length)

        ## inp student
        input_ids, input_mask, _ = self.text2inp(text, max_length=self.max_text_length, flag_special_token=True)
        ## bluebert inp student
        list_bert_input_ids, list_bert_input_mask, mask_h_src = \
            self.convert_text_to_bert_inp(text, max_length=self.max_text_length, cut_length=self.bert_cut_length)

        ## inp student
        input_ids_teacher, input_mask_teacher, _ = self.text2inp(text, max_length=self.max_text_length, flag_special_token=True)
        ## bluebert inp teacher
        list_bert_input_ids_teacher, list_bert_input_mask_teacher, mask_h_src_teacher = \
            self.convert_text_to_bert_inp(labels_text, max_length=self.max_text_length, cut_length=self.bert_cut_length)

        dict_output = {
            'y': tgt,
            'y_class': tgt_class,
            'tgt_weight': tgt_weight,
            'tgt_weight_class': tgt_weight_class,

            'pos_all': tgt_all[:self.max_label_count],
            'pos_all_class': tgt_all_class[:self.max_label_count],

            'label_inp_missing': label_inp_missing,
            'label_inp_padding_mask_missing': label_inp_padding_mask_missing,
            'label_tgt_missing': label_tgt_missing,

            'label_inp_false': label_inp_false,
            'label_inp_padding_mask_false': label_inp_padding_mask_false,
            'label_tgt_false': label_tgt_false,

            'x': input_ids,
            'x_mask': input_mask,

            'x_teacher': input_ids_teacher,
            'x_mask_teacher': input_mask_teacher,

            'x_bert_cut': list_bert_input_ids,
            'x_bert_mask_cut': list_bert_input_mask,
            'x_bert_mask_src': mask_h_src,
            'token_count_bert_src': mask_h_src.count(0),

            'x_bert_cut_teacher': list_bert_input_ids_teacher,
            'x_bert_mask_cut_teacher': list_bert_input_mask_teacher,
            'x_bert_mask_src_teacher': mask_h_src_teacher,
            'token_count_bert_src_teacher': mask_h_src_teacher.count(0),

            'label_count': label_count,
        }

        dict_output = {key: torch.tensor(value) for key, value in dict_output.items()}
        if item not in self.valid_item:
            self.valid_item.append(item)
        return (dict_output, origin_text)


    def construct_label2label_data(self, item=None):
        item = random.randrange(self.label_util.label_size) if not item else item

        def get_co_code(code, depth):
            if depth > self.max_label_count:
                return []
            # print('get_co_code', code, len(self.label_code2co_occur_label_code[code]))
            co_code = random.sample(self.label_code2co_occur_label_code[code], 1)[0]
            if co_code == code:
                co_code = self.label_util.index2code[random.randrange(self.label_util.label_size)]
            return [co_code] + get_co_code(co_code, depth + 1)

        code_count = 10 + int(12 * (random.random() - 0.5))
        list_codes = list(set(get_co_code(self.label_util.index2code[item], 0)))
        random.shuffle(list_codes)
        list_codes = list_codes[:code_count]
        if self.label_util.index2code[item] not in list_codes:
            list_codes = [self.label_util.index2code[item]] + list_codes
        random.shuffle(list_codes)
        labels = '##'.join(list_codes)

        def construct_noise_words(depth):
            if depth > 100:
                return []
            word_count = 3 + int(6 * (random.random() - 0.5))
            words = self.sample_words(word_count=word_count)
            if random.random() < 0.98:
                return words + construct_noise_words(depth + word_count)
            else:
                return words
        text = ''
        for code in list_codes:
            label_name = self.label_util.code2text[code]
            text_left = ' '.join(construct_noise_words(0))
            text_right = ' '.join(construct_noise_words(0))
            text += text_left + ' ' + label_name + ' ' + text_right + ' '

        return text[:-1], labels



    def construct_label_task_data_missing(self, labels):
        mask_count = max(1, min(len(labels) // 2, int(1 - math.log(random.random()))))
        label_inp = labels.copy()
        random.shuffle(label_inp)
        label_tgt = label_inp[0]
        for idx in range(mask_count):
            if random.random() < 0.5:
                label_inp[idx] = random.randrange(self.label_util.label_size)
            else:
                if random.random() < 0.5:
                    label_inp[idx] = random.sample(labels, 1)[0]
                else:
                    pass
        while len(label_inp) < self.max_label_count:
            if random.random() < 0.5:
                label_inp.append(random.randrange(self.label_util.label_size))
            else:
                label_inp.append(random.sample(labels, 1)[0])

        label_inp_padding_mask = [0] * len(labels)
        label_inp_padding_mask += [1] * (self.max_label_count - len(labels))

        return label_inp, label_inp_padding_mask, label_tgt

    def construct_label_task_data_false(self, labels):
        mask_count = max(1, min(len(labels) // 2, int(1 - math.log(random.random()))))
        label_inp = labels.copy()[:self.max_label_count - 1]
        random.shuffle(label_inp)

        while len(label_inp) < self.max_label_count:
            label_inp.append(random.randrange(self.label_util.label_size))
        label_inp_padding_mask = [0] * len(label_inp)
        label_tgt = 0

        return label_inp, label_inp_padding_mask, label_tgt



    def convert_label_seq_to_plain_text(self, labels_, noise_length=100):
        labels = labels_ + ['']
        text = ''
        list_patch = [''] * (len(labels))
        for idx, patch in enumerate(list_patch):
            patch_length = random.randrange(max(1, noise_length))
            words = self.sample_words(patch_length)
            list_patch[idx] = ' '.join(words)
            text += ' ' + ' '.join(words) + labels[idx] + ' '
        return text.strip()

    def convert_text_to_inp(self, text, max_length=4000, flag_special_token=True):
        return self.text2inp(text, max_length, flag_special_token)


    def convert_text_to_bert_inp(self, text, max_length=4000, cut_length=502):
        input_ids = self.tokenizer(text)['input_ids'][1:-1][:max_length]

        count = len(input_ids)
        mask_h_src = [0] * len(input_ids) + [1] * (max_length - count)

        input_ids.extend([0] * (max_length - count))
        list_bert_input_ids = [
            ([self.tokenizer.cls_token_id] +
             input_ids[init_position:init_position + cut_length-2] +
             [self.tokenizer.sep_token_id]) for init_position in range(0, max_length, cut_length - 2)
        ]

        list_bert_input_mask = [np.array([input_id != 0 for input_id in bert_input_ids]).astype('int').tolist()
                                for bert_input_ids in list_bert_input_ids]

        return list_bert_input_ids, list_bert_input_mask, mask_h_src


    def sample_words(self, word_count):
        if word_count == 0:
            return ''
        word_idx = random.randrange(len(self.list_words) - word_count)
        words = self.list_words[word_idx: word_idx + word_count]
        return words


    def text_argumentation(self, text, ratio=1):
        if random.random () < ratio:
            return text
        # return text
        list_word = text.split()
        text_final = ''
        for word in list_word:
            ac = random.randrange(50)
            if ac == 0:
                word = ' '.join(self.sample_words(random.randrange(1, 3)))
            if ac == 1:
                word += ' ' + ' '.join(self.sample_words(random.randrange(1, 3)))
            if ac == 2:
                word = ''
            text_final += ' ' + word
        if random.random() < self.noise_rate:
            text_final = self.text_argumentation_shuffle(text_final)
        return text_final

    def text_argumentation_shuffle(self, text):
        text = text.split()
        list_patch = []
        # list_cut_idx_ = random.sample([_ for _ in range(len(text))], min(20, len(text) // 10))
        list_cut_idx_ = random.sample([_ for _ in range(len(text))], len(text) // 20)

        list_cut_idx_.sort()
        list_cut_idx = [0]
        for idx in list_cut_idx_:
            if idx - list_cut_idx[-1] < 10:
                continue
            else:
                list_cut_idx.append(idx)
        list_cut_idx.append(-1)
        for i, idx in enumerate(list_cut_idx[:-1]):
            list_patch.append(' '.join(text[idx: list_cut_idx[i + 1]]))
        random.shuffle(list_patch)
        text = ' '.join(list_patch)
        return text



    def cut_text(self, text, max_length):
        list_words = text.split()
        length = len(list_words)
        if length <= max_length:
            final = list_words
        else:
            final = []
            ac = 3 if self.flag_eval else random.randrange(4)
            if ac == -1: #
                # list_idx = random.sample([_ for _ in range(length)], max_length)
                list_idx = np.random.choice([_ for _ in range(length)], max_length, replace=False)
                list_idx.sort()
                for idx, word in enumerate(list_words):
                    if idx in list_idx:
                        final.append(word)
            if ac == 1:
                cut_length = length - max_length
                init_pos = random.randrange(0, length - cut_length)
                final = list_words[:init_pos] + list_words[init_pos + cut_length:]
            if ac == 2:
                final = list_words[-max_length:]
            if ac == 3 or final == []:
                final = list_words[:max_length]

        return ' '.join(final)

    def raw_text2tokens(self, text):
        list_word = text.split()
        tokens = list_word.copy()

        for i, token in enumerate(tokens):
            tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

        return tokens

    def text2inp(self, text, max_length, flag_special_token=True):
        text = self.cut_text(text, max_length)
        tokens = self.raw_text2tokens(text)
        tokens = [self.vocab.sos_index] + tokens + [self.vocab.eos_index] if flag_special_token else tokens
        tokens = tokens[:max_length]


        length = len(tokens)
        tokens += [self.vocab.pad_index] * (max_length - length)
        mask = [0] * length + [1] * (max_length - length)

        return tokens, mask, length
