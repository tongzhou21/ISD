
import torch
import torch.nn as nn
# from longformer import MIMICLongformer
import torch.nn.functional as F
import math
import numpy as np
import os

from munkres import Munkres


class ModelMultiLabel(nn.Module):
    def __init__(self, FLAGS, vocab, label_util, dataset):
        super(ModelMultiLabel, self).__init__()
        print('MODEL: ' + os.path.basename(__file__))

        self.FLAGS = FLAGS
        self.num_labels = label_util.label_size
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.hidden_dim = FLAGS.hidden_size
        self.hidden_dim_label = FLAGS.hidden_size_label
        self.share_attn_count = FLAGS.share_attn_count
        self.nhead_decoder = FLAGS.nhead_decoder
        self.layer_count_decoder = FLAGS.layer_count_decoder
        self.nhead_encoder = FLAGS.nhead_encoder
        self.layer_count_encoder = FLAGS.layer_count_encoder
        # pos emb
        self.pos_emb = PositionalEncoding(d_model=self.hidden_dim, dropout=0.1, max_len=FLAGS.max_text_length)
        # token emb
        self.token_emb = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.token_emb.weight = torch.load('model_save/token_emb.full.skipgram.{:d}.pkl'.format(self.hidden_dim))

        self.init_token_emb = self.token_emb.weight.clone().to(0)
        # label emb
        self.label_emb = nn.Embedding(self.num_labels, self.hidden_dim_label)

        try:
            self.label_emb.weight = torch.load(FLAGS.label_emb_path)
        except:
            print('no label_emb init')


        self.list_kernel_size = FLAGS.kernel_size
        self.encoder_text = nn.ModuleList([
            nn.Conv1d(self.hidden_dim, self.hidden_dim_label // len(self.list_kernel_size),
                      kernel_size=kernel_size, padding=kernel_size//2) for kernel_size in self.list_kernel_size
        ])
        # # rnn
        # self.encoder_text = nn.LSTM(self.hidden_dim, self.hidden_dim_label // 2,
        #                             num_layers=1, batch_first=True, bidirectional=True)
        # # transformer
        # self.encoder_text = nn.ModuleList(
        #     [nn.TransformerEncoderLayer(self.hidden_dim_label, self.nhead_encoder, dim_feedforward=self.hidden_dim_label * 4,
        #         dropout=FLAGS.dropout_attn, activation='relu') for _ in range(self.layer_count_encoder)])
        # # longformer
        # self.encoder_text = MIMICLongformer(
        #     attention_window=[16, 16],
        #     num_hidden_layers=2,
        #     hidden_size=self.hidden_dim,
        #     intermediate_size=self.hidden_dim * 4,
        #     num_attention_heads=2
        # )
        # self.encoder_text.init_word_embedding(self.token_emb.weight)

        self.share_attn_emb = nn.Embedding(self.share_attn_count, self.hidden_dim_label)

        self.share_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim_label, num_heads=1, dropout=FLAGS.dropout_attn)
        self.label_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim_label, num_heads=1, dropout=FLAGS.dropout_attn)

        self.co_occur_decoder = nn.ModuleList(
            [nn.TransformerDecoderLayer(self.hidden_dim_label, self.nhead_decoder, dim_feedforward = self.hidden_dim_label * 4,
                dropout=FLAGS.dropout_attn, activation='relu') for _ in range(self.layer_count_decoder)])

        self.label_bias = nn.Parameter(torch.randn(self.num_labels))

        self.dropout = nn.Dropout(p=FLAGS.dropout_emb)
        self.layernorm = nn.LayerNorm(self.hidden_dim_label)
        self.layernorm_emb = nn.LayerNorm(self.hidden_dim)
        # self.loss_function = F.binary_cross_entropy_with_logits()
        self.munkres = Munkres()
        self.ff_count = 0
        self.last_distill_loss = None

    def encode(self, x, x_mask, token_count):
        bsz = x.size(0)

        h_emb_token = self.dropout(self.token_emb(x))
        h_emb = self.layernorm_emb(self.pos_emb(h_emb_token.transpose(0, 1)).transpose(0, 1)) \
            if self.FLAGS.flag_pos_emb else h_emb_token
        # h_emb = h_emb.detach()

        cnn_output = torch.cat([torch.tanh(conv(h_emb.permute(0, 2, 1))) for conv in self.encoder_text], dim=1)
        h_src = cnn_output.transpose(1, 2)
        # # lstm
        # h_src, _ = self.encoder_text(h_emb)
        # # transformer
        # h_src = self.layernorm_emb(self.pos_emb(h_emb_token.transpose(0, 1)).transpose(0, 1)).transpose(0, 1)
        # for layer_id, layer in enumerate(self.encoder_text):
        #     h_src = layer(src=h_src, src_key_padding_mask=dict_data['x_mask'].bool())
        # h_src = h_src.transpose(0, 1)
        # # longformer
        # outputs = self.encoder_text(dict_data['x'])
        # h_src, pooled = outputs

        h_src_ = h_src * (1 - x_mask).unsqueeze(-1).repeat(1, 1, h_src.size(-1))
        avg_pooled = h_src_.sum(dim=1) / token_count.unsqueeze(-1)

        return h_src, avg_pooled

    def caculate_share_attention(self, share_attn_query,
                                 h_src, h_src_mask,
                                 label_emb, label_bias,
                                 tgt, tgt_weight,
                                 prob_ratio_share=0):
        bsz = h_src.size(0)

        h_share_attn = self.share_attn(share_attn_query.unsqueeze(1).repeat(1, bsz, 1),
                                       h_src.transpose(0, 1), h_src.transpose(0, 1),
                                       key_padding_mask=h_src_mask.bool())[0].transpose(0, 1) # over cnn

        h_share_attn_context = self.co_occur_decode(h_share_attn, None, h_src, h_src_mask.bool())
        h_share_attn = h_share_attn_context if self.FLAGS.flag_coocc else h_share_attn


        pred_label_share_attn = torch.matmul(label_emb.unsqueeze(0), h_share_attn.transpose(-1, -2))
        pred_label_share = torch.max(pred_label_share_attn, dim=-1)[0].add(label_bias)

        prob_sigmoid_share = torch.sigmoid(pred_label_share)
        loss_pred_label_share = F.binary_cross_entropy(prob_sigmoid_share.double(), tgt, weight=tgt_weight)

        h_label_attn = self.label_attn(label_emb.unsqueeze(1).repeat(1, bsz, 1),
                                       h_share_attn.transpose(0, 1), h_share_attn.transpose(0, 1)
                                       )[0].transpose(0, 1) # over share attn

        pred_label_label = label_emb.mul(h_label_attn).sum(dim=-1).add(label_bias)
        prob_sigmoid_label = torch.sigmoid(pred_label_label)
        loss_pred_label_label = F.binary_cross_entropy(prob_sigmoid_label.double(), tgt, weight=tgt_weight)


        prob_sigmoid_final = prob_ratio_share * prob_sigmoid_share + (1 - prob_ratio_share) * prob_sigmoid_label

        return prob_sigmoid_final, loss_pred_label_share, loss_pred_label_label, h_share_attn


    def forward(self, dict_data, train_type=None):
        bsz = dict_data['x'].size(0)

        ######## student ########
        h_src, avg_pooled = self.encode(dict_data['x'], dict_data['x_mask'], dict_data['token_count'])

        # share attn & label attn
        prob_sigmoid_final, loss_pred_label_share, loss_pred_label_label, h_share_attn, = \
            self.caculate_share_attention(share_attn_query=self.share_attn_emb.weight,
                                          h_src=h_src, h_src_mask=dict_data['x_mask'],
                                          label_emb=self.label_emb.weight, label_bias=self.label_bias,
                                          tgt=dict_data['y'], tgt_weight=dict_data['tgt_weight'],
                                          prob_ratio_share=1) # only share
        loss_pred_label_final = F.binary_cross_entropy(prob_sigmoid_final.double(), dict_data['y'],
                                                       weight=dict_data['tgt_weight'])

        ######## teacher ########
        h_src_labels_cut, _ = self.encode(dict_data['x_labels_cut'].reshape(-1, dict_data['x_labels_cut'].size(-1)),
                                          dict_data['x_mask_labels_cut'].reshape(-1, dict_data['x_mask_labels_cut'].size(-1)),
                                          dict_data['token_count_labels_cut'].reshape(-1))
        h_src_labels_cut = h_src_labels_cut.reshape(bsz, -1, h_src_labels_cut.size(-1))
        h_src_labels_cut_mask = dict_data['x_mask_labels_cut'].reshape(bsz, -1)
        prob_sigmoid_final_labels, loss_pred_label_share_labels, loss_pred_label_label_labels, h_share_attn_labels = \
            self.caculate_share_attention(share_attn_query=self.share_attn_emb.weight,
                                          h_src=h_src_labels_cut, h_src_mask=h_src_labels_cut_mask,
                                          label_emb=self.label_emb.weight, label_bias=self.label_bias,
                                          tgt=dict_data['y'], tgt_weight=dict_data['tgt_weight'],
                                          prob_ratio_share=1)

        ######## distill ########
        if 'distill' in train_type:
            loss_distill = self.distill_loss(student=h_share_attn, teacher=h_share_attn_labels)
        else:
            loss_distill = torch.zeros(1).to(0)

        loss_mask_label_missing = self.mask_label_task_missing(h_src, dict_data)
        loss_mask_label_false = self.mask_label_task_false(h_src, dict_data)
        loss_mask_label = (loss_mask_label_missing + loss_mask_label_false) / 2
        # loss_mask_label = 0.8  * loss_mask_label_missing + 0.2 * loss_mask_label_false
        # loss_mask_label = (loss_mask_label_missing + loss_mask_label_missing) / 2

        loss_teacher = 0.5 * (loss_pred_label_share_labels + loss_pred_label_share_labels)
        loss_student = 0.5 * (loss_pred_label_share + loss_pred_label_share)

        teacher_ratio, student_ratio, distill_ratio = 0, 0, 0
        if 'teacher' in train_type and 'student' in train_type:
            teacher_ratio, student_ratio = 0.5, 0.5
        if 'teacher' in train_type and 'student' not in train_type:
            teacher_ratio, student_ratio = 0.95, 0.05
        if 'teacher' not in train_type and 'student' in train_type:
            teacher_ratio, student_ratio = 0.05, 0.95
        if 'teacher' not in train_type and 'student' not in train_type:
            teacher_ratio, student_ratio = 0, 0
        if 'distill' in train_type:
            distill_ratio = 1

        loss_total = teacher_ratio * loss_teacher + \
                     student_ratio * loss_student + \
                     distill_ratio * self.FLAGS.weight_loss_distill * loss_distill + \
                     self.FLAGS.weight_loss_mask_label * loss_mask_label

        indicator = {
            '100*loss_pred_final': 100 * loss_pred_label_final.cpu().item(),
            '100*loss_pred_share': 100 * loss_pred_label_share.cpu().item(),
            '100*loss_pred_label': 100 * loss_pred_label_label.cpu().item(),
            '100*loss_pred_share_teacher': 100 * loss_pred_label_share_labels.cpu().item(),
            '100*loss_pred_label_teacher': 100 * loss_pred_label_label_labels.cpu().item(),
            'loss_distill': loss_distill.cpu().item(),

            'prob_mean': prob_sigmoid_final.view(-1).mean().cpu().item(),

            'count1': (prob_sigmoid_final > 0.5).float().sum(-1).view(-1).mean().cpu().item(),
            'token_emb_diff': torch.abs(self.init_token_emb - self.token_emb.weight).view(-1).sum().cpu().item(),
            'loss_mask_label': loss_mask_label.cpu().item(),
            'loss_mask_label_missing': loss_mask_label_missing.cpu().item(),
            'loss_mask_label_false': loss_mask_label_false.cpu().item(),

            'indicator_for_lr': loss_total.cpu().item(),
        }

        return prob_sigmoid_final, loss_total, indicator

    def mask_label_task_missing(self, h_src, dict_data):
        x = self.label_emb(dict_data['label_inp_missing'])
        h = self.co_occur_decode(x, dict_data['label_inp_padding_mask_missing'].bool(), h_src, dict_data['x_mask'].bool())
        h_mask = h[:, 0]
        pred = F.linear(h_mask, self.label_emb.weight, bias=self.label_bias)
        loss_mask = F.cross_entropy(pred, dict_data['label_tgt_missing'])

        return loss_mask

    def mask_label_task_false(self, h_src, dict_data):
        x = self.label_emb(dict_data['label_inp_false'])
        h = self.co_occur_decode(x, dict_data['label_inp_padding_mask_false'].bool(), h_src, dict_data['x_mask'].bool())
        h_mask = h[:, -1]
        pred = F.linear(h_mask, self.label_emb.weight, bias=self.label_bias)
        loss_mask = F.cross_entropy(pred, dict_data['label_tgt_false'])

        return loss_mask



    def co_occur_decode(self, h_share_attn, h_share_attn_mask, h_src, h_src_mask):
        h_share_attn_context = h_share_attn.transpose(0, 1)

        for layer_id, layer in enumerate(self.co_occur_decoder):
            h_share_attn_context = layer(tgt=h_share_attn_context, memory=h_src.transpose(0, 1),
                                         tgt_key_padding_mask=h_share_attn_mask, memory_key_padding_mask=h_src_mask)
        h_share_attn_context = h_share_attn_context.transpose(0, 1)

        return h_share_attn_context

    def distill_loss(self, student, teacher):
        bsz = teacher.size(0)
        matrix_sim = torch.cosine_similarity(student.unsqueeze(1), teacher.unsqueeze(2), dim=-1)
        np_matrix_sim = matrix_sim.detach().cpu().numpy() * 100
        np_matrix_sim = np_matrix_sim.astype('int')
        sim_all, sim_count = 0, 0
        for b in range(1): #
            indexes = self.munkres.compute(-np_matrix_sim[b])
            for i, j in indexes:
                sim_all += matrix_sim[b][i][j]
                sim_count += 1
            # print(sim_all / sim_count)
        sim_all /= sim_count
        loss_distill = 1 - sim_all
        return loss_distill


    def share_attn_regulation_loss(self, h_div, pred_label_share, h_global, loss_pred_label, dict_data):
        loss_div = max(torch.tensor(0), torch.cosine_similarity(
            h_div.unsqueeze(0), h_div.unsqueeze(1), dim=-1).view(-1).mean())

        # loss_dist = torch.softmax(
        #     torch.mean(pred_label_share, dim=1),
        #     dim=-1).max(dim=-1)[0].mean()
        loss_dist = torch.mean(
            torch.softmax(pred_label_share, dim=-1),
            dim=1).max(dim=-1)[0].mean()

        pred_label_avg = F.linear(h_global, self.label_emb.weight, bias=self.label_bias)
        loss_pred_label_avg = F.binary_cross_entropy(torch.sigmoid(pred_label_avg).double(), dict_data['y'],
                                                 weight=dict_data['tgt_weight'])
        loss_rel = max(torch.tensor(0), loss_pred_label- loss_pred_label_avg)


        indicator = {
            'loss_dist': loss_dist.cpu().item(),
            'loss_div': loss_div.cpu().item(),
            'loss_rel': loss_rel.cpu().item(),
        }
        loss_total = loss_dist + loss_div + loss_rel
        return loss_total, indicator


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
