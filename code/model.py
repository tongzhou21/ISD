
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertModel, AutoModel

import math
import numpy as np
import os

from munkres import Munkres


class ModelMultiLabel(nn.Module):
    def __init__(self, FLAGS, vocab, label_util):
        super(ModelMultiLabel, self).__init__()
        print('MODEL: ' + os.path.basename(__file__))

        self.FLAGS = FLAGS
        self.num_labels = label_util.label_size
        self.share_attn_count = FLAGS.share_attn_count
        self.nhead_decoder = FLAGS.nhead_decoder
        self.layer_count_decoder = FLAGS.layer_count_decoder

        ## dim
        self.emb_dim = self.FLAGS.emb_dim
        self.hidden_dim = self.emb_dim
        self.hidden_dim_label = self.FLAGS.hidden_dim

        ## word emb
        self.vocab = vocab
        self.token_emb = nn.Embedding(len(self.vocab), self.emb_dim)

        ## label emb
        self.label_emb = nn.Embedding(self.num_labels, self.hidden_dim_label)
        self.label_emb.weight = torch.load(FLAGS.label_embedding)
        self.label_bias = nn.Parameter(torch.randn(self.num_labels))

        ## encoder
        self.list_kernel_size = FLAGS.kernel_size
        self.encoder_text = nn.ModuleList([
            nn.Conv1d(self.hidden_dim, self.hidden_dim_label // len(self.list_kernel_size),
                      kernel_size=kernel_size, padding=kernel_size//2) for kernel_size in self.list_kernel_size
        ])

        ## shared attention
        self.share_attn_emb = nn.Embedding(self.share_attn_count, self.hidden_dim_label)
        self.share_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim_label, num_heads=1, dropout=FLAGS.dropout_attn)
        self.label_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim_label, num_heads=1, dropout=FLAGS.dropout_attn)

        self.co_occur_decoder = nn.ModuleList(
            [nn.TransformerDecoderLayer(self.hidden_dim_label, self.nhead_decoder, dim_feedforward = self.hidden_dim_label * 4,
                dropout=FLAGS.dropout_attn, activation='relu') for _ in range(self.layer_count_decoder)])

        self.dropout = nn.Dropout(0.1)
        self.munkres = Munkres()

    def encode(self, x, x_mask):
        bsz = x.size(0)

        h_emb = self.dropout(self.token_emb(x))

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

        return h_src


    def caculate_share_attention(self, share_attn_query,
                                 h_src, h_src_mask,
                                 label_emb, label_bias,
                                 tgt, tgt_weight,
                                 prob_ratio_share=0):

        bsz = h_src.size(0)

        h_share_attn = self.share_attn(share_attn_query.unsqueeze(1).repeat(1, bsz, 1),
                                       h_src.transpose(0, 1), h_src.transpose(0, 1),
                                       key_padding_mask=h_src_mask.bool())[0].transpose(0, 1) # over cnn

        h_share_attn, alpha = self.share_attn(share_attn_query.unsqueeze(1).repeat(1, bsz, 1),
                                              h_src.transpose(0, 1), h_src.transpose(0, 1),
                                              key_padding_mask=h_src_mask.bool())
        h_share_attn = h_share_attn.transpose(0, 1)
        std_dev = torch.sqrt(torch.mean(torch.abs(alpha - (1 / alpha.size(-1))) ** 2, dim=-1)).view(-1).mean()

        h_share_attn_context = self.co_occur_decode(h_share_attn, None, h_src, h_src_mask.bool())
        h_share_attn = h_share_attn_context if self.FLAGS.flag_coocc else h_share_attn


        pred_label_share_attn = torch.matmul(label_emb.unsqueeze(0), h_share_attn.transpose(-1, -2))
        pred_label_share = torch.max(pred_label_share_attn, dim=-1)[0].add(label_bias)

        prob_sigmoid_share = torch.sigmoid(pred_label_share)
        # loss_pred_label_share = F.binary_cross_entropy(prob_sigmoid_share.double(), tgt, weight=tgt_weight)
        loss_pred_label_share = F.binary_cross_entropy_with_logits(pred_label_share, tgt, weight=tgt_weight)

        h_label_attn = self.label_attn(label_emb.unsqueeze(1).repeat(1, bsz, 1),
                                       h_share_attn.transpose(0, 1), h_share_attn.transpose(0, 1)
                                       )[0].transpose(0, 1) # over share attn

        pred_label_label = label_emb.mul(h_label_attn).sum(dim=-1).add(label_bias)
        prob_sigmoid_label = torch.sigmoid(pred_label_label)
        # loss_pred_label_label = F.binary_cross_entropy(prob_sigmoid_label.double(), tgt, weight=tgt_weight)
        loss_pred_label_label = F.binary_cross_entropy_with_logits(pred_label_label, tgt, weight=tgt_weight)


        prob_sigmoid_final = prob_ratio_share * prob_sigmoid_share + (1 - prob_ratio_share) * prob_sigmoid_label

        return prob_sigmoid_final, loss_pred_label_share, loss_pred_label_label, h_share_attn, std_dev


    def forward(self, dict_data, train_type=None):
        bsz = dict_data['y'].size(0)

        ######## student ########
        h_src = self.encode(dict_data['x'], dict_data['x_mask'])

        # share attn & label attn
        prob_sigmoid_final, loss_pred_label_share, loss_pred_label_label, h_share_attn, std_dev = \
            self.caculate_share_attention(share_attn_query=self.share_attn_emb.weight,
                                          h_src=h_src, h_src_mask=dict_data['x_mask'],
                                          label_emb=self.label_emb.weight, label_bias=self.label_bias,
                                          tgt=dict_data['y'], tgt_weight=dict_data['tgt_weight'],
                                          prob_ratio_share=0.5)
        # loss_pred_label_final = F.binary_cross_entropy(prob_sigmoid_final.double(), dict_data['y'],
        #                                                weight=dict_data['tgt_weight'])

        ######## teacher ########
        h_src_teacher = self.encode(dict_data['x_teacher'], dict_data['x_mask_teacher'])

        # share attn & label attn
        prob_sigmoid_final_labels, loss_pred_label_share_labels, loss_pred_label_label_labels, h_share_attn_labels, std_dev_teacher = \
            self.caculate_share_attention(share_attn_query=self.share_attn_emb.weight,
                                          h_src=h_src_teacher, h_src_mask=dict_data['x_bert_mask_src_teacher'],
                                          label_emb=self.label_emb.weight, label_bias=self.label_bias,
                                          tgt=dict_data['y'], tgt_weight=dict_data['tgt_weight'],
                                          prob_ratio_share=0.5)

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

        loss_teacher = 0.5 * (loss_pred_label_share_labels + loss_pred_label_label_labels)
        loss_student = 0.5 * (loss_pred_label_share + loss_pred_label_label)

        loss_total = 0.5 * loss_teacher.float() + \
                     0.5 * loss_student.float() + \
                     self.FLAGS.weight_loss_distill * loss_distill.float() + \
                     self.FLAGS.weight_loss_mask_label * loss_mask_label.float()

        indicator = {
            # '100*loss_pred_final': 100 * loss_pred_label_final.cpu().item(),
            '100*loss_pred_share': 100 * loss_pred_label_share.cpu().item(),
            '100*loss_pred_label': 100 * loss_pred_label_label.cpu().item(),
            '100*loss_pred_share_teacher': 100 * loss_pred_label_share_labels.cpu().item(),
            '100*loss_pred_label_teacher': 100 * loss_pred_label_label_labels.cpu().item(),
            'loss_distill': loss_distill.cpu().item(),

            'prob_mean': prob_sigmoid_final.view(-1).mean().cpu().item(),

            'count1': (prob_sigmoid_final > 0.5).float().sum(-1).view(-1).mean().cpu().item(),
            # 'token_emb_diff': torch.abs(self.init_token_emb - self.token_emb.weight).view(-1).sum().cpu().item(),
            'loss_mask_label': loss_mask_label.cpu().item(),
            'loss_mask_label_missing': loss_mask_label_missing.cpu().item(),
            'loss_mask_label_false': loss_mask_label_false.cpu().item(),
            '100*std_dev': std_dev.cpu().item() * 100,
            '100*std_dev_teacher': std_dev_teacher.item() * 100,
            'indicator_for_lr': loss_total.cpu().item(),
        }

        return prob_sigmoid_final, loss_total, indicator

    def mask_label_task_missing(self, h_src, dict_data):
        x = self.label_emb(dict_data['label_inp_missing'])
        h = self.co_occur_decode(x, dict_data['label_inp_padding_mask_missing'].bool(), h_src, dict_data['x_bert_mask_src'].bool())
        h_mask = h[:, 0]
        pred = F.linear(h_mask, self.label_emb.weight, bias=self.label_bias)
        loss_mask = F.cross_entropy(pred, dict_data['label_tgt_missing'])

        return loss_mask

    def mask_label_task_false(self, h_src, dict_data):
        x = self.label_emb(dict_data['label_inp_false'])
        h = self.co_occur_decode(x, dict_data['label_inp_padding_mask_false'].bool(), h_src, dict_data['x_bert_mask_src'].bool())
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

