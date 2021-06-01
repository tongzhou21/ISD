import argparse

def parse_args(flag_print=False):
    parser = argparse.ArgumentParser(description='train a neural network on some code terms')

    #### version
    parser.add_argument('--version', type=str, default='mimic_iii_full') # mimic_iii_full | mimic_ii | mimic_iii_50


    #### data
    parser.add_argument('--data_path_train', type=str, default='data/{:s}_train.txt'.format(parser.parse_args().version))
    parser.add_argument('--data_path_eval', type=str, default='data/{:s}_eval.txt'.format(parser.parse_args().version))
    parser.add_argument('--data_path_test', type=str, default='data/{:s}_test.txt'.format(parser.parse_args().version))


    #### label description
    parser.add_argument('--label_descriptions_path', type=str, default='data/ICD9_descriptions')

    #### data parameters
    parser.add_argument('--max_text_length', type=int, default=4000) # 3600
    parser.add_argument('--max_label_count', type=int, default=40) #
    parser.add_argument('--bert_cut_length', type=int, default=502)

    #### training parameters
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)  # 3e-4
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr_bert', type=float, default=3e-5)  # 3e-5
    parser.add_argument('--weight_decay_bert', type=float, default=0.01) # 0.01

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--real_batch_size', type=int, default=32)
    parser.add_argument('--warmup_proportion', default=0.1, type=float) # 5epoch

    #### model parameters
    parser.add_argument("--emb_dim", default=100, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--kernel_size", default=[5, 7, 9, 11], type=list)  # [5, 7, 9, 11]


    parser.add_argument("--share_attn_count", default=64, type=int)
    parser.add_argument('--dropout_attn', type=float, default=0.1)
    parser.add_argument("--nhead_decoder", default=1, type=int)
    parser.add_argument("--layer_count_decoder", default=2, type=int) #
    parser.add_argument("--noise_rate", default=0, type=float) # 0.05
    parser.add_argument("--text_argument_ratio", default=0, type=float) # 0.1
    parser.add_argument("--rsc_drop_count", default=40, type=int)
    parser.add_argument("--teacher_data_noise_length", default=0, type=int)

    parser.add_argument('--vocab_path', type=str, default='data/vocab_mimic.pkl')
    parser.add_argument("--bert_file", default='~/transformers_models/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12/', type=str)
    # parser.add_argument("--label_embedding", default='model_save/label_emb.bluebert.{:s}.pkl'.format(parser.parse_args().version), type=str)
    parser.add_argument("--label_embedding", default='model_save/label_emb.{:s}.{:d}.pkl'.format(parser.parse_args().version, parser.parse_args().hidden_dim), type=str)

    #### loss weights
    parser.add_argument("--weight_loss_cat", default=0, type=float)
    parser.add_argument("--weight_loss_semantic", default=0, type=float)
    parser.add_argument("--weight_loss_class", default=0, type=float)
    parser.add_argument("--weight_loss_inner", default=0, type=float)
    parser.add_argument("--weight_loss_tea", default=0, type=float)
    parser.add_argument("--weight_loss_co_occur", default=0, type=float) # 0.005
    parser.add_argument("--weight_loss_bce", default=0, type=float)
    parser.add_argument("--weight_loss_bce_word", default=0, type=float) # 0.1
    parser.add_argument("--weight_loss_sim_attribute_attn", default=0.005, type=float) # 0.005
    parser.add_argument("--weight_loss_bce_share", default=0.25, type=float) # 0.75
    parser.add_argument("--weight_loss_bce_label", default=0.25, type=float) # 0.25
    parser.add_argument("--weight_loss_bce_class", default=0, type=float) # 0.5 #

    parser.add_argument("--weight_loss_bce_share_teacher", default=0.25, type=float)
    parser.add_argument("--weight_loss_bce_label_teacher", default=0.25, type=float)
    parser.add_argument("--weight_loss_distill", default=0.001, type=float) # 0.5
    parser.add_argument("--weight_loss_mask_label", default=1e-3, type=float)# 0.001 # 5e-4
    parser.add_argument("--weight_loss_share_attn_regulation", default=0, type=float)

    parser.add_argument("--flag_pos_emb", default=False, type=bool)
    parser.add_argument("--flag_coocc", default=True, type=bool)

    #### save
    parser.add_argument("--last_checkpoint", default=None, type=str) # mimic_iii_full.distill.gpu0.last.checkpoint.pkl

    parser.add_argument("--model_pkl_pretrain", default=None, type=str)
    parser.add_argument("--model_name", default='model_save/{:s}.cnn.distill.apex'.format(parser.parse_args().version))
    parser.add_argument("--train_type", default='teacher+student+distill')
    parser.add_argument("--visible_device", default='1', type=str)
    parser.add_argument("--apex", default=False, type=bool)
    parser.add_argument('--freeze_bert', type=bool, default=False)

    parser.add_argument("--random_seed", default=1234, type=int)

    parser.add_argument("--print_case", default=False, type=bool)

    ## print configs
    args = parser.parse_args()

    if flag_print:
        for arg in vars(args):
            print('{} = {}'.format(arg.upper(), getattr(args, arg)))
        print('')

    return args

# FLAGS = parse_args()



