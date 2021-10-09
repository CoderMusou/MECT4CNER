import fitlog

from Modules.CNNRadicalLevelEmbedding import CNNRadicalLevelEmbedding
from Utils.load_data import *
from Utils.paths import *
from Utils.utils import norm_static_embedding, MyFitlogCallback, print_info, get_peking_time
from model import MECTNER

use_fitlog = False
if not use_fitlog:
    fitlog.debug()
fitlog.set_log_dir('logs')
load_dataset_seed = 100
fitlog.add_hyper(load_dataset_seed, 'load_dataset_seed')
fitlog.set_rng_seed(load_dataset_seed)
import sys

sys.path.append('../')
import argparse
from fastNLP.core import Trainer
# from trainer import Trainer
from fastNLP.core import Callback
import torch
import collections
import torch.optim as optim
import torch.nn as nn
from fastNLP import LossInForward
from fastNLP.core.metrics import SpanFPreRecMetric, AccuracyMetric
from fastNLP.core.callback import WarmupCallback, GradientClipCallback, EarlyStopCallback
from fastNLP import LRScheduler, DataSetIter, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
# from models import LSTM_SeqLabel,LSTM_SeqLabel_True
from fastNLP import logger

import traceback
import warnings
import sys

# print('tuner_params：')
# print(tuner_params)
# exit()

parser = argparse.ArgumentParser()

parser.add_argument('--status', default='train', choices=['train'])
parser.add_argument('--msg', default='_')
parser.add_argument('--train_clip', default=False, help='是不是要把train的char长度限制在200以内')
parser.add_argument('--device', default='0')
parser.add_argument('--debug', default=0, type=int)
parser.add_argument('--gpumm', default=False, help='查看显存')
parser.add_argument('--see_convergence', default=False)
parser.add_argument('--see_param', default=False)
parser.add_argument('--test_batch', default=-1)
parser.add_argument('--seed', default=100, type=int)
parser.add_argument('--test_train', default=False)
parser.add_argument('--number_normalized', type=int, default=0,
                    choices=[0, 1, 2, 3], help='0不norm，1只norm char,2norm char和bigram，3norm char，bigram和lattice')
parser.add_argument('--lexicon_name', default='yj', choices=['lk', 'yj'])
parser.add_argument('--update_every', default=1, type=int)
parser.add_argument('--use_pytorch_dropout', type=int, default=0)

parser.add_argument('--char_min_freq', default=1, type=int)
parser.add_argument('--bigram_min_freq', default=1, type=int)
parser.add_argument('--lattice_min_freq', default=1, type=int)
parser.add_argument('--only_train_min_freq', default=True)
parser.add_argument('--only_lexicon_in_train', default=False)

parser.add_argument('--word_min_freq', default=1, type=int)

# hyper of training
# parser.add_argument('--early_stop',default=40,type=int)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--batch', default=10, type=int)
parser.add_argument('--optim', default='sgd', help='sgd|adam')
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--embed_lr_rate', default=1, type=float)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--init', default='uniform', help='norm|uniform')
parser.add_argument('--self_supervised', default=False)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--norm_embed', default=True)
parser.add_argument('--norm_lattice_embed', default=True)

parser.add_argument('--warmup', default=0.1, type=float)

# hyper of model
parser.add_argument('--use_bert', type=int)
parser.add_argument('--model', default='transformer', help='lstm|transformer')
parser.add_argument('--lattice', default=1, type=int)
parser.add_argument('--use_bigram', default=1, type=int)
parser.add_argument('--hidden', default=-1, type=int)
parser.add_argument('--ff', default=3, type=int)
parser.add_argument('--layer', default=1, type=int)
parser.add_argument('--head', default=8, type=int)
parser.add_argument('--head_dim', default=20, type=int)
parser.add_argument('--scaled', default=False)
parser.add_argument('--ff_activate', default='relu', help='leaky|relu')

parser.add_argument('--k_proj', default=False)
parser.add_argument('--q_proj', default=True)
parser.add_argument('--v_proj', default=True)
parser.add_argument('--r_proj', default=True)

parser.add_argument('--attn_ff', default=False)

# parser.add_argument('--rel_pos', default=False)
parser.add_argument('--use_abs_pos', default=False)
parser.add_argument('--use_rel_pos', default=True)
# 相对位置和绝对位置不是对立的，可以同时使用
parser.add_argument('--rel_pos_shared', default=True)
parser.add_argument('--add_pos', default=False)
parser.add_argument('--learn_pos', default=False)
parser.add_argument('--pos_norm', default=False)
parser.add_argument('--rel_pos_init', default=1)
parser.add_argument('--four_pos_shared', default=True, help='只针对相对位置编码，指4个位置编码是不是共享权重')
parser.add_argument('--four_pos_fusion', default='ff_two', choices=['ff', 'attn', 'gate', 'ff_two', 'ff_linear'],
                    help='ff就是输入带非线性隐层的全连接，'
                         'attn就是先计算出对每个位置编码的加权，然后求加权和'
                         'gate和attn类似，只不过就是计算的加权多了一个维度')

parser.add_argument('--four_pos_fusion_shared', default=True, help='是不是要共享4个位置融合之后形成的pos')

# parser.add_argument('--rel_pos_scale',default=2,help='在lattice且用相对位置编码时，由于中间过程消耗显存过大，'
#                                                  '所以可以使4个位置的初始embedding size缩小，'
#                                                  '最后融合时回到正常的hidden size即可')

parser.add_argument('--pre', default='')
parser.add_argument('--post', default='nda')

over_all_dropout = -1
parser.add_argument('--embed_dropout_before_pos', default=False)
parser.add_argument('--embed_dropout', default=0.5, type=float)
parser.add_argument('--gaz_dropout', default=0.5, type=float)
parser.add_argument('--char_dropout', default=0, type=float)
parser.add_argument('--output_dropout', default=0.3, type=float)
parser.add_argument('--pre_dropout', default=0.5, type=float)
parser.add_argument('--post_dropout', default=0.3, type=float)
parser.add_argument('--ff_dropout', default=0.15, type=float)
parser.add_argument('--ff_dropout_2', default=-1, type=float)
parser.add_argument('--attn_dropout', default=0, type=float)
parser.add_argument('--embed_dropout_pos', default='0')
parser.add_argument('--abs_pos_fusion_func', default='nonlinear_add',
                    choices=['add', 'concat', 'nonlinear_concat', 'nonlinear_add', 'concat_nonlinear', 'add_nonlinear'])

parser.add_argument('--dataset', default='weibo', help='weibo|resume|ontonotes|msra')
parser.add_argument('--label', default='all', help='ne|nm|all')

args = parser.parse_args()
if args.ff_dropout_2 < 0:
    args.ff_dropout_2 = args.ff_dropout

if over_all_dropout > 0:
    args.embed_dropout = over_all_dropout
    args.output_dropout = over_all_dropout
    args.pre_dropout = over_all_dropout
    args.post_dropout = over_all_dropout
    args.ff_dropout = over_all_dropout
    args.attn_dropout = over_all_dropout

if args.lattice and args.use_rel_pos and args.update_every == 1:
    args.train_clip = True

now_time = get_peking_time()
logger.add_file('log/{}'.format(now_time), level='info')
if args.test_batch == -1:
    args.test_batch = args.batch // 2
fitlog.add_hyper(now_time, 'time')
if args.debug:
    # args.dataset = 'toy'
    pass

if args.device != 'cpu':
    assert args.device.isdigit()
    device = torch.device('cuda:{}'.format(args.device))
else:
    device = torch.device('cpu')

refresh_data = False
# import random
# print('**'*12,random.random,'**'*12)


for k, v in args.__dict__.items():
    print_info('{}:{}'.format(k, v))

raw_dataset_cache_name = os.path.join('cache', args.dataset +
                                      '_trainClip:{}'.format(args.train_clip)
                                      + 'bgminfreq_{}'.format(args.bigram_min_freq)
                                      + 'char_min_freq_{}'.format(args.char_min_freq)
                                      + 'word_min_freq_{}'.format(args.word_min_freq)
                                      + 'only_train_min_freq{}'.format(args.only_train_min_freq)
                                      + 'number_norm{}'.format(args.number_normalized)
                                      + 'load_dataset_seed{}'.format(load_dataset_seed)
                                      )

if args.dataset == 'ontonotes':
    datasets, vocabs, embeddings = load_ontonotes4ner(ontonote4ner_cn_path, yangjie_rich_pretrain_unigram_path,
                                                      yangjie_rich_pretrain_bigram_path,
                                                      _refresh=refresh_data, index_token=False,
                                                      train_clip=args.train_clip,
                                                      _cache_fp=raw_dataset_cache_name,
                                                      char_min_freq=args.char_min_freq,
                                                      bigram_min_freq=args.bigram_min_freq,
                                                      only_train_min_freq=args.only_train_min_freq
                                                      )
elif args.dataset == 'resume':
    datasets, vocabs, embeddings = load_resume_ner(resume_ner_path, yangjie_rich_pretrain_unigram_path,
                                                   yangjie_rich_pretrain_bigram_path,
                                                   _refresh=refresh_data, index_token=False,
                                                   _cache_fp=raw_dataset_cache_name,
                                                   char_min_freq=args.char_min_freq,
                                                   bigram_min_freq=args.bigram_min_freq,
                                                   only_train_min_freq=args.only_train_min_freq
                                                   )
elif args.dataset == 'weibo':
    if args.label == 'ne':
        raw_dataset_cache_name = 'ne' + raw_dataset_cache_name
    elif args.label == 'nm':
        raw_dataset_cache_name = 'nm' + raw_dataset_cache_name
    datasets, vocabs, embeddings = load_weibo_ner(weibo_ner_path, yangjie_rich_pretrain_unigram_path,
                                                  yangjie_rich_pretrain_bigram_path,
                                                  _refresh=refresh_data, index_token=False,
                                                  _cache_fp=raw_dataset_cache_name,
                                                  char_min_freq=args.char_min_freq,
                                                  bigram_min_freq=args.bigram_min_freq,
                                                  only_train_min_freq=args.only_train_min_freq
                                                  )
elif args.dataset == 'msra':
    datasets, vocabs, embeddings = load_msra_ner_1(msra_ner_cn_path, yangjie_rich_pretrain_unigram_path,
                                                   yangjie_rich_pretrain_bigram_path,
                                                   _refresh=refresh_data, index_token=False, train_clip=args.train_clip,
                                                   _cache_fp=raw_dataset_cache_name,
                                                   char_min_freq=args.char_min_freq,
                                                   bigram_min_freq=args.bigram_min_freq,
                                                   only_train_min_freq=args.only_train_min_freq
                                                   )

if args.gaz_dropout < 0:
    args.gaz_dropout = args.embed_dropout

args.hidden = args.head_dim * args.head
args.ff = args.hidden * args.ff
args.q_proj = 1
args.k_proj = 0
args.v_proj = 1

if args.dataset == 'resume':
    args.ff_dropout = 0.1
    args.ff_dropout_2 = 0.1
    args.radical_dropout = 0.1
    args.char_dropout = 0.3
    args.head_dim = 16
    args.ff = 384
    args.hidden = 128
    args.warmup = 0.05
    args.lr = 0.0014
    args.components_embed_lr_rate = 0.0018
    args.momentum = 0.9
    args.epoch = 50
    args.seed = 9249
elif args.dataset == 'ontonotes':
    args.ff_dropout = 0.2
    args.ff_dropout_2 = 0.1
    args.radical_dropout = 0.4
    args.head_dim = 20
    args.ff = 480
    args.warmup = 0.1
    args.lr = 0.0005
    args.components_embed_lr_rate = 0.0005
    args.momentum = 0.9
    args.update_every = 1
    args.epoch = 100
elif args.dataset == 'msra':
    args.ff_dropout = 0.1
    args.ff_dropout_2 = 0.1
    args.radical_dropout = 0.2
    args.char_dropout = 0.1
    args.head_dim = 20
    args.ff = 480
    args.warmup = 0.1
    args.lr = 0.0014
    args.components_embed_lr_rate = 0.0012
    args.momentum = 0.85
    args.update_every = 1
    args.epoch = 100
elif args.dataset == 'weibo':
    args.ff_dropout = 0.2
    args.ff_dropout_2 = 0.4
    args.gaz_dropout = 0.5
    args.head_dim = 16
    args.ff = 384
    args.hidden = 128
    args.radical_dropout = 0.2
    args.warmup = 0.3
    args.lr = 0.0018
    args.components_embed_lr_rate = 0.0014
    args.momentum = 0.9
    args.epoch = 50

print('用的词表的路径:{}'.format(yangjie_rich_pretrain_word_path))

w_list = load_yangjie_rich_pretrain_word_list(yangjie_rich_pretrain_word_path,
                                              _refresh=refresh_data,
                                              _cache_fp='cache/{}'.format(args.lexicon_name))

cache_name = os.path.join('cache', (args.dataset + '_lattice' + '_only_train:{}' +
                                    '_trainClip:{}' + '_norm_num:{}'
                                    + 'char_min_freq{}' + 'bigram_min_freq{}' + 'word_min_freq{}' + 'only_train_min_freq{}'
                                    + 'number_norm{}' + 'lexicon_{}' + 'load_dataset_seed{}')
                          .format(args.only_lexicon_in_train,
                                  args.train_clip, args.number_normalized, args.char_min_freq,
                                  args.bigram_min_freq, args.word_min_freq, args.only_train_min_freq,
                                  args.number_normalized, args.lexicon_name, load_dataset_seed))
if args.dataset == 'weibo':
    if args.label == 'ne':
        cache_name = 'ne' + cache_name
    elif args.label == 'nm':
        cache_name = 'nm' + cache_name
datasets, vocabs, embeddings = equip_chinese_ner_with_lexicon(datasets, vocabs, embeddings,
                                                              w_list, yangjie_rich_pretrain_word_path,
                                                              _refresh=refresh_data, _cache_fp=cache_name,
                                                              only_lexicon_in_train=args.only_lexicon_in_train,
                                                              word_char_mix_embedding_path=yangjie_rich_pretrain_char_and_word_path,
                                                              number_normalized=args.number_normalized,
                                                              lattice_min_freq=args.lattice_min_freq,
                                                              only_train_min_freq=args.only_train_min_freq)

max_seq_len = max(*map(lambda x: max(x['seq_len']), datasets.values()))

for k, v in datasets.items():
    if args.lattice:
        v.set_input('lattice', 'bigrams', 'seq_len', 'target')
        v.set_input('lex_num', 'pos_s', 'pos_e')
        v.set_target('target', 'seq_len')
    else:
        v.set_input('chars', 'bigrams', 'seq_len', 'target')
        v.set_target('target', 'seq_len')

if args.norm_embed > 0:
    print('embedding:{}'.format(embeddings['char'].embedding.weight.size()))
    print('norm embedding')
    for k, v in embeddings.items():
        norm_static_embedding(v, args.norm_embed)

if args.norm_lattice_embed > 0:
    print('embedding:{}'.format(embeddings['lattice'].embedding.weight.size()))
    print('norm lattice embedding')
    for k, v in embeddings.items():
        print(k, v)
        norm_static_embedding(v, args.norm_embed)

dropout = collections.defaultdict(int)
dropout['embed'] = args.embed_dropout
dropout['gaz'] = args.gaz_dropout
dropout['output'] = args.output_dropout
dropout['pre'] = args.pre_dropout
dropout['post'] = args.post_dropout
dropout['ff'] = args.ff_dropout
dropout['ff_2'] = args.ff_dropout_2
dropout['attn'] = args.attn_dropout

fitlog.set_rng_seed(args.seed)
fitlog.add_hyper(args)

"""偏旁部首"""
embeddings['components'] = CNNRadicalLevelEmbedding(vocab=vocabs['lattice'], embed_size=50, char_emb_size=30,
                                                    filter_nums=[30],
                                                    kernel_sizes=[3], char_dropout=args.char_dropout,
                                                    dropout=args.radical_dropout, pool_method='max'
                                                    , include_word_start_end=False, min_char_freq=1)

model = MECTNER(embeddings['lattice'], embeddings['bigram'], embeddings['components'], args.hidden,
                k_proj=args.k_proj, q_proj=args.q_proj, v_proj=args.v_proj, r_proj=args.r_proj,
                label_size=len(vocabs['label']), max_seq_len=max_seq_len,
                dropout=dropout, dataset=args.dataset, ff_size=args.ff)

# print("test:", vocabs['lattice'].to_word(24))
for n, p in model.named_parameters():
    print('{}:{}'.format(n, p.size()))

with torch.no_grad():
    print_info('{}init pram{}'.format('*' * 15, '*' * 15))
    for n, p in model.named_parameters():
        if 'embedding' not in n and 'pos' not in n and 'pe' not in n \
                and 'bias' not in n and 'crf' not in n and 'randomAttention' not in n and p.dim() > 1:
            try:
                if args.init == 'uniform':
                    nn.init.xavier_uniform_(p)
                    print_info('xavier uniform init:{}'.format(n))
                elif args.init == 'norm':
                    print_info('xavier norm init:{}'.format(n))
                    nn.init.xavier_normal_(p)
            except:
                print_info(n)
                exit(1208)
    print_info('{}init pram{}'.format('*' * 15, '*' * 15))

loss = LossInForward()
encoding_type = 'bmeso'
if args.dataset == 'weibo' or args.dataset == 'tc':
    encoding_type = 'bio'

f1_metric = SpanFPreRecMetric(vocabs['label'], pred='pred', target='target', seq_len='seq_len',
                              encoding_type=encoding_type)
acc_metric = AccuracyMetric(pred='pred', target='target', seq_len='seq_len', )
acc_metric.set_metric_name('label_acc')
metrics = [
    f1_metric,
    acc_metric
]

bigram_embedding_param = list(model.bigram_embed.parameters())
gaz_embedding_param = list(model.lattice_embed.parameters())
components_embed_param = list(model.components_embed.parameters())
embedding_param = bigram_embedding_param
embedding_param = embedding_param + gaz_embedding_param
embedding_param_ids = list(map(id, embedding_param + components_embed_param))
non_embedding_param = list(filter(lambda x: id(x) not in embedding_param_ids, model.parameters()))

param_ = [{'params': non_embedding_param},
          {'params': embedding_param, 'lr': args.lr * args.embed_lr_rate},
          {'params': components_embed_param, 'lr': args.components_embed_lr_rate}]

optimizer = optim.SGD(param_, lr=args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)

if 'msra' in args.dataset:
    datasets['dev'] = datasets['test']

fitlog_evaluate_dataset = {'test': datasets['test']}
evaluate_callback = MyFitlogCallback(fitlog_evaluate_dataset, verbose=1)
lrschedule_callback = LRScheduler(lr_scheduler=LambdaLR(optimizer, lambda epoch: 1 / (1 + 0.05 * epoch)))
clip_callback = GradientClipCallback(clip_type='value', clip_value=5)

callbacks = [evaluate_callback, lrschedule_callback, clip_callback, WarmupCallback(warmup=args.warmup)]

if args.status == 'train':
    trainer = Trainer(datasets['train'], model, optimizer, loss,
                      args.batch // args.update_every,
                      update_every=args.update_every,
                      n_epochs=args.epoch,
                      dev_data=datasets['dev'],
                      metrics=metrics,
                      device=device, callbacks=callbacks, dev_batch_size=args.test_batch,
                      test_use_tqdm=False,
                      print_every=5,
                      check_code_level=-1)

    trainer.train()
