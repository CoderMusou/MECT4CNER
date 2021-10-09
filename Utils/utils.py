import collections
import math
import time
from copy import deepcopy

import datetime
import fitlog
import pytz
import torch
from fastNLP import FitlogCallback, Callback, EarlyStopError, logger


class EarlyStopCallback(Callback):
    r"""
    多少个epoch没有变好就停止训练，相关类 :class:`~fastNLP.core.callback.EarlyStopError`
    """

    def __init__(self, patience):
        r"""

        :param int patience: epoch的数量
        """
        super(EarlyStopCallback, self).__init__()
        self.patience = patience
        self.wait = 0

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        if not is_better_eval:
            # current result is getting worse
            if self.wait == self.patience:
                raise EarlyStopError("Early stopping raised.")
            else:
                self.wait += 1
        else:
            self.wait = 0

    def on_exception(self, exception):
        if isinstance(exception, EarlyStopError):
            logger.info("Early Stopping triggered in epoch {}!".format(self.epoch))
        else:
            raise exception  # 抛出陌生Error


class MyFitlogCallback(FitlogCallback):
    def __init__(self, data=None, tester=None, log_loss_every=0, verbose=0, log_exception=False):
        super().__init__(data, tester, log_loss_every, verbose, log_exception)
        self.better_test_f = 0
        self.better_test_result = None

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        if better_result:
            eval_result = deepcopy(eval_result)
            eval_result['step'] = self.step
            eval_result['epoch'] = self.epoch
            fitlog.add_best_metric(eval_result)
            # 保存最好的模型
            # torch.save(self.model.state_dict(), '/data/ws/radical_vec/model_parameter.pkl')
        fitlog.add_metric(eval_result, step=self.step, epoch=self.epoch)
        # if eval_result['SpanFPreRecMetric']['f'] < 0.2:
        #     raise EarlyStopError("Early stopping raised.")
        if len(self.testers) > 0:
            for key, tester in self.testers.items():
                try:
                    eval_result = tester.test()
                    if self.verbose != 0:
                        self.pbar.write("FitlogCallback evaluation on {}:".format(key))
                        self.pbar.write(tester._format_eval_results(eval_result))
                    fitlog.add_metric(eval_result, name=key, step=self.step, epoch=self.epoch)
                    if better_result:
                        self.better_test_f = eval_result['SpanFPreRecMetric']['f']
                        fitlog.add_best_metric(eval_result, name=key)
                except Exception as e:
                    self.pbar.write("Exception happens when evaluate on DataSet named `{}`.".format(key))
                    raise e

    def on_train_end(self):
        fitlog.finish()

    def on_exception(self, exception):
        fitlog.finish(status=1)
        if self._log_exception:
            fitlog.add_other(repr(exception), name='except_info')


def get_embedding(max_seq_len, embedding_dim, padding_idx=None, rel_pos_init=0):
    """Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    rel pos init:
    如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化，
    如果是1，那么就按-max_len,max_len来初始化
    """
    num_embeddings = 2*max_seq_len+1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    if rel_pos_init == 0:
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    else:
        emb = torch.arange(-max_seq_len,max_seq_len+1, dtype=torch.float).unsqueeze(1)*emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb


def get_crf_zero_init(label_size, include_start_end_trans=False, allowed_transitions=None,
                 initial_method=None):
    import torch.nn as nn
    from fastNLP.modules import ConditionalRandomField
    crf = ConditionalRandomField(label_size, include_start_end_trans)

    crf.trans_m = nn.Parameter(torch.zeros(size=[label_size, label_size], requires_grad=True))
    if crf.include_start_end_trans:
        crf.start_scores = nn.Parameter(torch.zeros(size=[label_size], requires_grad=True))
        crf.end_scores = nn.Parameter(torch.zeros(size=[label_size], requires_grad=True))
    return crf


def norm_static_embedding(x,norm=1):
    with torch.no_grad():
        x.embedding.weight /= (torch.norm(x.embedding.weight, dim=1, keepdim=True) + 1e-12)
        x.embedding.weight *= norm


def get_bigrams(words):
    result = []
    for i, w in enumerate(words):
        if i != len(words)-1:
            result.append(words[i]+words[i+1])
        else:
            result.append(words[i]+'<end>')

    return result


class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_w = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self,w):

        current = self.root
        for c in w:
            current = current.children[c]

        current.is_w = True

    def search(self,w):
        '''

        :param w:
        :return:
        -1:not w route
        0:subroute but not word
        1:subroute and word
        '''
        current = self.root

        for c in w:
            current = current.children.get(c)

            if current is None:
                return -1

        if current.is_w:
            return 1
        else:
            return 0

    def get_lexicon(self,sentence):
        result = []
        for i in range(len(sentence)):
            current = self.root
            for j in range(i, len(sentence)):
                current = current.children.get(sentence[j])
                if current is None:
                    break

                if current.is_w:
                    result.append([i,j,sentence[i:j+1]])

        return result


def get_skip_path(chars,w_trie):
    sentence = ''.join(chars)
    result = w_trie.get_lexicon(sentence)

    return result


def print_info(*inp, islog=True, sep=' '):
    from fastNLP import logger
    if islog:
        print(*inp,sep=sep)
    else:
        inp = sep.join(map(str,inp))
        logger.info(inp)


def get_peking_time():

    tz = pytz.timezone('Asia/Shanghai')  # 东八区

    t = datetime.datetime.fromtimestamp(int(time.time()), tz).strftime('%Y_%m_%d_%H_%M_%S')
    return t
