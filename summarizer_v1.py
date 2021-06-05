# -*- coding: cp949 -*- 

from konlpy.tag import Komoran
from collections import Counter
from collections import defaultdict
from scipy.sparse import csr_matrix
import kss
import numpy as np
import math
import scipy as sp
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize


def scan_vocabulary(sents, tokenize=None, min_count=2):
    #중복제거하면서 단어 빈도 기록
    counter = Counter(w for sent in sents for w in tokenize(sent))
    #최소 2번이상 나온것만 추림.
    counter = {w:c for w,c in counter.items() if c >= min_count}
    #print(counter)
    #내림차순
    idx_to_word = [ w for w, _ in sorted(counter.items(), key=lambda x:-x[1])]
    #print('id',idx_to_vocab)
    word_to_idx = {word:idx for idx, word in enumerate(idx_to_word)}
    #print('vi',vocab_to_idx)
    return idx_to_word, word_to_idx


def sent_graph(sents, tokenize=None, min_count=2, min_sim=0.3, vocab_to_idx=None):
    

    if vocab_to_idx is None:
        idx_to_vocab, vocab_to_idx = scan_vocabulary(sents, tokenize, min_count)
    else:
        idx_to_vocab = [vocab for vocab, _ in sorted(vocab_to_idx.items(), key=lambda x:x[1])]

    x = vectorize_sents(sents, tokenize, vocab_to_idx)
    xt = numpy_textrank_similarity_matrix(x, min_sim, batch_size=1000)
    return xt

def vectorize_sents(sents, tokenize, vocab_to_idx):
    rows, cols, data = [], [], []
    for i, sent in enumerate(sents):
        counter = Counter(tokenize(sent))
        for token, count in counter.items():
            j = vocab_to_idx.get(token, -1)
            if j == -1:
                continue
            rows.append(i)
            cols.append(j)
            data.append(count)
    n_rows = len(sents)
    n_cols = len(vocab_to_idx)
    return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))


def numpy_textrank_similarity_matrix(x, min_sim=0.3, min_length=1, batch_size=1000):
    n_rows, n_cols = x.shape

    # Boolean matrix
    rows, cols = x.nonzero()
    data = np.ones(rows.shape[0])
    z = csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    # Inverse sentence length
    size = np.asarray(x.sum(axis=1)).reshape(-1)
    size[np.where(size <= min_length)] = 10000
    size = np.log(size)

    mat = []
    for bidx in range(math.ceil(n_rows / batch_size)):

        # slicing
        b = int(bidx * batch_size)
        e = min(n_rows, int((bidx+1) * batch_size))

        # dot product
        inner = z[b:e,:] * z.transpose()

        # sentence len[i,j] = size[i] + size[j]
        norm = size[b:e].reshape(-1,1) + size.reshape(1,-1)
        norm = norm ** (-1)
        norm[np.where(norm == np.inf)] = 0

        # normalize
        sim = inner.multiply(norm).tocsr()
        rows, cols = (sim >= min_sim).nonzero()
        data = np.asarray(sim[rows, cols]).reshape(-1)

        # append
        mat.append(csr_matrix((data, (rows, cols)), shape=(e-b, n_rows)))

    mat = sp.sparse.vstack(mat)
    return mat

def pagerank(x, df=0.85, max_iter=30, bias=None):
    
    assert 0 < df < 1

    # initialize
    A = normalize(x, axis=0, norm='l1')
    R = np.ones(A.shape[0]).reshape(-1,1)

    # check bias
    if bias is None:
        bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)
    else:
        bias = bias.reshape(-1,1)
        bias = A.shape[0] * bias / bias.sum()
        assert bias.shape[0] == A.shape[0]
        bias = (1 - df) * bias

    # iteration
    for _ in range(max_iter):
        R = df * (A * R) + bias

    return R


class KeysentenceSummarizer:
   
    def __init__(self, sents=None, tokenize=None, min_count=2,
        min_sim=0.3, vocab_to_idx=None,
        df=0.85, max_iter=30):

        self.tokenize = tokenize
        self.min_count = min_count
        self.min_sim = min_sim
        self.vocab_to_idx = vocab_to_idx
        self.df = df
        self.max_iter = max_iter

        if sents is not None:
            self.train_textrank(sents)

    def train_textrank(self, sents, bias=None):
        
        g = sent_graph(sents, self.tokenize, self.min_count,
            self.min_sim,  self.vocab_to_idx)
        self.R = pagerank(g, self.df, self.max_iter, bias).reshape(-1)
    
    def summarize(self, sents, topk=30, bias=None):
        
        n_sents = len(sents)
        self.train_textrank(sents, bias)
        idxs = self.R.argsort()[-topk:]
        keysents = [(idx, self.R[idx], sents[idx]) for idx in reversed(idxs)]
        return keysents

news3 =""" 브라질, 인도발 코로나 변이 상륙할라…긴장 속 모니터링 강화 (상파울루=연합뉴스) 김재순 특파원 = 브라질 보건당국이 인도발 신종 코로나바이러스 감염증(코로나19) 변이 바이러스 상륙 가능성에 긴장하고 있다.18일(현지시간) 브라질 언론에 따르면 상파울루시 당국은 인도발 변이 바이러스가 브라질에서도 나타날 가능성이 크다는 지적에 따라 상파울루주 정부 산하 부탄탕연구소와 함께 모니터링을 강화하기로 했다고 밝혔다.상파울루시는 브라질에서 코로나19 확진자와 사망자가 가장 먼저 나온 곳으로, 인근 과룰류스 국제공항을 통해 외국인 입국이 대규모로 이뤄지고 있다.상파울루시 관계자는 이날 기자회견을 통해 "인도발 변이 바이러스 상륙을 막기 위한 사전 조치로 부탄탕연구소와 협력해 외국인에 대한 철저한 추적 관찰이 이뤄질 것"이라고 말했다."""
#즉, 입력값 str
text_list = kss.split_sentences(news3)
summarizer = KeysentenceSummarizer(tokenize = lambda x:x.split(),min_sim = 0.3)
keysents = summarizer.summarize(text_list, topk=3)
print()
for _, _, text_list in keysents:
    print(text_list)