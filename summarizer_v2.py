# -*- coding: cp949 -*- 

import platform
from collections import Counter
import kss
import numpy as np
from functools import partial
import numpy as np
from sklearn.preprocessing import normalize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances

# tokenizer import
from konlpy.tag import Okt, Komoran, Hannanum, Kkma

if platform.system() == "Windows":
    try:
        from eunjeon import Mecab
    except:
        print("please install eunjeon module")
else:  # Ubuntu일 경우
    from konlpy.tag import Mecab

from typing import List, Callable, Union, Any, TypeVar, Tuple, Dict


def get_tokenizer(tokenizer_name):
    if tokenizer_name == "komoran":
        tokenizer = Komoran()
    elif tokenizer_name == "okt":
        tokenizer = Okt()
    elif tokenizer_name == "mecab":
        tokenizer = Mecab()
    elif tokenizer_name == "hannanum":
        tokenizer = Hannanum()
    elif tokenizer_name == "kkma":
        tokenizer = Kkma()
    else:
        tokenizer = Mecab()
    return tokenizer

def get_tokens(sent: List[str], noun=False, tokenizer="mecab") -> List[str]:
    tokenizer = get_tokenizer(tokenizer)

    if noun:
        nouns = tokenizer.nouns(sent)
        nouns = [word for word in nouns if len(word) > 1]
        return nouns

    return tokenizer.morphs(sent)


def vectorize_sents(
    sents: List[str],
    stopwords=None,
    min_count=2,
    tokenizer="mecab",
    noun=False,
    mode="tfidf",
):

    if mode == "tfidf":
        vectorizer = TfidfVectorizer(
            stop_words=stopwords,
            tokenizer=partial(get_tokens, noun=noun, tokenizer="mecab"),
            min_df=min_count,
        )
    else:
        vectorizer = CountVectorizer(
            stop_words=stopwords,
            tokenizer=partial(get_tokens, noun=noun, tokenizer="mecab"),
            min_df=min_count,
        )

    vec = vectorizer.fit_transform(sents)
    vocab_idx = vectorizer.vocabulary_
    idx_vocab = {idx: vocab for vocab, idx in vocab_idx.items()}
    return vec, vocab_idx, idx_vocab

    
def similarity_matrix(x, min_sim=0.3, min_length=1):

    # binary csr_matrix
    numerators = (x > 0) * 1

    # denominator
    min_length = 1
    denominators = np.asarray(x.sum(axis=1))
    denominators[np.where(denominators <= min_length)] = 10000
    denominators = np.log(denominators)
    denom_log1 = np.matmul(denominators, np.ones(denominators.shape).T)
    denom_log2 = np.matmul(np.ones(denominators.shape), denominators.T)

    sim_mat = np.dot(numerators, numerators.T)
    sim_mat = sim_mat / (denom_log1 + denom_log2)
    sim_mat[np.where(sim_mat <= min_sim)] = 0

    return sim_mat


def cosine_similarity_matrix(x, min_sim=0.3):
    sim_mat = 1 - pairwise_distances(x, metric="cosine")
    sim_mat[np.where(sim_mat <= min_sim)] = 0

    return sim_mat


def sent_graph(
    sents: List[str],
    min_count=2,
    min_sim=0.3,
    tokenizer="mecab",
    noun=False,
    similarity=None,
    stopwords: List[str] = ["뉴스", "그리고"],
):

    mat, vocab_idx, idx_vocab = vectorize_sents(
        sents, stopwords, min_count=min_count, tokenizer=tokenizer
    )

    if similarity == "cosine":
        mat = cosine_similarity_matrix(mat, min_sim=min_sim)
    else:
        mat = similarity_matrix(mat, min_sim=min_sim)

    return mat, vocab_idx, idx_vocab


def pagerank(mat: np.ndarray, df=0.85, max_iter=50):
    
    assert 0 < df < 1

    A = normalize(mat, axis=0, norm="l1")
    N = np.ones(A.shape[0]) / A.shape[0]

    R = np.ones(A.shape[0])
    # iteration
    for _ in range(max_iter):
        R = df * np.matmul(A, R) + (1 - df) * N


    return R

#즉, 입력값 str 형태
news3 =""" 브라질, 인도발 코로나 변이 상륙할라…긴장 속 모니터링 강화 (상파울루=연합뉴스) 김재순 특파원 = 브라질 보건당국이 인도발 신종 코로나바이러스 감염증(코로나19) 변이 바이러스 상륙 가능성에 긴장하고 있다.18일(현지시간) 브라질 언론에 따르면 상파울루시 당국은 인도발 변이 바이러스가 브라질에서도 나타날 가능성이 크다는 지적에 따라 상파울루주 정부 산하 부탄탕연구소와 함께 모니터링을 강화하기로 했다고 밝혔다.상파울루시는 브라질에서 코로나19 확진자와 사망자가 가장 먼저 나온 곳으로, 인근 과룰류스 국제공항을 통해 외국인 입국이 대규모로 이뤄지고 있다.상파울루시 관계자는 이날 기자회견을 통해 "인도발 변이 바이러스 상륙을 막기 위한 사전 조치로 부탄탕연구소와 협력해 외국인에 대한 철저한 추적 관찰이 이뤄질 것"이라고 말했다."""
sents= kss.split_sentences(news3)
stopwords = ["연합뉴스", "가방"]
mat, vocab_idx, idx_vocab = sent_graph(sents, stopwords=stopwords, similarity="sim")
R = pagerank(mat)
topk = 3
idxs = R.argsort()[-topk:]
#keysents = [(idx, R[idx], sents[idx]) for idx in sorted(idxs)]
for idx in sorted(idxs):
    print(sents[idx])
