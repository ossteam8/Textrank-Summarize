from typing import List, Callable, Union, Any, TypeVar, Tuple, Dict
from functools import partial
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import numpy as np

class TextRank:
    
    def vectorize_sents(corp_doc_topic = corp_doc_topic, min_count=2, tokenizer="mecab", noun=True):
        vectorizer = CountVectorizer(tokenizer=lambda x: x,lowercase=False)
        vec = vectorizer.fit_transform(corp_doc_topic)
        vocab_idx = vectorizer.vocabulary_
        idx_vocab = {idx: vocab for vocab, idx in vocab_idx.items()}

        return vec, vocab_idx, idx_vocab

    def word_similarity_matrix(x, min_sim=0.3):
        #문장 간 유사도가 0.3 보다 작은 경우에는 edge 를 연결지 않으면서 단어 그래프 생성.
        sim_mat = 1 - pairwise_distances(x.T, metric="cosine")
        sim_mat[np.where(sim_mat <= min_sim)] = 0

        return sim_mat

    
    def word_graph(corp_doc_topic = corp_doc_topic,min_count=5,min_sim=0.3,tokenizer="mecab",noun=True,):

        mat, vocab_idx, idx_vocab = vectorize_sents(
            corp_doc_topic,min_count=min_count, tokenizer=tokenizer, noun=noun
        )

        mat = word_similarity_matrix(mat, min_sim=min_sim)
        return mat, vocab_idx, idx_vocab
    
    
    def pagerank(x: np.ndarray, df=0.85, max_iter=50):

        assert 0 < df < 1

        A = normalize(mat, axis=0, norm="l1")
        N = np.ones(A.shape[0]) / A.shape[0]

        R = np.ones(A.shape[0])
        for _ in range(max_iter):
            R = df * np.matmul(A, R) + (1 - df) * N
        

        return R