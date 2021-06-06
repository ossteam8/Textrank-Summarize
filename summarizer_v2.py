
# -*- coding: cp949 -*- 
import platform
import kss
import numpy as np
from functools import partial
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

if platform.system() == "Windows":
    try:
        from eunjeon import Mecab
    except:
        print("please install eunjeon module")
else:  # Ubuntu�� ���
    from konlpy.tag import Mecab

from typing import List, Callable, Union, Any, TypeVar, Tuple, Dict


def get_tokenizer(tokenizer_name):
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
    noun=False
):

    vectorizer = TfidfVectorizer(
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

    #���尣 ���絵 ���, ���尣 ���絵�� 0.3���ϸ� ���� �������� ����.
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



def sent_graph(
    sents: List[str],
    min_count=2,
    min_sim=0.3,
    tokenizer="mecab",
    noun=False,
    stopwords: List[str] = ["���մ���", "�߾��Ϻ�","�Ѱܷ�","�����Ϻ�","�Ӵ�������","�����Ϻ�"]
):
    
    # TF-IDF + Cosine similarity 

    mat, vocab_idx, idx_vocab = vectorize_sents(
        sents, stopwords, min_count=min_count, tokenizer=tokenizer
    )

    
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

news3 = """�Ѱ����� ����� ä �߰ߵ� �����ξ��� ���� ���� �Բ� ���� ���̴� A�� ���� ��Ը� ��Ҹ� �����ߴ�. ���⿡ ��ǿ� ���� ���� �Ը��� �䱸�ϴ� ��ü�� ������ A���� �޴���ȭ�� �߰��� ȯ���ȭ���� ������ ����ϸ鼭 �̸��� ���Ѱ� ���л� ��ǡ��� ��ҡ���������� ������ �ִ�. �ش� ��ü�� �߰� ��߱��� �����ߴ�.A�� �� �ݰ� "������ ����ؾ�" �վ��� ��� ������ ������ ���� ���� ���簡 ������ �ܰ迡 ���������� ��ҡ���߿� ��ȸ���� �̾����鼭 ȥ���� ���ߵǰ� �ִ�. 5�� A�� �� � ������ ������ ��ȣ��(�������� ������Ʈ�ʽ�)�� A���� �� ���� � ���� ��������� ������ ��Ʃ���� ��ΰ� ���� 7�Ϻ��� ������ ����� �����̴�.A���� �븮�ϴ� �� ��ȣ���� ��� ��󿡴� ��Ʃ���� ��ΰŻ� �ƴ϶� ���ͳ� Ŀ�´�Ƽ � �Խñ��̳� ����� �ۼ��� �̵鵵 ���Ե� �����̴�. �� ��ȣ��� �������� ģ�� A �� �� ������ �ֺ��ε鿡 ���� ���������� ����޶�� ��û�������� �Խù��� ������ �þ�� �ִ١��� ���Ϻ� ������ �����ѵ��� �Ѿ�鼭 ���ؿ� ������ ���� �� �������� �ִ١��� ������. �̾� ����ó�� ����ϴ� ����� ���� ���ٸ� �ּ� ���� ���� ����ؾ� �� �� ���١��� ���ٿ���.�̵��� ������ 29�Ͽ��� ��ȸ�� ���� �վ��� ���� ���� ������� ������ �����ϴ� ��ȸ�� �����. ���� 1�Ͽ��� ���� ���α� �������û �տ��� ����ȸ���� ���� ������籹�� ���� �ʵ����� �ν� ����� �վ� ��� ������ ���� ��Ȥ�� ���ؿԴ١��� �����ߴ�. �����縦 ���� �� ��Ʃ�� �������� TV�� ä�� ��ڴ�. A�� ���� ���Ѽ� �� ���Ƿ� ����ϰڴٰ� ���� ��Ʃ�� �߿� '������ TV���� ���Ե� �ִ�.������ ģ�� "������ ���"�� "������ ����"��ȥ���� �Ѱ� ���"""
#��, �Է°� str ����

sents= kss.split_sentences(news3)
mat, vocab_idx, idx_vocab = sent_graph(sents)
R = pagerank(mat)
topk = 3
idxs = R.argsort()[-topk:]
#keysents = [(idx, R[idx], sents[idx]) for idx in sorted(idxs)]
for idx in sorted(idxs):
    print(sents[idx])
