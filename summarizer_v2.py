
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
else:  # Ubuntu일 경우
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

    #문장간 유사도 계산, 문장간 유사도가 0.3이하면 간선 연결하지 않음.
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
    stopwords: List[str] = ["연합뉴스", "중앙일보","한겨레","국민일보","머니투데이","동아일보"]
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

news3 = """한강에서 사망한 채 발견된 손정민씨의 실종 당일 함께 술을 마셨던 A씨 측이 대규모 고소를 예고했다. 여기에 사건에 대한 진실 규명을 요구하는 단체가 경찰과 A씨의 휴대전화를 발견한 환경미화원을 검찰에 고발하면서 이른바 ‘한강 대학생 사건’이 고소·고발전으로 번지고 있다. 해당 단체는 추가 고발까지 예고했다.A씨 측 반격 "수만명 고소해야" 손씨의 사망 원인을 밝히기 위한 경찰 수사가 마무리 단계에 접어들었지만 고소·고발에 집회까지 이어지면서 혼란이 가중되고 있다. 5일 A씨 측 등에 따르면 정병원 변호사(법무법인 원앤파트너스)는 A씨와 그 가족 등에 대한 허위사실을 제기한 유튜버와 블로거 등을 7일부터 경찰에 고소할 예정이다.A씨를 대리하는 정 변호사의 고소 대상에는 유튜버와 블로거뿐 아니라 인터넷 커뮤니티 등에 게시글이나 댓글을 작성한 이들도 포함될 예정이다. 정 변호사는 “수차례 친구 A 및 그 가족과 주변인들에 관한 위법행위를 멈춰달라고 요청했음에도 게시물이 오히려 늘어나고 있다”며 “일부 내용은 수인한도를 넘어서면서 피해와 고통은 점점 더 심해지고 있다”고 밝혔다. 이어 “선처를 희망하는 사람이 전혀 없다면 최소 수만 명은 고소해야 할 것 같다”고 덧붙였다.이들은 지난달 29일에도 집회를 열고 손씨의 실종 당일 목격자의 제보를 독려하는 집회를 열어다. 지난 1일에는 서울 종로구 서울경찰청 앞에서 기자회견을 열고 “수사당국이 경찰 초동수사 부실 논란과 손씨 사망 경위에 대한 의혹을 피해왔다”고 주장했다. 반진사를 만든 건 유튜브 ‘종이의 TV’ 채널 운영자다. A씨 측이 명예훼손 등 혐의로 고소하겠다고 밝힌 유튜버 중엔 '종이의 TV‘도 포함돼 있다.손정민 친구 "수만명 고소"에 "끝까지 간다"…혼란의 한강 사망"""
#즉, 입력값 str 형태

sents= kss.split_sentences(news3)
mat, vocab_idx, idx_vocab = sent_graph(sents)
R = pagerank(mat)
topk = 3
idxs = R.argsort()[-topk:]
#keysents = [(idx, R[idx], sents[idx]) for idx in sorted(idxs)]
for idx in sorted(idxs):
    print(sents[idx])
