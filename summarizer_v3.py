# -*- coding: cp949 -*- 

from konlpy.tag import Kkma
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np
 

class SentenceTokenizer(object):
    def __init__(self):
        self.kkma = Kkma()
        self.twitter = Okt()
        self.stopwords = ['중인' ,'만큼', '마찬가지', '꼬집었', "연합뉴스", "데일리", "동아일보", "중앙일보", "조선일보", "기자"
        ,"아", "휴", "아이구", "아이쿠", "아이고", "어", "나", "우리", "저희", "따라", "의해", "을", "를", "에", "의", "가",]
    def text2sentences(self, text):
        sentences = self.kkma.sentences(text)
        for idx in range(0, len(sentences)):
            if len(sentences[idx]) <= 10:
                sentences[idx-1] += (' ' + sentences[idx])
                sentences[idx] = ''
        return sentences
    def get_nouns(self, sentences):
        nouns = []
        for sentence in sentences:
            if sentence is not '':
                nouns.append(' '.join([noun for noun in self.twitter.nouns(str(sentence))
                                       if noun not in self.stopwords and len(noun) > 1]))
        return nouns


class GraphMatrix(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.graph_sentence = []
        
    def build_sent_graph(self, sentence):
        tfidf_mat = self.tfidf.fit_transform(sentence).toarray()
        self.graph_sentence = np.dot(tfidf_mat, tfidf_mat.T)
        return  self.graph_sentence
 
class Rank(object):
    def get_ranks(self, graph, d=0.85): # d = damping factor
        A = graph
        matrix_size = A.shape[0]
        for id in range(matrix_size):
            A[id, id] = 0 # diagonal 부분을 0으로 
            link_sum = np.sum(A[:,id]) # A[:, id] = A[:][id]
            if link_sum != 0:
                A[:, id] /= link_sum
            A[:, id] *= -d
            A[id, id] = 1
            
        B = (1-d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B) # 연립방정식 Ax = b
        return {idx: r[0] for idx, r in enumerate(ranks)}

class TextRank(object):
    def __init__(self, text):
        self.sent_tokenize = SentenceTokenizer()
        
        self.sentences = self.sent_tokenize.text2sentences(text)
        
        self.nouns = self.sent_tokenize.get_nouns(self.sentences)
                
        self.graph_matrix = GraphMatrix()
        self.sent_graph = self.graph_matrix.build_sent_graph(self.nouns)        
        
        self.rank = Rank()
        self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)
        self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True)
    
        
    def summarize(self, sent_num=3):
        summary = []
        index=[]
        for idx in self.sorted_sent_rank_idx[:sent_num]:
            index.append(idx)
        
        index.sort()
        for idx in index:
            summary.append(self.sentences[idx])
        
        return summary
        

#즉, 입력값 str형태
news3 =""" 브라질, 인도발 코로나 변이 상륙할라…긴장 속 모니터링 강화 (상파울루=연합뉴스) 김재순 특파원 = 브라질 보건당국이 인도발 신종 코로나바이러스 감염증(코로나19) 변이 바이러스 상륙 가능성에 긴장하고 있다.18일(현지시간) 브라질 언론에 따르면 상파울루시 당국은 인도발 변이 바이러스가 브라질에서도 나타날 가능성이 크다는 지적에 따라 상파울루주 정부 산하 부탄탕연구소와 함께 모니터링을 강화하기로 했다고 밝혔다.상파울루시는 브라질에서 코로나19 확진자와 사망자가 가장 먼저 나온 곳으로, 인근 과룰류스 국제공항을 통해 외국인 입국이 대규모로 이뤄지고 있다.상파울루시 관계자는 이날 기자회견을 통해 "인도발 변이 바이러스 상륙을 막기 위한 사전 조치로 부탄탕연구소와 협력해 외국인에 대한 철저한 추적 관찰이 이뤄질 것"이라고 말했다."""

textrank = TextRank(news3)
for row in textrank.summarize(3):
    print(row)
    print()
