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
        self.stopwords = ['����' ,'��ŭ', '��������', '������', "���մ���", "���ϸ�", "�����Ϻ�", "�߾��Ϻ�", "�����Ϻ�", "����"
        ,"��", "��", "���̱�", "������", "���̰�", "��", "��", "�츮", "����", "����", "����", "��", "��", "��", "��", "��",]
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
            A[id, id] = 0 # diagonal �κ��� 0���� 
            link_sum = np.sum(A[:,id]) # A[:, id] = A[:][id]
            if link_sum != 0:
                A[:, id] /= link_sum
            A[:, id] *= -d
            A[id, id] = 1
            
        B = (1-d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B) # ���������� Ax = b
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
        

#��, �Է°� str����
news3 =""" �����, �ε��� �ڷγ� ���� ����Ҷ󡦱��� �� ����͸� ��ȭ (���Ŀ��=���մ���) ����� Ư�Ŀ� = ����� ���Ǵ籹�� �ε��� ���� �ڷγ����̷��� ������(�ڷγ�19) ���� ���̷��� ��� ���ɼ��� �����ϰ� �ִ�.18��(�����ð�) ����� ��п� ������ ���Ŀ��� �籹�� �ε��� ���� ���̷����� ����������� ��Ÿ�� ���ɼ��� ũ�ٴ� ������ ���� ���Ŀ���� ���� ���� ��ź�������ҿ� �Բ� ����͸��� ��ȭ�ϱ�� �ߴٰ� ������.���Ŀ��ô� ��������� �ڷγ�19 Ȯ���ڿ� ����ڰ� ���� ���� ���� ������, �α� ������� ���������� ���� �ܱ��� �Ա��� ��Ը�� �̷����� �ִ�.���Ŀ��� �����ڴ� �̳� ����ȸ���� ���� "�ε��� ���� ���̷��� ����� ���� ���� ���� ��ġ�� ��ź�������ҿ� ������ �ܱ��ο� ���� ö���� ���� ������ �̷��� ��"�̶�� ���ߴ�."""

textrank = TextRank(news3)
for row in textrank.summarize(3):
    print(row)
    print()
