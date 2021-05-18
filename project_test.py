# -*- coding: cp949 -*- 
import kss
from goose3 import Goose
from konlpy.tag import Komoran
from tensorflow.keras.preprocessing.text import Tokenizer

text = "옹호론 소수층 문명론 지배계급 제조업자 연극계 국민대학교 자연어처리 국민 대학교 한강 국정원 아이오아이 산토끼 영정도 잠실역 숭례문 문재인 채희선 김푸름 문 대통령은 이날 청와대 춘추관에서 이같이 말하며 “인수위 없이 임기를 시작하고 쉼 없이 달려왔지만, 임기 마치는 그날까지 앞만 보고 가야 하는 것이 우리 정부의 피할 수 없는 책무라고 생각한다”고 했다. 이어 “코로나 사태가 발생한 지 벌써 1년 3개월이 지났다. 이렇게 오래갈 줄 몰랐다. 이토록 인류의 삶을 송두리째 뒤흔들 줄 몰랐다”고 했다. 문 대통령은 “감염병과 방역 조치로 인한 고통, 막심한 경제적 피해와 실직, 경험해보지 못한 평범한 일상의 상실, 이루 헤아릴 수 없는 어려움을 겪고 계신 국민들께 깊은 위로의 말씀을 드린다”고 했다."

text2 = "신종 코로나바이러스 감염증(코로나19)이 크게 확산 중인 인도에서 사망한 줄 알았던 70대 여성 확진자가 화장 직전에 깨어났다고 인디아투데이 등 현지 매체가 15일(현지시간) 보도했다.  매체에 따르면 이날 인도 마하라슈트라주 푸네 지구 바라마티시 무드할레 마을에 거주하는 76세 여성 확진자가 코로나로 사망한 것으로 판단됐다가 화장 직전에 깨어났다.  코로나19 양성 판정을 받은 이 여성은 집안에서 격리됐다가 고령으로 상태가 점차 악화하자 인근 지역병원으로 옮겨졌다. 이 여성은 지난 10일 바라마티 병원으로 옮겨졌다. 그러나 가족들은 병상을 확보할 수 없었다. 결국 이 여성은 다른 병원으로 옮기려는 중 의식을 잃었다. 맥박과 의식이 없었다. 사망한 줄 안 가족들은 집으로 돌아와 장례식을 준비했다.  이 여성은 상여를 타고 화장터로 옮겨졌고 친척들은 애도의 눈물을 흘렸다. 이 여성이 불길 속으로 들어가기 직전 깜짝 놀랄 일이 일어났다. 갑자기 이 여성이 울기 시작하면서 눈을 뜬 것이다. 가족들은 충격에 휩싸였다. 가족들은 바로 이 여성을 병원으로 옮겼다. 이 여성은 현재 바라마티 실버 쥬빌레 병원에 입원해있다. 인도에서는 지난 3월부터 코로나19가 크게 재확산하고 있다. 인도의 일일 신규 확진자 수는 지난 7일 41만4188명으로 최고치를 기록한 후 조금씩 줄어 여전히 하루 30만명 안팎의 많은 감염자가 보고되고 있다. 일일 신규 사망자 수는 4000명대 안팎에서 줄지 않고 있다. 기록적인 수치가 이어지면서 의료시설의 수용 한계가 초과되고 있다. 장례가 어려울 정도로 많은 사망자가 발생하면서 최근 갠지스강에 코로나19 사망자들로 추측되는 시신이 수백 구 떠내려오는 사건도 발생했다. 이후 갠지스강 근처에는 경찰이 상주하며 확성기로 “강물에 시신을 버리지 말라”고 계도하고 있다."

text_list = kss.split_sentences(text)
komoran = Komoran()

j = []
for i in text_list :
    sentences = komoran.nouns(i)
    result = []
    #print('sentences')
    #print(sentences)
    for word in sentences :
        #print(word)
        if len(word) > 1 :
            result.append(word)
    #print('result')
    #print(result)
    j.append(result)

#print(len(j))
print(j, '\n')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(j)
word_index = tokenizer.word_index
print(word_index ,'\n')
word_counts = tokenizer.word_counts
print(word_counts,'\n')
#입력으로 들어온 코퍼스에 대해 각 단어를 이미 정해진 인덱싱으로 변환
print(tokenizer.texts_to_sequences(j) ,'\n' )


#빈도수가 높은 상위 n(5)개 단어만 사용한다고 지정,
# 실제 적용은 texts_to sequence 에서 부터 적용. 인덱싱과 카운트도 적용하려면 따로 처리
vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 1)
tokenizer.fit_on_texts(j)
print(tokenizer.texts_to_sequences(j) ,'\n')



