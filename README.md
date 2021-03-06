
이 repositoey는 preprocessing, textrank_keyword, textrank_summarize 를 개발하면서 사용된 **test repository** 이다.

**실제 개발이 완성된 코드들은 아래 두 repository에 적용되어 있다.**
**이후 업데이트나 추가적인 개발 또한 두 repository에서 실행됩니다.**

**Keyword** =>  https://github.com/ossteam8/LDA-TextRank-keyword 

**Completed repository** => https://github.com/ossteam8/K-news-keyword


아래 코드를 통해 textrank_ketword, summary, preprocessing 을 각각 실행시킬 수 있습니다.


[standalone code]

[textrank](https://github.com/ossteam8/Textrank-Summarize/blob/develop/textrank_keyword.py)

[summarizer_v2.py](https://github.com/ossteam8/Textrank-Summarize/blob/main/summarizer_v2.py)

[preprocessing.ipynb](https://github.com/ossteam8/Textrank-Summarize/blob/main/test_notebook/preprocessing.ipynb)

# Textrank_Summarize

TextRank 는 sentence graph를 구축한 뒤, Graph ranking algorithm인 PageRank를 이용하여 핵심 문장을 선택하고 

이들을 이용하여 주어진 문서 집합내에서 대표할 수 있는 문장들을 선택하여 요약(summarization)한다. 

자연어 처리에서 Textrank를 이용하여 요약하는 것을 extractive approaches라고 한다.

```
전처리  -> sent graph 생성 -> textrank 적용  => 결과(요약)
```

먼저 입력 입력형식은 "기사본문 + 기사제목" 으로 구성된 string이다. 이를 문장 단위로 분리하고, 명사만을 추출하며 전처리를 한다.

다음으로, Tf-idf cosine similarity를 사용하여 sent graph를 형성한다.

Tf-idf는 빈도수에 초점을 둔 것이 아닌 중요한 특징에 가중치를 부여한다.

따라서 tf-idf vectorize하여 각 문장은 아래의 그림과 같이 STM(Sentence Term Matrix)형태로 표현되며 각 문장의 단어는 tf-idf로 표현된다.

<img width="437" alt="tf-idf(stm)" src="https://user-images.githubusercontent.com/80442377/121154072-94ef9e00-c881-11eb-8d30-6d0cf9021b38.png">

이때, 문장 간 유사도를 측정하기 위하여 Cosine similarity 가 이용되는데, TextRank 는 아래와 같은 문장 간 유사도 척도를 제안했습니다. 

두 문장에 공통으로 등장한 단어의 개수를 각 문장의 단어 개수의 log 값의 합으로 나눈 것 이다.

<img width="309" alt="cosine" src="https://user-images.githubusercontent.com/80442377/121154131-a89b0480-c881-11eb-8649-efee1f71910e.png">

### PageRank 공식

<img width="310" alt="pagerank공식" src="https://user-images.githubusercontent.com/80442377/120992094-3a8d0980-c7bd-11eb-9ad8-1f957a45f8a9.png">

