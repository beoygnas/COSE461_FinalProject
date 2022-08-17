# 가사-제목 예측모델 설계 및 구현을 통한 text summarization model 이해
 

## Introduction 

노래의 제목은 곡에 대한 정보와 가사를 함축적으로 표현할 수 있어야 하고, 이런 제목과 가사의 연관성이 NLP task 중 하나인 text summarization을 통해 설명을 될 수 있다는 직관에서 시작한 프로젝트이다. 프로젝트를 통해 이루고자 했던 목표는 다음과 같다.

1. 가사-제목 예측 모델의 구현과 학습을 통한, 결과 확인 및 모델 간 성능의 비교 
* Baseline : LSTM 기반의 seq2seq 모델
* 비교모델 : transformer 기반의 pretained T5



2. NLP summarization task의 pipeline 이해 및 경험.
* 데이터 수집, 전처리, 모델 빌드, training/evaulate 등의 과정을 코드로 작성, 이해

<br>

## Related Works
1. Neural Abstractive Text Summarization with Sequence-to-Sequence Models
2. Text Summarization with Pretrained Encoders
3. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer 


<br>

## Approach 
1. Abstractive summarization
* 주어진 가사로부터 단순히 반복되는 문장을 뽑아내는 extractive summarizaion 보다는, 유의미한 추론을 통해 예측을 하는 abstractive summarization 모델이 적합하다고 판단.

2. Baseline model : LSTM 기반의 seq2seq model

* 고전적인 모델인만큼 구현하기도 쉽고, 성능에서의 차이 또한 유의미할 것이라고 판단.
* 3층의 LSTM layer를 쌓은 encoder와, LSTM layer decoder로 구성 확실한 비교를 위해 Attention layer는 제외하기로 함.


3. 비교모델 : T5

* Text-to-text framework 로 summarization task에 대해 더 적합
* self attention layer
* 학습 환경을 고려하여, pretrained T5-base 모델을 finetuning huggingface library 이용하여 구현

<br>

## Experiments 

#### 1. Data

* Music Dataset : 1950 to 2019 (Kaggle)
* 1950년부터 2019년까지의 빌보드 연간차트 top100에 대한 dataset 약 28000 쌍의 (가사 – 제목) 
* 축약어, 불용어 등에 대한 처리 및 중복되는 행 제거. 약 28000개 -> 23689개
* 문장 길이에 따른 빈도수를 토대로 모델의 input/output 최대 길이 조정.

#### 2. Evaluation method
* ROUGE score 
    * 자연어처리 분야 모델 전반의 성능 평가에 사용
    * 노래 제목의 길이는 대부분 10자 이내이며, 각 단어가 유기적으로 연관이 있기보다는 노래를 대표하는 몇 개의 단어로 이루어져 있는 경우가 대부분이기에 ROUGE-1 score를 사용

* 단어가 일치하진 않아도, 같은 맥락을 표현할 수 있기 때문에 rouge score가 절대적인 기준이 될 수 없음. 따라서, testset에서 일부분을 추출하여 예측한 제목과 원래 제목에 대하여 정성적으로 평가함. 

#### 3. Experimental details

* LSTM 기반 seq2seq 모델

        Model : 3개의 LSTM layer encoder + 1 LSTM layer decoder
        Learning rate : 0.001
        Optimizer : rmsdrop optimizer
        Loss function : sparse categorical cross entrophy
        Epoch : 50
        Batch size : 256
        총 학습 시간 : 1시간 42분 32초
        학습환경 : google colab
* T5 

        Model : pretained T5-base
        Epoch : 10
        Batch size : 16
        Learning rate : 1e-4
        총 학습 시간 : 4시간 3분 57초
        학습 환경 : google colab


#### 4. result

* LSTM 기반 seq2seq 모델(좌), T5(우)

<p>
    <img src="https://i.imgur.com/rdUCgNY.png" height="49%" width="49%">
    <img src="https://i.imgur.com/QCDQbkD.png" height="49%" width="49%">
    <br><br>
</p> 

## Analysis

#### 1. LSTM 기반 seq2seq 모델

- 모든 prediction이 ‘love’라는 한 단어
- 무의미한 Rouge 점수
- overfitting!
<p>
    <img src="https://i.imgur.com/M8hDuJw.png" height="49%" width="49%">
</p>



#### 2. LSTM 기반 seq2seq 모델 분석

- 원시 모델
    - LSTM 기반의 seq2seq은 지금은 잘 사용되지 않는 고전적인 모델임. 
    - 모델의 성능을 월등히 높여줄 수 있는 attention layer나 pretrained 같은 방법의 부재

- 적은 학습량 
    - epoch 10 / 50 / 100으로 세팅 후 진행한 모든 prediction 의 결과가 같았음.
    -  학습 환경과 시간의 제약으로 더 많은 epoch에서 시도를 해보진 못했지만, 적은 학습량이 좋지 않은 결과에 영향을 미쳤을 수 있음.

- 데이터셋의 특징 
    -  학습에 사용했던 데이터셋은 빌보트 차트의 가요에 대한 것으로, 가요 특성 상 ‘사랑’이라는 주제를 가진 노래가 많을 것이며 이를 반영하여 학습과정에서 love라는 단어의 토큰에 대해 overfitting 되었을 가능성도 존재함.

#### 3. T5

- LSTM 보다 월등히 좋은 결과
    - Transformer 기반 (selft-attention 및 feed-forward network)
    - pretrain/finetuning으로 부족한 학습량 보완

- ROGUE 점수 이상의 학습결과
    - 글자 그대로 비슷한 prediction과 겹치는 글자는 없지만 의미가 유사한 prediction이 공존함.

<p>
    <img src="https://i.imgur.com/jROKvgB.png" height="75%" width="75%">
</p>
    


## Conclusion

#### 1. 프로젝트 한계 
- 데이터 부족
- 구현 및 학습환경의 한계


##### 2. 프로젝트 의의 

- 부족한 환경에서도 text summarization pipeline 의 구현 및 설계, 결과 확인.
- 고전적인 모델부터 비교적 최신 모델을 사용하여 비교 분석 할 수 있었음. 
- 자연어처리 및 머신러닝 task에 대한 insight를 가질 수 있었음.

