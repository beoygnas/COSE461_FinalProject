# cose461_finalproject

자연어처리 프로젝트

방향을 계속 고민중
1. extractive vs abstractive 
2. feature를 추가한 extractive로 갈지
3. kobertsum과의 비교? (abs가 없음)

baseline model : bertsumext
git주소 : https://github.com/nlpyang/PreSumm/blob/master/README.md

**5/16(월)**
1. 모델 확정 - bertsum (extractive 먼저)
2. 파이썬, 쿠다, pytorch 버젼 확인
    1. conda clean -all
3. 예제 학습 + test mode로 한번해보기? -> x
4. 자바를 설치해야지, preprocessing이 될ㅇ듯 맥으로 해볼예정
5. 

**5/17(화) **
	0. 자바 설치 -> corenlp 가능해짐
    1. 데이터 preprocessing 해보기
    2. 예제 트레이닝, evalutate 해보기 -> gpu 필요
    3. 아직 잘 모르겠는거.
		pretained model를 어디서?
		- 모델을 비워놔야 bert모델을 다운받는 듯


**5/18 (수)**
  1. git add/push/pull/commit 

자바 classpath 설정
export CLASSPATH=/Users/sangyeob/ku2022-1/nlp/finalproject/bertsum/PreSumm-master/stanford-corenlp-4.4.0/stanford-corenlp-4.4.0.jar

Data  Preparation 

step3. sentence splitting and tokenization
python ./src/preprocess.py -mode tokenize -raw_path ./cnn/stories -save_path ./stories_tokenized
	- 463.9 seconds
￼

step4. simpler json files
python ./src/preprocess.py -mode format_to_lines -raw_path ./stories_tokenized -save_path ./json_data/simple -n_cpus 1 -use_bert_basic_tokenizer false -map_path ./urls
	- 거의 4~5분 걸림.
	- 따로 출력 메세지는 없고, json_data에 똑같이 생성
	- 45개로 줄여줌.

step5. format to pytorch files
python ./src/preprocess.py -mode format_to_bert -raw_path ./json_data -save_path ./bert_data  -lower -n_cpus 1 -log_file ./logs/preprocess.log
	- 시작 : 05:14
	- 껏다 킴 : 05:25 정도


Model Training

model path
./models/bertext_cnndm_transformer.pt

command
python ./src/train.py -task ext -mode train -bert_data_path ./bert_data -ext_dropout 0.1 -model_path ./models/practice.pt -lr 2e-3 -visible_gpus 0,1,2 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -train_steps 50000 -accum_count 2 -log_file ./logs/ext_bert_cnndm -use_interval true -warmup_steps 10000 -max_pos 512 -visible_gpus -1


데이터과학 끝나고, 정리
내 데이터셋 정리하기
	1) extractive
	2) abstractive
	로 일단 분류  -> python


