#!/bin/bash

# RUN ME FROM REPO ROOT...

NUM_SAMPLES=100000;

PROBLEM=citizenship13;
PROBLEM_DIRNAME=${PROBLEM}_$((NUM_SAMPLES / 1000))k;
BASE_DIR=/mnt/fs5/$USER/generative-grading/data


for STRATEGY in standard uniform tempered 
do
	cd src/rubricsampling;
	python makeRawData.py $BASE_DIR/raw/$PROBLEM_DIRNAME/$STRATEGY/ $PROBLEM --sample-strategy $STRATEGY --num-samples $NUM_SAMPLES;
	python makeTieredData.py $BASE_DIR/raw/$PROBLEM_DIRNAME/$STRATEGY/;
	python makeDataSplits.py $BASE_DIR/raw/$PROBLEM_DIRNAME/$STRATEGY/;

	cd ../..;
	for SPLIT in train val test
	do
		python -m scripts.anonymize_data $BASE_DIR/raw/$PROBLEM_DIRNAME/$STRATEGY/$SPLIT $PROBLEM;
	done;
done

cd src/rubricsampling;
python makeGrammarUniqueIds.py $BASE_DIR/raw/$PROBLEM_DIRNAME $PROBLEM;
cd ../..;
python -m scripts.process_vocab $PROBLEM_DIRNAME;  # we assume all STRATEGY folders exist

for STRATEGY in standard uniform tempered 
do
	python -m scripts.process_data $PROBLEM_DIRNAME --split train --sampling-strategy $STRATEGY;
	python -m scripts.process_data $PROBLEM_DIRNAME --split val --sampling-strategy $STRATEGY;
	python -m scripts.process_data $PROBLEM_DIRNAME --split test --sampling-strategy $STRATEGY;
done

