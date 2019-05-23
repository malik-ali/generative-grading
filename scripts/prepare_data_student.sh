#!/bin/bash

# RUN ME FROM REPO ROOT...
PROBLEM=liftoff;
BASE_DIR=/mnt/fs5/$USER/generative-grading/data/real/education

python -m scripts.anonymize_data $BASE_DIR/$PROBLEM/raw/ $PROBLEM;
python -m scripts.process_student_data $PROBLEM;