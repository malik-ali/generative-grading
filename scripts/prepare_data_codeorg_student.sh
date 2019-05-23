#!/bin/bash

# RUN ME FROM REPO ROOT...
PROBLEM=codeorg9_100k;
BASE_DIR=/mnt/fs5/$USER/generative-grading/data/real/education;

cd src/rubricsampling;
python makeRawCodeOrgStudentData.py $BASE_DIR/$PROBLEM/raw/ $PROBLEM;

cd ../..;
python -m scripts.anonymize_data $BASE_DIR/$PROBLEM/raw/ $PROBLEM;

# --account-for-counts will make duplicates for programs that appear many times 
python -m scripts.process_codeorg_student_data $PROBLEM  # --account-for-counts;
