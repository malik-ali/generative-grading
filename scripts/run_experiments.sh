NUM_PARALLEL=10
NUM_EXPS=40
JOBS_PER_GPU=$((NUM_EXPS / NUM_PARALLEL))
# CONFIG='postagging.json'
CONFIG='autoregrnn.json'

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

echo 'Running experiment for' ${CONFIG}
cat configs/${CONFIG}

for i in $(seq 1 $NUM_PARALLEL)
    do
        echo Submitting job: $i
        python scripts/mass_experiments.py configs/$CONFIG $JOBS_PER_GPU > /dev/null 2>&1 &
        # python scripts/mass_experiments.py configs/$CONFIG $JOBS_PER_GPU &
        sleep 1
    done
echo 'Waiting for processes to complete...'
cat configs/${CONFIG}
echo 
wait
echo 'All child processes compelted'
