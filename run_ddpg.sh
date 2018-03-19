BASELINE_DIR=$(pwd)
GYM_HOME=/home/wil/workspace/buflightdev/projects/gym-flightcontrol
cd $GYM_HOME
COMMIT=$(git describe --always)
ENV=attitude-zero-start-v0
DIR_NAME=ALG=ddpg-IN=error_3x-ENV=${COMMIT}_${ENV}-REWARD=euclidean_gausian-end_on_nomonotonic_outofbounds
RESULT_HOME=/home/wil/workspace/buflightdev/projects/neurocontroller/att-sitl/results/baselines/${DIR_NAME}

export OPENAI_LOGDIR=$RESULT_HOME/logs

cd $BASELINE_DIR
python3 -m baselines.ddpg.main --env-id=$ENV \
 --ckpt-dir=$RESULT_HOME/checkpoints \
 --nb-epochs=5000 \
 --progress-dir=$RESULT_HOME/model-progress

# --noise-type=ou_0.3 \
