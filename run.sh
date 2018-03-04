BASELINE_DIR=$(pwd)
GYM_HOME=/home/wil/workspace/buflightdev/projects/gym-flightcontrol
cd $GYM_HOME
COMMIT=$(git describe --always)
ENV=attitude-zero-start-v0
RESULT_HOME=/home/wil/workspace/buflightdev/projects/neurocontroller/att-sitl/results/baselines/ALG=ddpg-IN=error_x3-ENV=${COMMIT}_${ENV}-REWARD=euclidean_gaussian_penalty
export OPENAI_LOGDIR=$RESULT_HOME/logs

cd $BASELINE_DIR
python3 -m baselines.ddpg.main --env-id=$ENV \
 --ckpt-dir=$RESULT_HOME/checkpoints \
 --nb-epochs=5000 \
 --progress-dir=$RESULT_HOME/model-progress
