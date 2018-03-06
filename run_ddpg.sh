BASELINE_DIR=$(pwd)
GYM_HOME=/home/wil/workspace/buflightdev/projects/gym-flightcontrol
cd $GYM_HOME
COMMIT=$(git describe --always)
ENV=attitude-zero-start-v0
#DIR_NAME=ALG=ddpg-ou_0.3-IN=error_3x-ENV=${COMMIT}_${ENV}-REWARD=error-end_on_nomonotonic_outofbounds
DIR_NAME=ALG=ddpg-ou_0.3-IN=error_3x-ENV=1948a3d_attitude-zero-start-v0-REWARD=error-end_on_nomonotonic_outofbounds
#euclidean_gaussian_epsilon_alpha_penalty
RESULT_HOME=/home/wil/workspace/buflightdev/projects/neurocontroller/att-sitl/results/baselines/${DIR_NAME}

export OPENAI_LOGDIR=$RESULT_HOME/logs

cd $BASELINE_DIR
python3 -m baselines.ddpg.main --env-id=$ENV \
 --ckpt-dir=$RESULT_HOME/checkpoints \
 --noise-type=ou_0.3 \
 --nb-epochs=5000 \
 --progress-dir=$RESULT_HOME/model-progress
