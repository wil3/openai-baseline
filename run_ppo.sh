
BASELINE_DIR=$(pwd)
GYM_HOME=/home/wil/workspace/buflightdev/projects/gym-flightcontrol
cd $GYM_HOME
COMMIT=$(git describe --always)
ENV=attitude-episodic-3-v0
DIR_NAME=ALG=ppo-ENV=${COMMIT}_${ENV}
RESULT_HOME=/home/wil/workspace/buflightdev/projects/results/${DIR_NAME}
export OPENAI_LOGDIR=$RESULT_HOME/logs
cd $BASELINE_DIR

python3 -m baselines.ppo1.run_flightcontrol \
 --env-id=$ENV \
 --ckpt-dir=$RESULT_HOME/checkpoints \
 --flight-log-dir=$RESULT_HOME/model-progress \
 --num-timesteps=1000000

spd-say "Your results are ready for review"
