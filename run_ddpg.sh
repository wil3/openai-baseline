BASELINE_DIR=$(pwd)
GYM_HOME=/home/wil/workspace/buflightdev/projects/gym-flightcontrol
cd $GYM_HOME
COMMIT=$(git describe --always)


ENV=attitude-episodic-3-v0
DIR_NAME=ALG=ddpg-ENV=${COMMIT}_${ENV}
RESULT_HOME=/home/wil/workspace/buflightdev/projects/results/${DIR_NAME}

export OPENAI_LOGDIR=$RESULT_HOME/logs

cd $BASELINE_DIR
python3 -m baselines.ddpg.main --env-id=$ENV \
 --ckpt-dir=$RESULT_HOME/checkpoints \
 --flight-log-dir=$RESULT_HOME/model-progress

 #--nb-epochs=5000 \
# --noise-type=ou_0.3 \
spd-say "Your results are ready for review"
