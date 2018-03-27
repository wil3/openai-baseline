
BASELINE_DIR=$(pwd)
GYM_HOME=/home/wil/workspace/buflightdev/projects/gym-flightcontrol
cd $GYM_HOME
COMMIT=$(git describe --always)
ENV=attitude-episodic-goal-5-v0
DIR_NAME=ALG=her-ENV=${COMMIT}_${ENV}
RESULT_HOME=/home/wil/workspace/buflightdev/projects/neurocontroller/att-sitl/results/baselines/${DIR_NAME}
export OPENAI_LOGDIR=$RESULT_HOME/logs
cd $BASELINE_DIR

python3 -m baselines.her.experiment.train\
 --env_name=$ENV \
 --logdir=$RESULT_HOME \
 --num_cpu=1
