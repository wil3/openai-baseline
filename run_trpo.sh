
BASELINE_DIR=$(pwd)
GYM_HOME=/home/wil/workspace/buflightdev/projects/gym-flightcontrol
cd $GYM_HOME
COMMIT=$(git describe --always)
#ENV=attitude-zero-start-v0
#ENV=attitude-ind-axis-v0
#ENV=attitude-continuous-v0
#ENV=attitude-inc-axis-v0
ENV=attitude-progressive-v0
DIR_NAME=ALG=trpo-IN=error_4x-ENV=${COMMIT}_${ENV}-tol=0.05-health=100
RESULT_HOME=/home/wil/workspace/buflightdev/projects/neurocontroller/att-sitl/results/baselines/${DIR_NAME}
export OPENAI_LOGDIR=$RESULT_HOME/logs
cd $BASELINE_DIR

python3 -m baselines.trpo_mpi.run_flightcontrol \
 --env-id=$ENV \
 --ckpt-dir=$RESULT_HOME/checkpoints \
 --flight-log-dir=$RESULT_HOME/model-progress
