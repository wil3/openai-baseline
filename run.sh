RESULT_HOME=/home/wil/workspace/buflightdev/projects/neurocontroller/att-sitl/results/baselines/ddpg
export OPENAI_LOGDIR=$RESULT_HOME/logs
python3 -m baselines.ddpg.main --env-id="flightcontrol-v0" \
 --ckpt-dir=$RESULT_HOME/checkpoints \
 --nb-epochs=10 \
 --nb-epoch-cycles=2 \
 --nb-rollout-steps=3 \
 --progress-dir=$RESULT_HOME/model-progress
