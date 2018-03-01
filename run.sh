RESULT_HOME=/home/wil/workspace/buflightdev/projects/neurocontroller/att-sitl/results/baselines/ddpg-abs_pid_penalty-window_size2-2
export OPENAI_LOGDIR=$RESULT_HOME/logs
python3 -m baselines.ddpg.main --env-id="flightcontrol-v0" \
 --ckpt-dir=$RESULT_HOME/checkpoints \
 --nb-epochs=5000 \
 --progress-dir=$RESULT_HOME/model-progress
 #--nb-epoch-cycles=2 \
 #--nb-rollout-steps=3 \
