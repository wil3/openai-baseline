#!/usr/bin/env python3
# noinspection PyUnresolvedReferences

import gym
from neuroflight_trainer.neuroflight_arxiv_trainer import GyroErrorGyroDeltaErrorEnv
from baselines.common.fc_learning_utils import FlightLog
import argparse
from mpi4py import MPI
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.common import set_global_seeds

def train(env, num_timesteps, seed, flight_log_dir, ckpt_dir, render, restore_dir,ckpt_freq, 
          optim_stepsize=3e-4, schedule="linear", gamma=0.99, optim_epochs=10, optim_batchsize=64, horizon=2048):
    from baselines.ppo1 import pposgd_simple
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    logger.set_level(logger.DISABLED)
    workerseed = seed + 1000000 * rank
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=32, num_hid_layers=2)
    flight_log = None
    # FIXME
    #if flight_log_dir:
    #    flight_log = FlightLog(flight_log_dir)
    if render:
        env.render()
    env.seed(workerseed)
    set_global_seeds(workerseed)
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=horizon,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=optim_epochs, optim_stepsize=optim_stepsize, optim_batchsize=optim_batchsize,
            gamma=0.99, lam=0.95, schedule=schedule,
            flight_log = flight_log,
            ckpt_dir = ckpt_dir,
            restore_dir = restore_dir,
            save_timestep_period= ckpt_freq
            )
    env.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_sdf', type=str)
    parser.add_argument('--num-timesteps', type=int, default=1e7)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--flight-log-dir', type=str, default=None)
    parser.add_argument('--ckpt-dir', type=str, default=None)
    parser.add_argument('--restore-dir', help="If we should restore a graph", type=str, default=None)
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--ckpt-freq', help='Timestep frequency checkpoints are made', type=int, default=1000)

    # Hyper parameters
    parser.add_argument('--stepsize', type=float, default=3e-4)
    parser.add_argument('--schedule', type=str, default="linear", help="One of constant, linear")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--horizon', type=int, default=2048)

    args = parser.parse_args()


    env = GyroErrorGyroDeltaErrorEnv(omega_bounds=[-300,300],
        max_sim_time=30.,
        noisefree=False,
        sigma=5,
        command_time=0.,
        command_time_off=[0.25, 1.0],
        command_time_on=[0.25, 1.0],
        aircraft_config=args.model_sdf
                                     )

    train(env, args.num_timesteps, args.seed, args.flight_log_dir,
          args.ckpt_dir,  args.render, args.restore_dir, args.ckpt_freq, schedule=args.schedule, optim_stepsize=args.stepsize, horizon=args.horizon, optim_batchsize=args.batchsize)

