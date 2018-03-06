#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
"""
from mpi4py import MPI
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi
from mpi4py import MPI
import argparse
import gym_flightcontrol
from baselines.common.fc_learning_utils import FlightLog

def train(env_id, num_timesteps, seed, flight_log_dir, ckpt_dir):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    rank = MPI.COMM_WORLD.Get_rank()
    env = gym.make(env_id)
    seed = seed + 1000000 * rank
    flight_log = FlightLog(flight_log_dir)
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
            flight_log = flight_log,
            ckpt_dir = ckpt_dir
        )
    env.close()
"""

from mpi4py import MPI
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi

def train(env_id, num_timesteps, seed, flight_log_dir, ckpt_dir):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=32, num_hid_layers=2)
    flight_log = FlightLog(flight_log_dir)
    env = gym.make(env_id)
    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5,
        vf_stepsize=1e-3,
            flight_log = flight_log,
            ckpt_dir = ckpt_dir
            )
    env.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str, default='attitude-zero-start-v0')
    parser.add_argument('--num-timesteps', type=int, default=1e7)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--flight-log-dir', type=str, default='./')
    parser.add_argument('--ckpt-dir', type=str, default='./')

    args = parser.parse_args()
    train(args.env_id, args.num_timesteps, args.seed, args.flight_log_dir,
          args.ckpt_dir )

