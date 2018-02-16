import os
import time
from collections import deque
import pickle

from baselines.ddpg.ddpg import DDPG
import baselines.common.tf_util as U
from baselines.common.instrument import *

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI
import csv
import os


def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
    tau=0.01, eval_env=None, param_noise_adaption_interval=50 , ckpt_dir=None, progress_dir=None, progress_update_interval=1, save_per_epoch=1, seed=1):
    rank = MPI.COMM_WORLD.Get_rank()

    max_steps = nb_epochs * nb_epoch_cycles * nb_rollout_steps
    
    # Set up metrics and helpers for restore
    metric_vars = build_metric_vars(max_steps)
    metric_ops, metric_phs = get_metric_ops_phs(metric_vars)
    random_state_vars = build_random_state_vars()
    random_state_ops, random_state_phs = get_random_state_ops_phs(random_state_vars)

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    #logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver()
    else:
        saver = None

    random_state = None

    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    #with U.single_threaded_session() as sess:
    with U.make_session(num_cpu=1, supervise=False) as sess:
        # Prepare everything.
        agent.initialize(sess)

        restored = False
        if ckpt_dir:
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                logger.info('Restored model from checkpoint {}'.format(ckpt.model_checkpoint_path))
                restored = True

        agent.reset()
        done = False


        global_step = 0
        episode_step = 0
        t = 0
        epoch = 0
        episodes = 0
        episode_reward = 0.

        start_time = time.time()


        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_actions = []
        epoch_qs = []
        #epoch_episodes = 0

        episode_data = []
        start_epoch = 0

        if restored:
            [global_step,  
             episode_step, 
             t, 
             epoch, 
             episodes, 
             episode_reward, 
             epoch_episode_steps, 
             epoch_episode_rewards,  
             epoch_qs] = sess.run(metric_vars)

            start_epoch = epoch
            logger.info(" Restoring global_step={} episode_step={}, t={}, epoch={}, episodes={}".format(global_step, episode_step, t, epoch, episodes))

            epoch_episode_steps = epoch_episode_steps[:episodes].tolist()
            epoch_episode_rewards = epoch_episode_rewards[:episodes].tolist()
            epoch_qs = epoch_qs[:global_step].tolist()

            (k, pos, has_gauss, cached) = sess.run(random_state_vars)
            k = np.array(k, dtype=np.uint32)
            # Make sure the state was saved previously 
            if np.any(k):
                random_state = ('MT19937', k, pos, has_gauss, cached)
            else:
                logger.warn("Could not restore random state")

        env.seed(seed, state=random_state)
        if eval_env is not None:
            eval_env.seed(seed, state=random_state)

        obs = env.reset()
        if eval_env is not None:
            eval_obs = eval_env.reset()


        sess.graph.finalize()

        for epoch in range(start_epoch, nb_epochs):
            for cycle in range(nb_epoch_cycles):
                # Perform rollouts.
                for t_rollout in range(nb_rollout_steps):
                    # Predict next action.
                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    assert action.shape == env.action_space.shape

                    # Execute next action.
                    if rank == 0 and render:
                        env.render()
                    assert max_action.shape == action.shape
                    new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    episode_data.append(format_progress_data(episode_step, new_obs, max_action * action, info))

                    t += 1
                    if rank == 0 and render:
                        env.render()
                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    global_step += 1
                    agent.store_transition(obs, action, r, new_obs, done)
                    obs = new_obs

                    if done:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        #epoch_episodes += 1
                        episodes += 1

                        # Store the random state 
                        ep_rand_state = info["random_state"]
                        sess.run(random_state_ops, feed_dict={
                            random_state_phs[0]: ep_rand_state[1],
                            random_state_phs[1]: ep_rand_state[2],
                            random_state_phs[2]: ep_rand_state[3],
                            random_state_phs[3]: ep_rand_state[4],
                        })

                        agent.reset()
                        obs = env.reset()

                # Train.
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                for t_train in range(nb_train_steps):
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    cl, al = agent.train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()

                # Evaluate.
                eval_episode_rewards = []
                eval_qs = []
                if eval_env is not None:
                    eval_episode_reward = 0.
                    for t_rollout in range(nb_eval_steps):
                        eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                        eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        if render_eval:
                            eval_env.render()
                        eval_episode_reward += eval_r

                        eval_qs.append(eval_q)
                        if eval_done:
                            eval_obs = eval_env.reset()
                            eval_episode_rewards.append(eval_episode_reward)
                            eval_episode_rewards_history.append(eval_episode_reward)
                            eval_episode_reward = 0.

            mpi_size = MPI.COMM_WORLD.Get_size()
            # Log stats.
            # XXX shouldn't call np.mean on variable length lists
            duration = time.time() - start_time
            stats = agent.get_stats()
            combined_stats = stats.copy()
            combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
            combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
            combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
            combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
            combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
            combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
            combined_stats['total/duration'] = duration
            combined_stats['total/steps_per_second'] = float(t) / float(duration)
            combined_stats['total/episodes'] = episodes
            #combined_stats['rollout/episodes'] = epoch_episodes
            combined_stats['rollout/actions_std'] = np.std(epoch_actions)
            # Evaluation statistics.
            if eval_env is not None:
                combined_stats['eval/return'] = eval_episode_rewards
                combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                combined_stats['eval/Q'] = eval_qs
                combined_stats['eval/episodes'] = len(eval_episode_rewards)
            def as_scalar(x):
                if isinstance(x, np.ndarray):
                    assert x.size == 1
                    return x[0]
                elif np.isscalar(x):
                    return x
                else:
                    raise ValueError('expected scalar, got %s'%x)
            combined_stats_sums = MPI.COMM_WORLD.allreduce(np.array([as_scalar(x) for x in combined_stats.values()]))
            combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

            # Total statistics.
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()
            if rank == 0 and logdir:
                if hasattr(env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(env.get_state(), f)
                if eval_env and hasattr(eval_env, 'get_state'):
                    with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                        pickle.dump(eval_env.get_state(), f)

            # Save model
            if rank == 0 and epoch % save_per_epoch == 0 and ckpt_dir is not None:
                #pad
                epoch_episode_steps += [0.0] * (max_steps - len(epoch_episode_steps))
                epoch_episode_rewards += [0.0] * (max_steps - len(epoch_episode_rewards))
                #epoch_qs += [0.0] * (max_steps - len(epoch_qs))
                _epoch_qs = np.array(epoch_qs)
                _epoch_qs = np.pad(epoch_qs, ( (0, max_steps - len(epoch_qs)), (0, 0), (0,0) ), 'edge')

                update_metrics(metric_ops, metric_phs, global_step,  episode_step, t, epoch, episodes, episode_reward, epoch_episode_steps, epoch_episode_rewards,  _epoch_qs)
                #print("global step is =", global_step)
                #sess.run(op_gs, feed_dict = {ph_gs:global_step})

                task_name = "flightcontrol-ddpg-default.ckpt"
                fname = os.path.join(ckpt_dir, task_name)
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                saver.save(sess, fname)
                logger.info("Saving model to {}".format(fname))

            if epoch % progress_update_interval == 0:
                write_progress(progress_dir, epoch, episode_data)
            episode_data = []

