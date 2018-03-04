import numpy as np
import tensorflow as tf
import csv
import os
def build_random_state_vars():
    """ This is separate so they can be fed in the dict,
    From https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.get_state.html
     1-D array of 624 unsigned integer keys.
     an integer pos.
     an integer has_gauss.
     a float cached_gaussian
    """
    k = tf.get_variable("random_state_keys", [624], dtype=tf.int64, trainable=False)
    pos = tf.get_variable("random_state_pos", [], dtype=tf.int64, trainable=False)
    has_gauss = tf.get_variable("random_state_has_gauss", [], dtype=tf.int64, trainable=False)
    cached = tf.get_variable("random_state_cached_gauss", [], dtype=tf.float64, trainable=False)
    return [k, pos, has_gauss, cached]

def get_random_state_ops_phs(vars):
    ph = []
    update_ops = []
    with tf.variable_scope('random_state'):
        # Getting registration errors for uint32
        ph = [ tf.placeholder(tf.int64, [624]),
         tf.placeholder(tf.int64),
         tf.placeholder(tf.int64),
         tf.placeholder(tf.float64),
         ]

        for i in range(len(vars)):
            update_ops.append(tf.assign(vars[i], ph[i]))

    return update_ops, ph 

def get_metric_ops_phs(vars):
    ph = []
    update_ops = []
    with tf.variable_scope('metrics'):
        ph = [tf.placeholder(tf.int64),
              tf.placeholder(tf.int64, name="ph_episode_step"),
              tf.placeholder(tf.int64),
              tf.placeholder(tf.int64),
              tf.placeholder(tf.int64),
              tf.placeholder(tf.float64),

              tf.placeholder(tf.int64, shape=(None,), name="ph_epoch_episode_steps"),
              tf.placeholder(tf.float64, shape=(None,), name="ph_epoch_episode_rewards"),
              tf.placeholder(tf.float64, shape=(None, None, None,), name="ph_epoch_qs"),

              ]

        for i in range(len(vars)):
            update_ops.append(tf.assign(vars[i], ph[i]))
    return update_ops, ph

def build_metric_vars(max_size):
    with tf.variable_scope('metrics'):
        # Scalars
        global_step = tf.get_variable("global_step", [], dtype=tf.int64, trainable=False)
        episode_step = tf.get_variable("episode_step", [], dtype=tf.int64, trainable=False)
        t = tf.get_variable("t", [], dtype=tf.int64, trainable=False)
        epoch = tf.get_variable("epoch", [], dtype=tf.int64, trainable=False)
        # This will be the index for the episode arrays
        episodes = tf.get_variable("episodes", [], dtype=tf.int64, trainable=False)

        episode_reward = tf.get_variable("episode_reward", [], dtype=tf.float64, trainable=False)


        # Vectors
        epoch_episode_steps = tf.get_variable("epoch_episode_steps", [max_size], dtype=tf.int64, trainable=False)
        epoch_episode_rewards = tf.get_variable("epoch_episode_rewards", [max_size], dtype=tf.float64, trainable=False)
        # This is indexed by global step
        # FIXME this is the number of actions
        epoch_qs = tf.get_variable("epoch_qs", [max_size, 1, 1], dtype=tf.float64, trainable=False)

        return [global_step,  episode_step, t, epoch, episodes, episode_reward, epoch_episode_steps, epoch_episode_rewards,  epoch_qs]

def update_metrics(ops, ph, *args):
    fd = {}
    for i in range(len(args)):
        fd[ph[i]] = args[i]
    tf.get_default_session().run(ops, feed_dict = fd)



def format_progress_data(step, state, r, action, info):
    sim_time = 0 if "sim_time"  not in info else info["sim_time"]
    sp = np.zeros(3) if "sp" not in info else info["sp"]
    rpy = np.zeros(3) if "current_rpy" not in info else info["current_rpy"]

    """
    reward_rpy = np.zeros(3) if "reward_rpy"  not in info else info["reward_rpy"]
    pterm = np.zeros(3) if "Pterm" not in info else info["Pterm"]
    iterm = np.zeros(3) if "Iterm" not in info else info["Iterm"]
    dterm = np.zeros(3) if "Dterm" not in info else info["Dterm"]
    distance = np.zeros(3) if "Error_Reward" not in info else info["Error_Reward"]
    accel = np.zeros(3) if "Accel_Reward" not in info else info["Accel_Reward"]
    """

    data = {'step': step,
            'sim_time': sim_time,
            'sp_r': sp[0], 
            'sp_p' : sp[1],
            'sp_y': sp[2], 
            'r': rpy[0], 
            'p': rpy[1], 
            'y': rpy[2], 
            "m0": action[0], 
            "m1": action[1], 
            "m2": action[2], 
            "m3": action[3], 
            "reward": r,
            }
    """
            'r_reward' : reward_rpy[0], 
            'p_reward': reward_rpy[1], 
            'y_reward': reward_rpy[2],
            'P_r':pterm[0],
            'P_p':pterm[1],
            'P_y':pterm[2],
            'I_r':iterm[0],
            'I_p':iterm[1],
            'I_y':iterm[2],
            'D_r':dterm[0],
            'D_p':dterm[1],
            'D_y':dterm[2],
            'E_r':distance[0],
            'E_p':distance[1],
            'E_y':distance[2],
            "A_r": accel[0],
            "A_p": accel[1],
            "A_y": accel[2],

            } 
    """
    # Append the actual state used
    for i in range(len(state)): 
        data["s{}".format(i)] = state[i]

    return data

def write_progress(progress_dir, episode, episode_data):
    if not os.path.exists(progress_dir):
            os.makedirs(progress_dir)
    filename =  "ep-{}.csv".format(episode)
    filepath = os.path.join(progress_dir, filename)
    with open(filepath, 'w', newline='') as csvfile:
        # Get the first entry and grab the keys used
        fieldnames = ['step', 'sim_time', 'sp_r', 'sp_p', 'sp_y', 'r', 'p', 'y', "m0", "m1", "m2", "m3", "reward"]#, 'r_reward', 'p_reward','y_reward']
#, 'P_r', 'P_p', 'P_y', 'I_r', 'I_p', 'I_y','D_r', 'D_p', 'D_y', "E_r", "E_p", "E_y", "A_r", "A_p", "A_y"] 
        # Append states used at the end to make csv file more readable
        i = 0
        while True:
            state_key = "s{}".format(i)
            if state_key in episode_data[0]:
                fieldnames.append(state_key)
                i += 1
            else:
                break

        #fieldnames = episode_data[0].keys() 
        data_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        data_writer.writeheader()
        for step in episode_data:
            data_writer.writerow(step)
