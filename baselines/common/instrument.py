
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

def update_random_state_op(vars):
    update_ops = []
    # Getting registration errors for uint32
    k = tf.placeholder(tf.int64, [624])
    pos = tf.placeholder(tf.int64)
    has_gauss = tf.placeholder(tf.int64)
    cached = tf.placeholder(tf.float64)

    # Skip first becuase its a string
    update_ops.append(tf.assign(vars[0], k))
    update_ops.append(tf.assign(vars[1], pos))
    update_ops.append(tf.assign(vars[2], has_gauss))
    update_ops.append(tf.assign(vars[3], cached))

    return update_ops, [k, pos, has_gauss, cached]

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



def format_progress_data(step, state, action, info):
    sim_time = 0 if "sim_time"  not in info else info["sim_time"]
    reward_rpy = [0,0,0] if "reward_rpy"  not in info else info["reward_rpy"]

    return {'step': step,
            'sim_time': sim_time,
            'sp_r': state[0], 
            'sp_p' : state[1],
            'sp_y': state[2], 
            'r': state[3], 
            'p': state[4], 
            'y': state[5], 
            "m0": action[0], 
            "m1": action[1], 
            "m2": action[2], 
            "m3": action[3], 
            'r_reward' : reward_rpy[0], 
            'p_reward': reward_rpy[1], 
            'y_reward': reward_rpy[2]} 

def write_progress(progress_dir, episode, episode_data):
    if not os.path.exists(progress_dir):
            os.makedirs(progress_dir)
    filename =  "ep-{}.csv".format(episode)
    filepath = os.path.join(progress_dir, filename)
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = ['step', 'sim_time', 'sp_r', 'sp_p', 'sp_y', 'r', 'p', 'y', "m0", "m1", "m2", "m3", 'r_reward', 'p_reward', 'y_reward'] 
        data_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        data_writer.writeheader()
        for step in episode_data:
            data_writer.writerow(step)
