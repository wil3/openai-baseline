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


class FlightLog:
    def __init__(self, save_dir=None):
        self.log = []
        self.save_dir = save_dir
        if self.save_dir and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Build dynamically
        self.log_fieldnames = [] 

        self.reward_sum = 0

    def add(self, step, state, r, action, info):
        # Add order to appear when saved to file
        record = {'step': step}
        if "step" not in self.log_fieldnames:
            self.log_fieldnames.append("step")

        # Get out of info if exists
        if "sim_time" in info:
            record["sim_time"] = info["sim_time"]
            if "sim_time" not in self.log_fieldnames:
                self.log_fieldnames.append("sim_time")

        record["reward"] = r
        if "reward" not in self.log_fieldnames:
            self.log_fieldnames.append("reward")

        self.reward_sum += r
        record["reward_sum"] = self.reward_sum
        if "reward_sum" not in self.log_fieldnames:
            self.log_fieldnames.append("reward_sum")

        if "sp" in info:
            sp = info["sp"]
            record.update({"sp_r": sp[0], "sp_p": sp[1], "sp_y": sp[2]})
            if "sp_r" not in self.log_fieldnames:
                self.log_fieldnames += ['sp_r', 'sp_p', 'sp_y']

        if "current_rpy" in info:
            rpy = info["current_rpy"]
            record.update({"r": rpy[0], "p": rpy[1], "y": rpy[2]})
            if "r" not in self.log_fieldnames:
                self.log_fieldnames += ["r", "p", "y"]
        

        record.update({"m0": action[0], "m1": action[1], "m2": action[2], "m3":
                       action[3]}) 
        if "m0" not in self.log_fieldnames:
            self.log_fieldnames += ["m0", "m1", "m2", "m3"]

        

        for i in range(len(state)): 
            state_name = "s{}".format(i)
            record[state_name] = state[i]
            if state_name not in self.log_fieldnames:
                self.log_fieldnames.append(state_name)

        if "health" in info:
            record["health"] = info["health"]
            if "health" not in self.log_fieldnames:
                self.log_fieldnames.append("health")
        if "axis" in info:
            record["axis"] = info["axis"]
            if "axis" not in self.log_fieldnames:
                self.log_fieldnames.append("axis")

        add = False 
        if "debug" in info:
            debug = info["debug"]
            #add = debug["log"]
            for key, val in debug.items():
                log_key = "dbg-{}".format(key)
                record[log_key] = val
                if log_key not in self.log_fieldnames:
                    self.log_fieldnames.append(log_key)


        #if add: 
        self.log.append(record)

    def clear(self):
        self.log = []
        self.reward_sum = 0

    def save(self, episode):
        filename =  "ep-{}.csv".format(episode)
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w', newline='') as csvfile:
            log_writer = csv.DictWriter(csvfile,
                                         fieldnames=self.log_fieldnames)
            log_writer.writeheader()
            for record in self.log:
                log_writer.writerow(record)
        return filepath



