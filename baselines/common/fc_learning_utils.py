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

        self.precision = 6

        self.last_sp = []
        self.error_sum_rpy = np.zeros(3)
        self.error_sum = 0
        self.steps = np.ones(3)

        self.summary_fieldnames = [] 
        self.ep_summary = {}

        self.max_sim_time = 0

    def add_list(self, step, state, r, action, info):
        for i in info:
            self.add(step, state, r, action, i)

    def add(self, step, state, r, action, info):
        if isinstance(info, list):
            self.add_list(step, state, r, action, info)


        # Add order to appear when saved to file
        record = {}


        # Remove step and sim time to save space logging
        # since these are sequential
        
        #record = {'step': step}
        #if "step" not in self.log_fieldnames:
        #    self.log_fieldnames.append("step")

        # Get out of info if exists
        if "sim_time" in info:
            if not info["sim_time"]:
                return 
            record["sim_time"] = info["sim_time"]
            self.max_sim_time = info["sim_time"]
            if "sim_time" not in self.log_fieldnames:
                self.log_fieldnames.append("sim_time")

        record["reward"] = self._format(r)
        if "reward" not in self.log_fieldnames:
            self.log_fieldnames.append("reward")

        self.reward_sum += r

        #record["reward_sum"] = self.reward_sum
        #if "reward_sum" not in self.log_fieldnames:
        #    self.log_fieldnames.append("reward_sum")
        sp=None
        sp_prefix=None
        if "desired_rate" in info: #gymfc-dev
            sp = info["desired_rate"]
            sp_prefix = "desired_rate_"
        elif "sp" in info: #gymfc
            sp = info["sp"]
            sp_prefix = "sp_"

        if sp_prefix:
            # Only write when the sp changes to save logging space
            # for episodic this will only occur the first time, continuous
            # will change
            if len(self.last_sp) == 0 or (self.last_sp != sp).any():
                record.update({"{}r".format(sp_prefix): self._format(sp[0]), "{}p".format(sp_prefix): self._format(sp[1]), "{}y".format(sp_prefix): self._format(sp[2])})
            else:
                record.update({"{}r".format(sp_prefix): "", "{}p".format(sp_prefix): "", "{}y".format(sp_prefix): ""})
            self.last_sp = sp

            if "{}r".format(sp_prefix) not in self.log_fieldnames:
                self.log_fieldnames += ['{}r'.format(sp_prefix), '{}p'.format(sp_prefix), '{}y'.format(sp_prefix)]

        if "measured_rate" in info:
            rpy = info["measured_rate"]
            record.update({"measured_rate_r": self._format(rpy[0]), "measured_rate_p": self._format(rpy[1]), "measured_rate_y": self._format(rpy[2])})
            if "measured_rate_r" not in self.log_fieldnames:
                self.log_fieldnames += ["measured_rate_r", "measured_rate_p", "measured_rate_y"]
        
        rpy = []
        rpy_prefix = None
        rpy_keys = []
        if "true_rate" in info:
            rpy = info["true_rate"]
            rpy_keys = ["true_rate_r","true_rate_p","true_rate_y"]
        elif "current_rpy" in info:
            rpy = info["current_rpy"]
            rpy_keys = ["r", "p", "y"]


        if len(rpy) > 0:

            record.update({rpy_keys[0]: self._format(rpy[0]),
                           rpy_keys[1]: self._format(rpy[1]), 
                           rpy_keys[2]: self._format(rpy[2])})
            if rpy_keys[0] not in self.log_fieldnames:
                self.log_fieldnames += rpy_keys 
        
            self.error_sum += np.sum(np.abs(sp - rpy))
            self.error_sum_rpy += np.abs(sp - rpy)


	
        if "measured_motor" in info:
            measured_rpm_motor = info["measured_motor"]
            record.update({"measured_rpm_m0": self._format(measured_rpm_motor[0]), "measured_rpm_m1": self._format(measured_rpm_motor[1]), "measured_rpm_m2": self._format(measured_rpm_motor[2]), "measured_rpm_m3":
                           self._format(measured_rpm_motor[3])}) 
            if "measured_rpm_m0" not in self.log_fieldnames:
                self.log_fieldnames += ["measured_rpm_m0", "measured_rpm_m1", "measured_rpm_m2", "measured_rpm_m3"]

            true_rpm_motor = info["true_motor"]
            record.update({"true_rpm_m0": self._format(true_rpm_motor[0]), "true_rpm_m1": self._format(true_rpm_motor[1]), "true_rpm_m2": self._format(true_rpm_motor[2]), "true_rpm_m3":
                           self._format(true_rpm_motor[3])}) 
            if "true_rpm_m0" not in self.log_fieldnames:
                self.log_fieldnames += ["true_rpm_m0", "true_rpm_m1", "true_rpm_m2", "true_rpm_m3"]


        y_fields = []
        for i in range(len(action)):
            key = "y{}".format(i)
            y_fields.append(key)
            record.update({key : self._format(action[i])})
        if "y0" not in self.log_fieldnames:
            self.log_fieldnames += y_fields

        """
        for i in range(len(state)): 
            state_name = "s{}".format(i)
            record[state_name] = self._format(state[i])
            if state_name not in self.log_fieldnames:
                self.log_fieldnames.append(state_name)
        """

        """
        if "health" in info:
            record["health"] = info["health"]
            if "health" not in self.log_fieldnames:
                self.log_fieldnames.append("health")
        if "axis" in info:
            record["axis"] = info["axis"]
            if "axis" not in self.log_fieldnames:
                self.log_fieldnames.append("axis")
        """
        add = False 
        if "debug" in info:
            debug = info["debug"]
            #add = debug["log"]
            for key, val in debug.items():
                log_key = "dbg-{}".format(key)
                record[log_key] = val
                if log_key not in self.log_fieldnames:
                    self.log_fieldnames.append(log_key)

        if "summary" in info:
            summary = info["summary"]
            self.ep_summary = summary
            for key, val in summary.items():
                if key not in self.summary_fieldnames:
                    self.summary_fieldnames.append(key)

        #if add: 
        self.log.append(record)

    def _format(self, value):
        return "{:.6f}".format(value)

    def is_wave(self):
        if len(self.observation_history) > 2:
# We are rounding these because very small chnages (ie -8) are triggering this
            current = np.round(self.observation_history[-1].true_angular_velocity_rpy)
            previous_1 = np.round(self.observation_history[-2].true_angular_velocity_rpy)
            delta_current = current - previous_1
            slope_dir_rpy = np.sign(delta_current)
            slope_direction_changed = (slope_dir_rpy != self.last_slope_dir_rpy) & np.logical_not(np.isnan(self.last_slope_dir_rpy))
            self.last_slope_dir_rpy = slope_dir_rpy

            slopes_not_zero = np.abs(slope_dir_rpy)
            
            should_apply_penalty = self.axis_enabled * slopes_not_zero * np.nan_to_num(slope_direction_changed)

    def clear(self):
        self.log = []
        self.error_sum = 0
        self.reward_sum = 0
        self.error_sum_rpy = np.zeros(3)

    def save(self, episode):

        self.save_progress(episode)

        filename =  "ep-{}.csv".format(episode)
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w', newline='') as csvfile:
            log_writer = csv.DictWriter(csvfile,
                                         fieldnames=self.log_fieldnames)
            log_writer.writeheader()
            for record in self.log:
                log_writer.writerow(record)
        return filepath

    def save_progress(self, ep):
        fieldnames = ["ep", "total_reward", "error_sum", "total_time", "e_r", "e_p", "e_y"]
        filename = "progress.csv"
        filepath = os.path.join(self.save_dir,filename)
        file_exists = os.path.isfile(filepath)
        # TODO add ci
        with open(filepath, 'a', newline='') as csvfile:
            log_writer = csv.DictWriter(csvfile,
                                         fieldnames=fieldnames)
            if not file_exists:
                log_writer.writeheader()
            ep_summary = {"ep": ep, "total_reward": self.reward_sum, "total_time": self.max_sim_time, "error_sum": self.error_sum, "e_r": self.error_sum_rpy[0], "e_p":self.error_sum_rpy[1], "e_y":self.error_sum_rpy[2]}
            log_writer.writerow(ep_summary)




