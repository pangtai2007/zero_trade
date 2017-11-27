import gym
import random
import logging
import tensorflow as tf

from utils import get_model_dir
from networks.cnn import CNN
from networks.mlp import MLPSmall
from agents.statistic import Statistic
from environments.trader_environment import TraderEnvironment
from environments.market_db import MarketDb


flags = tf.app.flags

# Deep q Network
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not. gpu use NHWC and gpu use NCHW for data_format')
flags.DEFINE_string('agent_type', 'DQN', 'The type of agent [DQN]')
flags.DEFINE_boolean('double_q', False, 'Whether to use double Q-learning')
flags.DEFINE_string('network_header_type', 'nips', 'The type of network header [mlp, nature, nips]')
flags.DEFINE_string('network_output_type', 'normal', 'The type of network output [normal, dueling]')

# Environment
flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_integer('n_action_repeat', 4, 'The number of actions to repeat')
flags.DEFINE_integer('max_random_start', 30, 'The maximum number of NOOP actions at the beginning of an episode')
flags.DEFINE_integer('history_length', 1, 'The length of history of observation to use as an input to DQN')
flags.DEFINE_integer('max_r', +1, 'The maximum value of clipped reward')
flags.DEFINE_integer('min_r', -1, 'The minimum value of clipped reward')
flags.DEFINE_string('observation_dims', '[80, 80]', 'The dimension of gym observation')
flags.DEFINE_boolean('random_start', True, 'Whether to start with random state')
flags.DEFINE_boolean('use_cumulated_reward', False, 'Whether to use cumulated reward or not')

# Training
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('max_delta', None, 'The maximum value of delta')
flags.DEFINE_integer('min_delta', None, 'The minimum value of delta')
flags.DEFINE_float('ep_start', 1., 'The value of epsilon at start in e-greedy')
flags.DEFINE_float('ep_end', 0.003, 'The value of epsilnon at the end in e-greedy')
flags.DEFINE_integer('batch_size', 128, 'The size of batch for minibatch training')
flags.DEFINE_integer('max_grad_norm', None, 'The maximum norm of gradient while updating')
flags.DEFINE_float('discount_r', 0.99, 'The discount factor for reward')

# Timer
flags.DEFINE_integer('t_train_freq', 4, '')

# Below numbers will be multiplied by scale
flags.DEFINE_integer('scale', 10000, 'The scale for big numbers')
flags.DEFINE_integer('memory_size', 20, 'The size of experience memory (*= scale)')
flags.DEFINE_integer('t_target_q_update_freq', 1, 'The frequency of target network to be updated (*= scale)')
flags.DEFINE_integer('t_test', 1, 'The maximum number of t while training (*= scale)')
flags.DEFINE_integer('t_ep_end', 100, 'The time when epsilon reach ep_end (*= scale)')
flags.DEFINE_integer('t_train_max', 5000, 'The maximum number of t while training (*= scale)')
flags.DEFINE_float('t_learn_start', 5, 'The time when to begin training (*= scale)')
flags.DEFINE_float('learning_rate_decay_step', 5, 'The learning rate of training (*= scale)')

# Optimizer
flags.DEFINE_float('learning_rate', 0.002, 'The learning rate of training')
flags.DEFINE_float('learning_rate_minimum', 0.0002, 'The minimum learning rate of training')
flags.DEFINE_float('learning_rate_decay', 0.9, 'The decay of learning rate of training')
flags.DEFINE_float('decay', 0.99, 'Decay of RMSProp optimizer')
flags.DEFINE_float('momentum', 0.0, 'Momentum of RMSProp optimizer')
flags.DEFINE_float('gamma', 0.99, 'Discount factor of return')
flags.DEFINE_float('beta', 0.01, 'Beta of RMSProp optimizer')

# Debug
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_string('log_level', 'INFO', 'Log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')
flags.DEFINE_string('tag', '', 'The name of tag for a model, only for debugging')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_string('test_dir', 'test_data', 'test_directory')
flags.DEFINE_string('train_dir', 'train_data', 'train_directory')
flags.DEFINE_string('out_file', 'trader.csv', 'train_directory')


# source 

flags.DEFINE_string('engine', 'text', 'test/text/influxdb')
flags.DEFINE_string('data', 'data', 'data directory path')

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print (" [*] GPU : %.4f" % fraction)
  return fraction

conf = flags.FLAGS

if conf.agent_type == 'DQN':
  from agents.deep_q import DeepQ
  TrainAgent = DeepQ
else:
  raise ValueError('Unknown agent_type: %s' % conf.agent_type)

logger = logging.getLogger()
logger.propagate = False
logger.setLevel(conf.log_level)

# set random seed
tf.set_random_seed(conf.random_seed)
random.seed(conf.random_seed)

def main(_):
  # preprocess

  for flag in ['memory_size', 't_target_q_update_freq', 't_test',
               't_ep_end', 't_train_max', 't_learn_start', 'learning_rate_decay_step']:
    setattr(conf, flag, getattr(conf, flag) * conf.scale)

  if conf.use_gpu:
    conf.data_format = 'NCHW'
  else:
    conf.data_format = 'NHWC'

  model_dir = get_model_dir(conf,
      ['use_gpu', 'max_random_start', 'n_worker', 'is_train', 'memory_size', 'gpu_fraction',
       't_save', 't_train', 'display', 'log_level', 'random_seed', 'tag', 'scale', 'observation_dims', 'train_dir', 'test_dir', 'out_file'])

  # start
  gpu_options = tf.GPUOptions(
      per_process_gpu_memory_fraction=calc_gpu_fraction(conf.gpu_fraction))

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #database_config = dict(host="localhost",port = 8086, username="root", password="", database="markets", engine="influxdb")
    if conf.is_train:
      data_dir = conf.train_dir
    else:
      data_dir = conf.test_dir
    database_config = dict(engine="text", data=data_dir)
    trader_config = dict(margin_rate = 0.1, commission_rate = 0.001, volume_multiple=10, init_capital = 100000, equity_limit = 0.9, stop_loss = 50, volume_level = 0.33, hold_equity_reward = True)
    env_config = dict(database=database_config, trader = trader_config, display=conf.display, out_file=conf.out_file)
    env = TraderEnvironment(conf.data_format, env_config)

    conf.observation_dims = env.observation_dims

    if conf.network_header_type in ['nature', 'nips']:
      pred_network = CNN(sess=sess,
                         data_format=conf.data_format,
                         history_length=conf.history_length,
                         observation_dims=conf.observation_dims,
                         output_size=env.env.action_space.n,
                         network_header_type=conf.network_header_type,
                         name='pred_network', trainable=True)
      target_network = CNN(sess=sess,
                           data_format=conf.data_format,
                           history_length=conf.history_length,
                           observation_dims=conf.observation_dims,
                           output_size=env.env.action_space.n,
                           network_header_type=conf.network_header_type,
                           name='target_network', trainable=False)
    elif conf.network_header_type == 'mlp':
      pred_network = MLPSmall(sess=sess,
                              observation_dims=conf.observation_dims,
                              history_length=conf.history_length,
                              output_size=env.env.action_space.n,
                              hidden_activation_fn=tf.sigmoid,
                              network_output_type=conf.network_output_type,
                              name='pred_network', trainable=True)
      target_network = MLPSmall(sess=sess,
                                observation_dims=conf.observation_dims,
                                history_length=conf.history_length,
                                output_size=env.env.action_space.n,
                                hidden_activation_fn=tf.sigmoid,
                                network_output_type=conf.network_output_type,
                                name='target_network', trainable=False)
    else:
      raise ValueError('Unkown network_header_type: %s' % (conf.network_header_type))
    stat = Statistic(sess, conf.t_test, conf.t_learn_start, model_dir, pred_network.var.values())
    agent = TrainAgent(sess, pred_network, env, stat, conf, target_network=target_network)

    if conf.is_train:
      agent.train(conf.t_train_max)
    else:
      agent.play(env.max_tick)

if __name__ == '__main__':
  tf.app.run()