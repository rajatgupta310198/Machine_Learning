#
# Solving OpenAI Gym CartPole game with Tensorflow
# Author Ilhwan Hwang (https://github.com/IlhwanHwang)
# Date 17.08.19
#

import gym
import tensorflow as tf
import numpy as np
import random
from argparse import Namespace

# Metaparameter initializer
# @Parameter
#	Nothing
# @Return
#	argparse.Namespace	metaparameter namespace
def init_meta():
	meta = Namespace()
	meta.OBS_DIM = 4
	meta.ACT_DIM = 2
	meta.HIDDEN1_DIM = 128
	meta.HIDDEN2_DIM = 128
	meta.HIDDEN3_DIM = 128
	meta.GAMMA = 0.9
	meta.LEARNING_RATE = 1e-4
	meta.EXP_SIZE = 2048
	meta.EPS_DECAY = 0.95
	meta.EPS_STEP = 10
	meta.EPS_MIN = 1e-5
	meta.BATCH_SIZE = 256
	meta.RENDER_INTERVAL = 64
	meta.LEARN_ITER = 1
	meta.SCORE_TARGET = 200
	return meta

# Xavier variable generator
# @Parameter
#	string	name to use as tensorflow variable name
#	array	shape of the variable
# @Return
#	tensorflow.Variable	resulted variable
def xavier_variable(name, shapce):
	if len(shape) == 1:
		inout = np.sum(shape) + 1
	else:
		inout = np.sum(shape)

	init_range = np.sqrt(6.0 / inout)
	initializer = tf.random_uniform_initializer(-init_range, init_range)
	return tf.get_variable(name, shape, initializer=initializer)

class Agent:

	# @Parameter
	#	argparse.Namespace	meta parameters
	#	tensorflow.Session 	default session
	def __init__(self, meta, sess):
		# Define structure
		self.W1 = xavier_variable('W1', [meta.OBS_DIM, meta.HIDDEN1_DIM])
		self.B1 = xavier_variable('B1', [meta.HIDDEN1_DIM])
		self.W2 = xavier_variable('W2', [meta.HIDDEN1_DIM, meta.HIDDEN2_DIM])
		self.B2 = xavier_variable('B2', [meta.HIDDEN2_DIM])
		self.W3 = xavier_variable('W3', [meta.HIDDEN2_DIM, meta.HIDDEN3_DIM])
		self.B3 = xavier_variable('B3', [meta.HIDDEN3_DIM])
		self.W4 = xavier_variable('W4', [meta.HIDDEN3_DIM, meta.ACT_DIM])
		self.B4 = xavier_variable('B4', [meta.ACT_DIM])

		self.obs_prev = tf.placeholder(tf.float32, [None, meta.OBS_DIM])
		self.obs_next = tf.placeholder(tf.float32, [None, meta.OBS_DIM])

		self.h1_prev = tf.nn.relu(tf.matmul(self.obs_prev, self.W1) + self.B1)
		self.h2_prev = tf.nn.relu(tf.matmul(self.h1_prev, self.W2) + self.B2)
		self.h3_prev = tf.nn.relu(tf.matmul(self.h2_prev, self.W3) + self.B3)
		self.Q_prev = tf.squeeze(tf.matmul(self.h3_prev, self.W4) + self.B4)
		self.Q_prev_softmax = tf.nn.softmax(self.Q_prev)

		self.h1_next = tf.nn.relu(tf.matmul(self.obs_next, self.W1) + self.B1)
		self.h2_next = tf.nn.relu(tf.matmul(self.h1_next, self.W2) + self.B2)
		self.h3_next = tf.nn.relu(tf.matmul(self.h2_next, self.W3) + self.B3)
		self.Q_next = tf.squeeze(tf.matmul(self.h3_next, self.W4) + self.B4)

		self.action = tf.placeholder(tf.float32, [None, meta.ACT_DIM])
		self.reward = tf.placeholder(tf.float32, [None, ])

		self.Q_current = tf.reduce_sum(tf.multiply(self.Q_prev, self.action), axis=1)
		self.Q_target = self.reward + meta.GAMMA * tf.reduce_max(self.Q_next, axis=1)

		self.loss = tf.reduce_mean(tf.square(self.Q_target - self.Q_current))
		self.train = tf.train.AdamOptimizer(meta.LEARNING_RATE).minimize(self.loss)

		# Experience queue
		self.exp_ind = 0
		self.exp_obs_prev = np.empty([meta.EXP_SIZE, meta.OBS_DIM])
		self.exp_action = np.empty([meta.EXP_SIZE, meta.ACT_DIM])
		self.exp_reward = np.empty([meta.EXP_SIZE, ])
		self.exp_obs_next = np.empty([meta.EXP_SIZE, meta.OBS_DIM])
		self.exp_full = False

		# Epsilon
		self.eps = 1

		# Session
		self.sess = sess

		# Save metadata for future use
		self.meta = meta

	# Feed experience sample to agent
	# @Parameter
	#	array	observation (S)
	#	array	action (a)
	#	float 	reward (r)
	#	array	observation (S')
	# @Return
	#	Nothing
	def feed(self, obs_prev, action, reward, obs_next):
		self.exp_obs_prev[self.exp_ind] = obs_prev
		self.exp_action[self.exp_ind] = action
		self.exp_reward[self.exp_ind] = reward
		self.exp_obs_next[self.exp_ind] = obs_next

		self.exp_ind += 1
		# Rotate the experiences
		if self.exp_ind >= self.meta.EXP_SIZE:
			self.exp_ind = 0
			self.exp_full = True

	# Run 1 batch session
	# @Parameter
	#	Nothing
	# @Return
	#	float 	loss
	def batch(self):
		if self.exp_full:
			indices = np.random.choice(self.meta.EXP_SIZE, self.meta.BATCH_SIZE)
			feed = {
				self.obs_prev : self.exp_obs_prev[indices], 
				self.action : self.exp_action[indices], 
				self.reward : self.exp_reward[indices], 
				self.obs_next : self.exp_obs_next[indices]
			}
			loss_value, _ = self.sess.run([self.loss, self.train], feed_dict=feed)
			return loss_value
		else:
			print("Experience queue is not filled yet")
			return 0.0

	# Check whether the experience queue is full
	# @Parameter
	#	Nothing
	# @Return
	#	bool	fullness
	def is_full(self):
		return self.exp_full

	# Make decision according to the observation
	# @Parameter
	#	array	observation (len == OBS_DIM)
	# @Return
	#	int 	action index
	#	array	action one-hot vector
	def act(self, obs):
		if random.random() < self.eps:
			act_ind = random.randrange(self.meta.ACT_DIM)
		else:
			feed = {self.obs_prev : np.reshape(obs, (1, -1))}
			# Use softmaxed Q as probability distribution of action
			pd = self.sess.run(self.Q_prev_softmax, feed_dict=feed)
			act_cand = range(self.meta.ACT_DIM)
			act_ind = np.random.choice(act_cand, 1, p=pd)
			act_ind = act_ind[0] # De-array-fy
		act = np.zeros(self.meta.ACT_DIM)
		act[act_ind] = 1
		return act_ind, act

	# Decay the epsilon
	def decay(self):
		if self.eps > self.meta.EPS_MIN:
			self.eps = self.eps * self.meta.EPS_DECAY

# The main routine
def main():
	env = gym.make('CartPole-v0')
	meta = init_meta()

	sess = tf.Session()
	agent = Agent(meta, sess)
	sess.run(tf.global_variables_initializer())

	global_count = 0

	# Episode loop
	for episode in range(4000):
		obs_prev = env.reset()
		reward = 0.0
		score = 0
		done = False
		loss_sum = 0.0
		loss_count = 0

		# Step loop
		while not done:
			env.render()
			global_count += 1

			if global_count % meta.EPS_STEP == 0:
				agent.decay()

			# Make choice
			act_ind, action = agent.act(obs_prev)

			# Do one step
			obs_next, _, done, _ = env.step(act_ind)

			# Reward is increasing gradually
			reward += 1
			# Score is just to display
			score += 1

			# If it fails, give a bitter taste
			if done and reward < meta.SCORE_TARGET:
				reward = -200
				obs_next = np.zeros_like(obs_next)

			# Feed the experience
			agent.feed(obs_prev, action, reward, obs_next)

			# Send observation to the next step
			obs_prev = obs_next

			if agent.is_full():
				for _ in range(meta.LEARN_ITER):
					loss_sum += agent.batch()
					loss_count += 1

		if loss_count == 0:
			loss_avg = 0
		else:
			loss_avg = loss_sum / loss_count
		print("Episode {}: loss = {:.2f}, score = {}".format(episode + 1, loss_avg, score))


if __name__ == "__main__":
    main()