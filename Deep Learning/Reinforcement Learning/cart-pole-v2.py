import numpy as np 
import tensorflow as tf
import gym, random

import matplotlib.pyplot as plt


class ActionReplay:
    def __init__(self, max_mem):
        self.max_mem = max_mem
        self.n = 0
        self.replay_memory = []
        

    def add_memory(self, state, action, reward, next_state, terminal):
        if self.n > self.max_mem:
            self.flush()
        self.replay_memory.append([state, action, reward, next_state, terminal])
        self.n = len(self.replay_memory)
    
    def flush(self):
        self.replay_memory.pop(0)

    def get_mini_batch(self, size=256):
        batch = random.sample(self.replay_memory, size)
        return batch

    def get_all(self):
        return self.replay_memory

class DQN:

    def __init__(self, lr, state_n, action_n):
        self.state_n = state_n
        self.lr = lr 
        self.action_n = action_n
        self.state = tf.placeholder(tf.float32, [None, self.state_n])
        self.action = tf.placeholder(tf.int32)
        self.next_state = tf.placeholder(tf.float32, [None, self.state_n])
        self.reward = tf.placeholder(tf.float32)
        self.learn_step = 0
        self.copy_iter = 1000
        self.discount = 0.99
        self.memory = ActionReplay(10000)
        self.epsilon = 0.9
        self.epsilon_max = 0.9
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.loss_h = []

        
        self.build_model()

        parameter_target = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        parameter_eval = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        with tf.variable_scope('replacement'):
            self.replace_params = [tf.assign(t, e) for t, e in zip(parameter_target, parameter_eval)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def build_model(self):
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope('eval_net'):
            layer_1_eval = tf.layers.dense(self.state, units=16, activation=tf.nn.tanh, kernel_initializer=w_initializer, bias_initializer=b_initializer)
            layer_2_eval = tf.layers.dense(layer_1_eval, units=16, activation=tf.nn.tanh, kernel_initializer=w_initializer, bias_initializer=b_initializer)
            layer_3_eval = tf.layers.dense(layer_2_eval, units=64, activation=tf.nn.tanh, kernel_initializer=w_initializer, bias_initializer=b_initializer)
            self.q = tf.layers.dense(layer_3_eval, self.action_n, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='q')

        with tf.variable_scope('target_net'):
            layer_1_target = tf.layers.dense(self.next_state, units=16, activation=tf.nn.tanh, kernel_initializer=w_initializer, bias_initializer=b_initializer)
            layer_2_target = tf.layers.dense(layer_1_target, units=16, activation=tf.nn.tanh, kernel_initializer=w_initializer, bias_initializer=b_initializer)
            layer_3_target = tf.layers.dense(layer_2_target, units=64, activation=tf.nn.tanh, kernel_initializer=w_initializer, bias_initializer=b_initializer)
            self.q_next = tf.layers.dense(layer_3_target, self.action_n, kernel_initializer=w_initializer, bias_initializer=b_initializer, name='q_next')

        #yi
        q_target = self.reward + self.discount*tf.reduce_max(self.q_next, 1)
        self.q_target = tf.stop_gradient(q_target)

        #Q(s', a')
        a_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32), self.action], axis=1)
        self.q_eval_wrt_a = tf.gather_nd(params=self.q, indices=a_indices)

        self.loss = tf.reduce_mean(tf.square(self.q_target - self.q_eval_wrt_a))
        self.train = tf.train.RMSPropOptimizer(self.lr, decay=0.01, momentum=0.03).minimize(self.loss, var_list=[tf.trainable_variables(scope='eval_net')])

    def replace(self):
        self.sess.run(self.replace_params)
        print('Replaced')

    def learn(self):
        if self.learn_step%self.copy_iter==0:
            self.sess.run(self.replace_params)
            print('Replaced')
        
        mini_batch = self.memory.get_all()
        states = []
        actions = []
        rewards = []
        next_states = []
        terminal = []
        random.shuffle(mini_batch)
        for item in mini_batch:
            states.append(item[0])
            actions.append(item[1])
            rewards.append(item[2])
            next_states.append(item[3])
            terminal.append(item[4])
        
        _, loss = self.sess.run([self.train, self.loss], feed_dict={
            self.state:states,
            self.action:actions,
            self.reward:rewards,
            self.next_state:next_states,
        })
        self.loss_h.append(loss)

        self.learn_step +=1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_min
        


    def choose_action(self, state):
        
        if np.random.uniform() > self.epsilon:
            q = self.sess.run(self.q, feed_dict={self.state:state.reshape(1, -1)})
            return np.argmax(q)
        else:
            return np.random.randint(0, self.action_n)


    def memorize(self, s, a, r, n_s, t):
        self.memory.add_memory(s, a, r, n_s, t)

env = gym.make('CartPole-v0')
dqn = DQN(0.01, 4, 2)
steps = 0
re = []
for i in range(500):
    state = env.reset()
    ep_reward = 0
    steps = 0

    while True:
        env.render()
        action = dqn.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        
        ep_reward += reward
        if dqn.memory.n > 510:
            dqn.learn()
        steps +=1
        if done:
            dqn.memorize(state, action, reward, next_state, done)
            break
        state = next_state
        dqn.memorize(state, action, reward, next_state, done)
    
    

    re.append(ep_reward)    
    print("Episode : {} Reward : {} Steps :{}".format(i, ep_reward, steps))

print("Max : {} , Min : {} , Average {}".format(max(re), min(re), np.mean(re)))