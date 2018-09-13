import tensorflow as tf
import gym, random
import numpy as np
import matplotlib.pyplot as plt

class ActionReplay:

    def __init__(self):

        self.state = []
        self.action = []
        self.reward = []
        self.next_state = []
        self.current = []
        self.is_final = []
        self.max_size  = 2300

    def get_mini_batch(self, size=32):

        mini_batch = {
            'state':[],
            'action':[],
            'reward':[],
            'next_state':[],
            'is_final':[]
        }
        samples = random.sample(range(len(self.state)), k=size)
        for index in samples:
            mini_batch['state'].append(self.state[index])
            mini_batch['action'].append(self.action[index])
            mini_batch['reward'].append(self.reward[index])
            mini_batch['next_state'].append(self.next_state[index])
            mini_batch['is_final'].append(self.is_final[index])

            #mini_batch.append({'state':self.state[index], 'action':self.action[index], 'next_state':self.next_state[index], 'is_final':self.is_final[index]})
        
        return mini_batch

    def add_replay_memory(self, state, action, reward, next_state, is_final):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.next_state.append(next_state)
        self.is_final.append(is_final)

    def is_full(self):
        if self.max_size < len(self.state):
            return True

    def flush_some(self):
        self.state = self.state[500:]
        self.action = self.action[500:]
        self.reward = self.reward[500:]
        self.next_state = self.next_state[500:]
        self.is_final = self.is_final[500:]


class DQN:
    '''
    Implement DQN
    '''

    def __init__(self, obs_dim, action_dim):
        self.observation = tf.placeholder("float",[None, obs_dim])
        self.reward = tf.placeholder("float")
        self.next_observation = tf.placeholder("float", [None, obs_dim])
        self.action = tf.placeholder(tf.int32)
        self.action_dim = action_dim
        self.build_network()
        self.memory = ActionReplay()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    
    def build_network(self):
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        
        layer_1_eval = tf.layers.dense(self.observation, 12, activation=tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer)
        layer_2_eval = tf.layers.dense(layer_1_eval, 24, activation=tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer)
        self.q = tf.layers.dense(layer_2_eval, self.action_dim, kernel_initializer=w_initializer,bias_initializer=b_initializer)

        layer_1_target = tf.layers.dense(self.next_observation, units=12, activation=tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer)
        layer_1_target = tf.layers.dense(layer_1_target, units=24, activation=tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer)
        self.next_q = tf.layers.dense(layer_1_target, units=self.action_dim, kernel_initializer=w_initializer, bias_initializer=b_initializer)
        #print(self.next_q, self.q)
        # Now we define loss fn
        self.target_q = self.reward + 0.9*tf.reduce_max(self.next_q)             #yi
        #.target_q = tf.stop_gradient(target_q)

        #Q(s',s') # if action dim > 1 else skip this part
        a_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32), self.action], axis=1)
        self.q_eval_wrt_a = tf.gather_nd(params=self.q, indices=a_indices)

        self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q_eval_wrt_a))  # (yi - Q(s', a'))^2

        self.train = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(self.loss)
    
    def memorize(self, state, action, reward, next_state, is_final):
        self.memory.add_replay_memory(state, action, reward, next_state, is_final)
        if self.memory.is_full():
            self.memory.flush_some()

    def save_mem(self):
        import pickle
        with open('action_Replay', 'wb') as fp:
            pickle.dump(self.memory, fp)
            fp.close()
    

    def select_action(self, qval):

        return np.argmax(qval)


env = gym.make('CartPole-v0')


dqn = DQN(4, 2)

#4 2
max_episodes = 1000
#Store action replay

loss_his = []
steps = 0
for i in range(max_episodes):
    obs = env.reset()
    done = False
    ep_r = 0
    while not done:
        env.render()
        qval  = dqn.sess.run(dqn.q, feed_dict={dqn.observation:obs.reshape(1, -1)})
        action = dqn.select_action(qval)
        print(action)
        next_observation, reward, done, info = env.step(action)
        #obs = observation
        #print(action, reward, done)
        # if done:
        #     reward = 210 + reward
        # else:
        #     reward = abs(next_observation[0] - obs[0])
        dqn.memorize(obs, action, reward, next_observation, done)
        ep_r += reward
        obs = next_observation
        
        if steps%320==0 and steps>100:

            mini_batch = dqn.memory.get_mini_batch(size=256)
            # break
            print('Trained')
            _, loss = dqn.sess.run([dqn.train, dqn.loss], feed_dict={
                dqn.observation:np.array(mini_batch['state']),
                dqn.action:mini_batch['action'],
                dqn.reward:mini_batch['reward'],
                dqn.next_observation:np.array(mini_batch['next_state'])
            })
            loss_his.append(loss)
        steps +=1
    print(i, ep_r, steps)
