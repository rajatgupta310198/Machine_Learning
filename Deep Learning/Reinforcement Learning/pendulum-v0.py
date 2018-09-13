# from dqn_phy import DQN

# import tensorflow as tf
# import numpy as np
# import gym



# env = gym.make('Pendulum-v0')
# dqn = DQN(3, 1)

# max_episodes = 1000
# #Store action replay

# loss_his = []
# steps = 0
# for i in range(max_episodes):
#     obs = env.reset()
#     done = False
#     ep_r = 0
#     while not done:
#         env.render()
#         qval  = dqn.sess.run(dqn.q, feed_dict={dqn.observation:obs.reshape(1, -1)})
        
#         action = dqn.select_action(qval)
        
#         next_observation, reward, done, info = env.step([action])
#         #obs = observation
#         #print(i, next_observation, action, reward, done)
#         dqn.memorize(obs, action, reward, next_observation, done)
#         ep_r += reward
#         obs = next_observation
        
#     #     if steps%100 and steps>100:

#     #         mini_batch = dqn.memory.get_mini_batch(size=32)
#     #         # break
#     #         #print('Trained')
#     #         _, loss = dqn.sess.run([dqn.train, dqn.loss], feed_dict={
#     #             dqn.observation:np.array(mini_batch['state']),
#     #             dqn.action:mini_batch['action'],
#     #             dqn.reward:mini_batch['reward'],
#     #             dqn.next_observation:np.array(mini_batch['next_state'])
#     #         })
#     #         loss_his.append(loss)
#     #     steps +=1
#     print(i, ep_r, steps)
    
        



# # plt.plot(range(len(loss_his)), loss_his)
# # plt.show()