from Agent import *
import gym

env = gym.make('CartPole-v0')

training_ep = 8000

agent = Sarsa(env.action_space.n, reward_decay=1, total_step=training_ep)

max_step = 0

for i in range(training_ep):
    obs = env.reset()
    done = False
    step = 0
    act = agent.choose_action(obs, i)
    while not done:
        obs_, reward, done, _ = env.step(act)
        env.render()
        act_ = agent.choose_action(obs_, i)
        agent.learn(obs, reward, act, obs_, act)
        # print(q_learn.q_tabel)
        obs = obs_
        step += 1

    if step > max_step:
        max_step = step
        print('new max_step: {}'.format(max_step))

print(agent.q_tabel)

# eval
while 1:
    obs = env.reset()
    done = False
    act = agent.choose_action_val(obs)
    x = env.step(act)
    env.render()

# env.close()