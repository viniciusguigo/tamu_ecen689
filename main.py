import gym

from agents import RandomAgent

env = gym.make('CarRacing-v0')
agent = RandomAgent(env)

# number of episodes
n_epis = 1

for i_episode in range(n_epis):
    observation = env.reset()
    t = 0

    while True:
        env.render()
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        t += 1

# close env
env.close()
