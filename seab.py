

if __name__ == '__main__':
    import gym
    import time
    env = gym.make('Humanoid-v1')
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action
        time.sleep(0.1)
    env.close()