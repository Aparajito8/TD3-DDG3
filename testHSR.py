import numpy as np
import torch
import gym
import argparse
import os
import time
import matplotlib.pyplot as plt
import visdom
import utils
import TD3
import OurDDPG
import DDPG
from PIL import ImageGrab

from myenv import hsr
Loadmodel = True
ModelInx = 4800

if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="hsr")  # OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=4000, type=int)  # Time steps initial random policy is used
    parser.add_argument("--save_freq", default=100, type=int)  # How often (episode) we save model and reward curves
    parser.add_argument("--max_episodes", default=5000, type=int)  # Max episodes to run environment
    parser.add_argument("--expl_noise", default=2.0)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_false")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")
    if not os.path.exists("./pictures"):
        os.makedirs("./pictures")
    env = hsr()
    state_dim = env.state_dim
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    policy.load(file_name, ModelInx)
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    # q = []
    # dist = []
    # bbox = (25, 126, 870, 720)
    # im = ImageGrab.grab(bbox)
    # im.save('./pictures/{}.png'.format(file_name,env.t))
    while True:
        action = policy.select_action(np.array(state))
        next_state, reward, done = env.step(action)
        s = next_state
        # q.append(s[:6].tolist())
        # dist.append(s[-2:].tolist())
        episode_reward+=reward
        episode_timesteps += 1
        state = next_state
        # if env.t%10==0:
        #     im = ImageGrab.grab(bbox)
        #     im.save('./pictures/{}_at_{}.png'.format(file_name, env.t))
        if done:

            env.stopSim()
            # np.save(f"./{file_name}_{ModelInx}_episode_joint",np.array(q))
            # np.save(f"./{file_name}_{ModelInx}_episode_dist", np.array(dist))
            print("reward",episode_reward)
            break



