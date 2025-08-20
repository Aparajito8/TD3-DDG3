import numpy as np
import torch
import gym
import argparse
import os
import matplotlib.pyplot as plt
import visdom
import utils
import TD3
import OurDDPG
import DDPG
from myenv import hsr
Loadmodel = False
ModelInx = 200

if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="DDPG")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="hsr")  # OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=4000, type=int)  # Time steps initial random policy is used
    parser.add_argument("--save_freq", default=200, type=int)  # How often (episode) we save model and reward curves
    parser.add_argument("--max_episodes", default=5000, type=int)  # Max episodes to run environment
    parser.add_argument("--expl_noise", default=2.0)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=128, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.001)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.01)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.0005)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_false")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = hsr()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    # if args.load_model != "":
    #     policy_file = file_name if args.load_model == "default" else args.load_model
    #     policy.load(f"./models/{policy_file}")
    ModelInx = 5000
    if Loadmodel:
        policy.load(file_name,ModelInx)

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    rewards = []
    vis = visdom.Visdom(port=5274)
    win = None
    for t in range(int(args.max_episodes)):
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        while not done:
            action = (policy.select_action(np.array(state))+
                      np.random.normal(0, max_action * args.expl_noise*max(0, args.start_timesteps -t) / args.start_timesteps, size=action_dim)).clip(-max_action, max_action)
            # Perform action
            next_state, reward, done = env.step(action)
            episode_timesteps+=1
            replay_buffer.add(state, action, next_state, reward, done)

            state = next_state
            episode_reward += reward
            # Train agent after collecting sufficient data
            if replay_buffer.size>=args.batch_size:
                policy.train(replay_buffer, args.batch_size)

            if done:
                print(f"Episode Num: {t + 1} Reward: {episode_reward:.3f}")
                rewards.append(episode_reward)

        # save model
        if (t+1)%args.save_freq==0:
            np.save(f"./results/{file_name}_{t+1}_episode",np.array(rewards))
            policy.save(file_name,t+1)
        if win is None:
            win = vis.line(X=np.arange(t, t + 1),
                           Y=np.array([np.array([episode_reward])]),
                           opts=dict(
                               ylabel='Reward',
                               xlabel='Episode',
                               title=file_name,
                            legend = ['episode_reward']))
        else:
            vis.line(X=np.array(
                [np.array(t).repeat(1)]),
                Y=np.array([np.array([episode_reward])]),
                win=win,
                update='append')