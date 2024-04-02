import numpy as np
import pandas as pd
import gym
# from gym_recorder.recorder import Recorder
import plotly.express as px
from environments import FrozenLake

from agents import QLearningAgent

def run(rl_env: gym.Env, num_episodes: int, is_training: bool, alpha: float, gamma: float, epsilon: float, decay: float, trained_weights_path: str | None):

    env_states, env_actions = rl_env.observation_space.n, rl_env.action_space.n
    agent = QLearningAgent(
        num_actions=env_actions,
        num_states=env_states,
        is_training=is_training,
        trained_path=trained_weights_path,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=decay
        )
    rewards_per_episode = np.zeros(num_episodes)
    for i in range(num_episodes):
        state = rl_env.reset()[0]
        terminated = False
        truncated = False

        while (not terminated and not truncated):
            action = agent.getAction(state, rl_env.action_space)
            # print(f"action: {action}")

            new_state, reward, terminated, truncated, _  = rl_env.step(action)

            agent.update(action=action, state=state, reward=reward, new_state=new_state)

            state = new_state
            rewards_per_episode[i] += reward

            # rl_env.txtqueue.append(f"Episode:{i}")
        print(f"Episode:{i} | total-rewards:{rewards_per_episode[i]}")
        agent.update_epsilon()

        if agent.epsilon == 0:
            agent.alpha = 0.0001

    
    rl_env.close()
    if agent.is_training:
        agent.save_trained(trained_weights_path, agent.Q)

    total_rewards = np.zeros(num_episodes)
    for e in range(num_episodes):
        total_rewards[e] = np.sum(rewards_per_episode[max(0,e-100):(e+1)])
    df = pd.DataFrame(dict(
        x = np.arange(num_episodes),
        y = total_rewards
    ))
    fig = px.line(df, x="x", y="y", title="Learned rewards", labels={"y": "cumulative-rewards per 100 episodes", "x": "Episodes"})
    fig.write_image("images/learned.png")
    


if __name__ == "__main__":
    num_episodes= 15000
    rl_env = FrozenLake(map_name="8x8", is_slippery=False, render_mode="human").frozen_lake
    # rl_env = Recorder(rl_env, episode_num=num_episodes)
    num_actions = rl_env.action_space.n
    num_states = rl_env.observation_space.n
    run(
        rl_env = rl_env,
        num_episodes = num_episodes,
        is_training = True, 
        alpha = 0.9, 
        gamma = 0.9, 
        epsilon =1, 
        decay = 0.0001,
        trained_weights_path = "saved/trainedQ.pickle"
        )