import numpy as np
import pandas as pd
import gym
# from gym_recorder.recorder import Recorder
import plotly.express as px

from agents import QLearningAgent

def train(rl_env: gym.Env, num_episodes: int, alpha, gamma, epsilon, decay):

    env_states, env_actions = rl_env.observation_space.n, rl_env.action_space.n
    agent = QLearningAgent(
        num_actions=env_actions,
        num_states=env_states,
        is_training=True,
        trained_path="saved/trainedQ.pickle",
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
        agent.save_trained("saved/trainedQ.pickle", agent.Q)

    total_rewards = np.zeros(num_episodes)
    for e in range(num_episodes):
        total_rewards[e] = np.sum(rewards_per_episode[max(0,e-100):(e+1)])
    df = pd.DataFrame(dict(
        x = np.arange(num_episodes),
        y = total_rewards
    ))
    fig = px.line(df, x="x", y="y", title="Learned rewards")
    fig.write_image("images/learned.png")
    


if __name__ == "__main__":
    num_episodes= 10000
    rl_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode=None)
    # rl_env = Recorder(rl_env, episode_num=num_episodes)
    num_actions = rl_env.action_space.n
    num_states = rl_env.observation_space.n
    train(
        rl_env=rl_env,
        num_episodes=num_episodes, 
        alpha= 0.8, 
        gamma= 0.95, 
        epsilon= 1, 
        decay = 0.0001,
        )