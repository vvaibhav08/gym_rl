import numpy as np
import pandas as pd

# from gym_recorder.recorder import Recorder
import plotly.express as px
from environments import LunarLander

from agents import QLearningAgent


def run(
    rl_env,
    num_episodes: int,
    is_training: bool,
    alpha: float,
    gamma: float,
    epsilon: float,
    decay: float,
    trained_weights_path: str | None,
):
    env_states, env_actions = rl_env.get_num_discrete_states(), rl_env.gym_env.action_space.n
    agent = QLearningAgent(
        num_actions=env_actions,
        num_states=env_states,
        is_training=is_training,
        trained_path=trained_weights_path,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=decay,
    )

    rewards_per_episode = np.zeros(num_episodes)
    if rl_env.gym_env.spec.id == "MountainCar-v0":
        reward_threshold = -1000
    elif rl_env.gym_env.spec.id == "LunarLander-v2":
        reward_threshold = -200
    for i in range(num_episodes):
        state = rl_env.gym_env.reset()[0]
        if isinstance(state, np.ndarray):
            state_elements = tuple(
                np.digitize(element, rl_env.state_discrete_space[i]) for i, element in enumerate(state)
            )

        terminated = False
        truncated = False

        if isinstance(state, np.ndarray):
            while not terminated and rewards_per_episode[i] > reward_threshold:
                action = agent.getAction(state_elements, rl_env.gym_env.action_space)
                new_state, reward, terminated, truncated, _ = rl_env.gym_env.step(action)
                new_state_elements = tuple(
                    np.digitize(element, rl_env.state_discrete_space[i]) for i, element in enumerate(new_state)
                )
                if any(ns >= max_idx for ns, max_idx in zip(new_state_elements, env_states)):
                    break
                agent.update(action=action, state=state_elements, reward=reward, new_state=new_state_elements)
                state = new_state
                state_elements = new_state_elements

                rewards_per_episode[i] += reward
        else:
            while not terminated and not truncated:
                action = agent.getAction(state, rl_env.gym_env.action_space)
                new_state, reward, terminated, truncated, _ = rl_env.gym_env.step(action)
                agent.update(action=action, state=state, reward=reward, new_state=new_state)
                state = new_state

                rewards_per_episode[i] += reward

        if i % 100 == 0:
            print(f"Episode:{i} | total-rewards:{rewards_per_episode[i]}")
        agent.update_epsilon()

        if agent.epsilon == 0:
            agent.alpha = 0.0001

    rl_env.gym_env.close()
    if agent.is_training:
        agent.save_trained(trained_weights_path, agent.Q)

        # plot training progress
        total_rewards = np.zeros(num_episodes)
        for e in range(num_episodes):
            total_rewards[e] = np.sum(rewards_per_episode[max(0, e - 100) : (e + 1)])
        df = pd.DataFrame(dict(x=np.arange(num_episodes), y=total_rewards))
        fig = px.line(
            df,
            x="x",
            y="y",
            title="Learned rewards",
            labels={"y": "cumulative-rewards per 100 episodes", "x": "Episodes"},
        )
        image_file = trained_weights_path.split("/")[1].split(".")[0] + ".png"
        fig.write_image(f"images/{image_file}")


if __name__ == "__main__":
    num_episodes = 5000
    # rl_env = FrozenLake(map_name="8x8", is_slippery=False, render_mode=None)
    # rl_env = MountainCar(render_mode=None)
    rl_env = LunarLander(render_mode="human")
    # rl_env = Recorder(rl_env, episode_num=num_episodes)
    FL_params = {"alpha": 0.9, "gamma": 0.9, "epsilon": 1, "decay": 0.0001, "num_episodes": 10000}
    MC_params = {"alpha": 0.9, "gamma": 0.9, "epsilon": 1, "decay": 0.0001, "num_episodes": 10000}
    LL_params = {"alpha": 0.1, "gamma": 0.9, "epsilon": 1, "decay": 0.00002, "num_episodes": 51000}
    run(
        rl_env=rl_env,
        num_episodes=15,  # LL_params["num_episodes"],
        is_training=False,
        alpha=LL_params["alpha"],
        gamma=LL_params["gamma"],
        epsilon=LL_params["epsilon"],
        decay=LL_params["decay"],
        trained_weights_path=f"saved/trainedQ_lunarlander_{LL_params['num_episodes']}.pickle",
    )
