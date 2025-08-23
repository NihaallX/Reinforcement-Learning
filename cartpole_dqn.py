"""
Reinforcement Learning with Gymnasium and Stable-Baselines3
Train and test a DQN agent on CartPole-v1.
"""

import gymnasium as gym
from stable_baselines3 import DQN
import torch
import matplotlib.pyplot as plt
import os


def train_agent(env_id="CartPole-v1", timesteps=100_000, model_path="dqn_cartpole", plot_path="reward_curve.png"):
    """
    Train a DQN agent, save the model, and plot rewards.
    Args:
        env_id (str): Gymnasium environment ID.
        timesteps (int): Number of training timesteps.
        model_path (str): Path to save the trained model.
        plot_path (str): Path to save the reward curve plot.
    """
    env = gym.make(env_id)
    model = DQN("MlpPolicy", env, verbose=1)
    episode_rewards = []
    obs, info = env.reset()
    total_reward = 0
    for step in range(1, timesteps + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        if done:
            episode_rewards.append(total_reward)
            obs, info = env.reset()
            total_reward = 0
        # Train the model every step
        model.learn(total_timesteps=1, reset_num_timesteps=False, progress_bar=False)
    model.save(model_path)
    env.close()
    print(f"Model saved to {model_path}")
    plot_rewards(episode_rewards, plot_path)
    print(f"Reward curve saved to {plot_path}")
    return episode_rewards


def test_agent(env_id="CartPole-v1", model_path="dqn_cartpole", episodes=5, record_video=True, video_folder="videos"):
    """
    Load a trained DQN agent, run test episodes with rendering, and optionally record video.
    Args:
        env_id (str): Gymnasium environment ID.
        model_path (str): Path to the trained model.
        episodes (int): Number of test episodes to run.
        record_video (bool): Whether to record a video of the agent.
        video_folder (str): Folder to save the video.
    """
    render_mode = "rgb_array" if record_video else "human"
    env = gym.make(env_id, render_mode=render_mode)
    if record_video:
        os.makedirs(video_folder, exist_ok=True)
        env = gym.wrappers.RecordVideo(env, video_folder=video_folder, episode_trigger=lambda ep: True)
        print(f"Recording video(s) to {video_folder}/")
    else:
        env = gym.make(env_id, render_mode="human")
    model = DQN.load(model_path)
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Episode {ep+1}: Reward = {total_reward}")
    env.close()
def plot_rewards(rewards, plot_path="reward_curve.png"):
    """
    Plot and save the episode rewards curve.
    Args:
        rewards (list): List of episode rewards.
        plot_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN on CartPole-v1: Episode Reward Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train or test a DQN agent on CartPole-v1.")
    parser.add_argument("mode", choices=["train", "test", "plot"], help="Mode: train, test, or plot rewards.")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Training timesteps (default: 100000)")
    parser.add_argument("--episodes", type=int, default=5, help="Test episodes (default: 5)")
    parser.add_argument("--no-video", action="store_true", help="Disable video recording during test.")
    args = parser.parse_args()
    if args.mode == "train":
        rewards = train_agent(timesteps=args.timesteps)
    elif args.mode == "test":
        test_agent(episodes=args.episodes, record_video=not args.no_video)
    elif args.mode == "plot":
        # Example: plot from a saved rewards file (not implemented)
        print("Plotting mode is not implemented. Use train mode to plot rewards.")

if __name__ == "__main__":
    main()
