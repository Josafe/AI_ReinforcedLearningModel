import gymnasium as gym
from src.agent import Agent
from src.replay_buffer import ReplayBuffer


def train():
    env = gym.make("CartPole-v1")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = Agent(state_dim, action_dim)
    buffer = ReplayBuffer()

    episodes = 100

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        print(f"\n[EPISODE {episode}] START")

        done = False
        while not done:
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)
            
            agent.train_step(buffer)

            state = next_state
            total_reward += reward

        agent.update_target()
        agent.epsilon *= 0.95

        print(f"[EPISODE {episode}] Total Reward: {total_reward}")

    env.close()