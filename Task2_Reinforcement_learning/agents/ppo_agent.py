from stable_baselines3 import PPO


def create_agent(env):
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=32,      # small rollout for speed
        batch_size=32,
    )
    return model