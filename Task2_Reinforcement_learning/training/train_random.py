from agents.ppo_agent import create_agent
from agents.chefhat_env import ChefHatEnv


def main():
    env = ChefHatEnv()
    model = create_agent(env)

    model.learn(total_timesteps=2000)
    model.save("ppo_chefhat_real")


if __name__ == "__main__":
    main()