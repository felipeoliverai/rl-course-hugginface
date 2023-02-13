import gym

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy



def main():
    # Retrieve the model from the hub
    ## repo_id =  id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
    ## filename = name of the model zip file from the repository
    checkpoint = load_from_hub(repo_id="Felipe474/ppo-LunarLander-v2", filename="ppo-LunarLander-v2.zip")
    model = PPO.load(checkpoint)

    # Evaluate the agent
    eval_env = gym.make('LunarLander-v2')
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    
    # Watch the agent play
    obs = eval_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs)
        obs, reward, done, info = eval_env.step(action)
        eval_env.render()
        if done:
            obs = eval_env.reset()
    eval_env.close()


if __name__ == "__main__":
    main()