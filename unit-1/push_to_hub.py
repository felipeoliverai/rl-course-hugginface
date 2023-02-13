import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from huggingface_sb3 import package_to_hub
import torch
from huggingface_hub import \
    notebook_login  # To log to our Hugging Face account to be able to upload models to the Hub.
from huggingface_sb3 import load_from_hub, package_to_hub, push_to_hub
from pyvirtualdisplay import Display
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy



def push_to_hub_func():
    # PLACE the variables you've just defined two cells above
    # Define the name of the environment

    model = PPO.load("ppo-LunarLander-v2.zip")
    model_name = "ppo-LunarLander-v2" 
    env_id = "LunarLander-v2"

    # TODO: Define the model architecture we used
    model_architecture = "PPO"

    ## Define a repo_id
    ## repo_id is the id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
    ## CHANGE WITH YOUR REPO ID
    repo_id = "Felipe474/ppo-LunarLander-v2" # Change with your repo id, you can't push with mine 😄

    ## Define the commit message
    commit_message = "Upload PPO LunarLander-v2 trained agent"

    # Create the evaluation env
    eval_env = DummyVecEnv([lambda: gym.make(env_id)])

    # PLACE the package_to_hub function you've just filled here
    package_to_hub(model=model, # Our trained model
                model_name=model_name, # The name of our trained model 
                model_architecture=model_architecture, # The model architecture we used: in our case PPO
                env_id=env_id, # Name of the environment
                eval_env=eval_env, # Evaluation Environment
                repo_id=repo_id, # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
                commit_message=commit_message)


if __name__ == "__main__":
    push_to_hub_func()