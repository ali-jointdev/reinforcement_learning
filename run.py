from main import Environment
from stable_baselines.common import make_vec_env
from stable_baselines import A2C
from stable_baselines.common.policies import MlpPolicy

if __name__ == '__main__':
    train = True
    if train:
        env = make_vec_env(Environment, n_envs=4)
        model = A2C(MlpPolicy, env, verbose=1)
        model.learn(total_timesteps=1000000)
        model.save('a2c')
    model = A2C.load('a2c')
    env = Environment()
    env.start(model)
