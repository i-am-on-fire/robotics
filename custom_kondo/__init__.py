from gymnasium.envs.registration import register

register(id='CustomKondo-v0', # id by which to refer to the new environment; the string is passed as an argument to gym.make() to create a copy of the environment
	entry_point='custom_kondo.envs.kondo_env:HumanoidEnv' # points to the class that inherits from gym.Env and defines the four basic functions, i.e. reset, step, render, close
)