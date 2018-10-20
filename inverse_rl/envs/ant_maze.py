from rllab.envs.mujoco.maze.maze_env import MazeEnv
from inverse_rl.envs.ant_env import CustomAntEnv


class AntMazeEnv(MazeEnv):

    MODEL_CLASS = CustomAntEnv
    ORI_IND = 6

    MAZE_HEIGHT = 2
    MAZE_SIZE_SCALING = 3.0
