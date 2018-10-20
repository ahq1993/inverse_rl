import tensorflow as tf
import os

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv


from inverse_rl.algos.irl_trpo import IRLTRPO
from sandbox.rocky.tf.policies.gaussian_mlp_inverse_policy import GaussianMLPInversePolicy
from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.models.eairl import *
from inverse_rl.models.empowerment import *
from inverse_rl.models.qvar import *
from inverse_rl.models.tf_util import load_prior_params
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts
from inverse_rl.utils.hyper_sweep import run_sweep_parallel, run_sweep_serial


DATA_DIR = 'data/ant_state_irl'
def main(exp_name=None, params_folder='data/ant_state_irl'):
    #env = TfEnv(CustomGymEnv('PointMazeLeft-v0', record_video=True, record_log=True,force_reset=True))
    env = TfEnv(CustomGymEnv('DisabledAnt-v0', record_video=False, record_log=False,force_reset=False))

    irl_itr=90# earlier IRL iterations overfit less; either 80 or 90 seems to work well. But I usually search through 60,65,70,75, .. uptil 100
    #params_file = os.path.join(DATA_DIR, '%s/itr_%d.pkl' % (params_folder, irl_itr))
    params_file = os.path.join(DATA_DIR, 'itr_%d.pkl' % (irl_itr))
    prior_params = load_prior_params(params_file)


    '''q_itr = 400  # earlier IRL iterations overfit less; 100 seems to work well.
    #params_file = os.p90ath.join(DATA_DIR, '%s/itr_%d.pkl' % (params_folder, irl_itr))
    params_file = os.path.join(DATA_DIR, 'itr_%d.pkl' % (q_itr))
    prior_params_q = load_prior_params(params_file)'''

    qvar = GaussianMLPInversePolicy(name='qvar_model', env_spec=env.spec, hidden_sizes=(32, 32))
    qvar_model = Qvar(env=env,qvar=qvar, expert_trajs=None,max_itrs=10)
    irl_model = EAIRL(env=env, expert_trajs=None, state_only=False, score_discrim=False)
    empw_model = Empowerment(env=env,max_itrs=1)
    t_empw_model = Empowerment(env=env,scope='t_efn', max_itrs=2, name='empowerment2')



    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))

    algo = IRLTRPO(
        init_irl_params=prior_params['irl_params'],
        init_empw_params=None,#prior_params['empw_params'],
        init_qvar_params=None,#prior_params['qvar_params'],
        init_policy_params=None,#prior_params['policy_params'],
        env=env,
        policy=policy,
		empw=empw_model,
		tempw=t_empw_model,
        qvar_model=qvar_model,
        irl_model=irl_model,
        n_itr=2000,
        batch_size=20000,
        max_path_length=500,
        discount=0.99,
        store_paths=False,
        train_irl=False,
        train_empw=False,
        train_qvar=False,
        irl_model_wt=1.0,
        entropy_weight=0.1,
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec),
        log_params_folder=params_folder,
        log_experiment_name=exp_name,
    )

    with rllab_logdir(algo=algo, dirname='data/ant_transfer'):#%s'%exp_name):
    #with rllab_logdir(algo=algo, dirname='data/ant_transfer%s'%exp_name):
        with tf.Session():
            algo.train()

if __name__ == "__main__":
    import os
    params_folders = os.listdir(DATA_DIR)
    params_dict = {
        'params_folder': params_folders,
    }
    main()
    #run_sweep_parallel(main, params_dict, repeat=3)

