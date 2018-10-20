import tensorflow as tf

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv

from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.algos.irl_trpo import IRLTRPO
from sandbox.rocky.tf.policies.gaussian_mlp_inverse_policy import GaussianMLPInversePolicy
from inverse_rl.models.eairl import *
from inverse_rl.models.qvar import *
from inverse_rl.models.empowerment import *
from inverse_rl.models.architectures import relu_net
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts, load_latest_experts_multiple_runs
from inverse_rl.utils.hyper_sweep import run_sweep_parallel, run_sweep_serial


def main(exp_name=None, fusion=True):
    env = TfEnv(CustomGymEnv('CustomAnt-v0', record_video=False, record_log=False))
    # load ~2 iterations worth of data from each forward RL experiment as demos
    experts = load_latest_experts_multiple_runs('data/ant_data_collect', n=2)
    #experts = load_latest_experts('data/ant_data_collect', n=5)

    #qvar: inverse model q(a|s,s')
    qvar= GaussianMLPInversePolicy(name='qvar_model', env_spec=env.spec, hidden_sizes=(32, 32))
    qvar_model = Qvar(env=env,qvar=qvar, expert_trajs=experts, fusion=True, max_itrs=10)
    #Empowerment-based Adversarial Inverse Reinforcement Learning, set score_discrim=True 
    irl_model = EAIRL(env=env, expert_trajs=experts, state_only=False, fusion=fusion, max_itrs=10, score_discrim=True)

    #Empowerment-based potential functions gamma* Phi(s')-Phi(s)
    empw_model = Empowerment(env=env,fusion=True, max_itrs=4)
    t_empw_model = Empowerment(env=env,scope='t_efn',fusion=True, max_itrs=2, name='empowerment2')


    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
    algo = IRLTRPO(
        env=env,
        policy=policy,
		empw=empw_model,
		tempw=t_empw_model,
        qvar_model=qvar_model,
        irl_model=irl_model,
        n_itr=130,
        batch_size=20000,
        max_path_length=500,
        discount=0.99,
        store_paths=True,
        target_empw_update=5,
        irl_model_wt=1.0,
        entropy_weight=0.1,
        lambda_i=1.0,
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec),
    )
    with rllab_logdir(algo=algo, dirname='data/ant_state_irl'):
    #with rllab_logdir(algo=algo, dirname='data/ant_state_irl/%s' % exp_name): # if you use multiple runs, use this line instead of above
        with tf.Session():
            algo.train()

if __name__ == "__main__":
    params_dict = {
        'fusion': [True]
    }
    main()
    #run_sweep_parallel(main, params_dict, repeat=3)

