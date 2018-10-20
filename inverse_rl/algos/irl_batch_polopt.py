import time

from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
import rllab.plotter as plotter
from sandbox.rocky.tf.policies.base import Policy
import tensorflow as tf
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
import numpy as np
from collections import deque

from inverse_rl.utils.hyperparametrized import Hyperparametrized
import pickle

class IRLBatchPolopt(RLAlgorithm, metaclass=Hyperparametrized):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
			empw,
			tempw,
            qvar_model,
            baseline,
            scope=None,
            n_itr=500,
            start_itr=0,
            batch_size=5000,
            max_path_length=500,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=True,
            whole_paths=True,
            fixed_horizon=False,
            sampler_cls=None,
            sampler_args=None,
            force_batch_sampler=False,
            init_pol_params = None,
            irl_model=None,
            irl_model_wt=1.0,
            discrim_train_itrs=10,
            zero_environment_reward=False,
            init_irl_params=None,
            init_empw_params=None,
            init_qvar_params=None,
            target_empw_update=5,
            train_irl=True,
            train_empw=True,
            train_qvar=True,
            key='',
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param empw: Empowerment (Potential Function)
        :param tempw: Target empowerment (Potential Function)
        :param qvar_model: Inverse Model q(a|s,s')
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param target_empw_update: Update tempw after every target_empw_update iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.env = env
        self.policy = policy
        self.empw = empw
        self.tempw = tempw
        self.qvar_model = qvar_model
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.init_pol_params = init_pol_params
        self.init_irl_params = init_irl_params
        self.init_empw_params = init_empw_params
        self.init_qvar_params = init_qvar_params
        self.irl_model = irl_model
        self.irl_model_wt = irl_model_wt
        self.no_reward = zero_environment_reward
        self.discrim_train_itrs = discrim_train_itrs
        self.train_irl = train_irl
        self.target_empw_update = target_empw_update
        self.train_empw = train_empw
        self.train_qvar = train_qvar
        self.__irl_params = None
        self.__empw_params = None
        self.__qvar_params = None

        if self.irl_model_wt > 0:
            assert self.irl_model is not None, "Need to specify a IRL model"

        if sampler_cls is None:
            if self.policy.vectorized and not force_batch_sampler:
                print('using vec sampler')
                sampler_cls = VectorizedSampler
            else:
                print('using batch sampler')
                sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()
        self.sampler = sampler_cls(self, **sampler_args)
        self.init_opt()

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr):
        return self.sampler.obtain_samples(itr)

    def process_samples(self, itr, paths):
        return self.sampler.process_samples(itr, paths,irl=True)

    def log_avg_returns(self, paths):
        undiscounted_returns = [sum(path["rewards"]) for path in paths]
        avg_return = np.mean(undiscounted_returns)
        return avg_return

    def get_irl_params(self):
        return self.__irl_params

    def get_empw_params(self):
        return self.__empw_params

    def get_qvar_params(self):
        return self.__qvar_params

    def compute_irl(self, paths, itr=0):
        r=0
        if self.no_reward:
            tot_rew = 0
            for path in paths:
                tot_rew += np.sum(path['rewards'])
                path['rewards'] *= 0
            logger.record_tabular('OriginalTaskAverageReturn', tot_rew/float(len(paths)))
            r=tot_rew/float(len(paths))

        if self.irl_model_wt <=0:
            return paths

        if self.train_irl:
            max_itrs = self.discrim_train_itrs
            lr=1e-3
            mean_loss_irl= self.irl_model.fit(paths, policy=self.policy,empw_model=self.empw,t_empw_model=self.tempw, itr=itr, max_itrs=max_itrs, lr=lr,
                                           logger=logger)

            logger.record_tabular('IRLLoss', mean_loss_irl)
            self.__irl_params = self.irl_model.get_params()
        else:
            self.irl_model.next_state(paths)
        probs = self.irl_model.eval(paths,empw_model=self.empw,t_empw_model=self.tempw, gamma=self.discount, itr=itr)

        logger.record_tabular('IRLRewardMean', np.mean(probs))
        logger.record_tabular('IRLRewardMax', np.max(probs))
        logger.record_tabular('IRLRewardMin', np.min(probs))


        if self.irl_model.score_trajectories:
            # TODO: should I add to reward here or after advantage computation?
            for i, path in enumerate(paths):
                path['rewards'][-1] += self.irl_model_wt * probs[i]
        else:
            for i, path in enumerate(paths):
                path['rewards'] += self.irl_model_wt * probs[i]
        return paths,r

    def compute_qvar(self, paths, itr=0):


        if self.train_qvar:
            max_itrs = self.discrim_train_itrs
            lr=1e-3
            mean_loss_q = self.qvar_model.fit(paths, itr=itr, max_itrs=max_itrs, lr=lr,logger=logger)

            logger.record_tabular('Qvar_Loss', mean_loss_q)
            self.__qvar_params = self.qvar_model.get_params()


    def compute_empw(self, paths, itr=0):
        if self.train_empw:
            max_itrs = self.discrim_train_itrs
            lr=1e-3
            mean_loss_empw = self.empw.fit(paths, irl_model=self.irl_model, tempw=self.tempw, policy=self.policy, qvar_model=self.qvar_model, itr=itr, max_itrs=max_itrs, lr=lr,logger=logger)

            logger.record_tabular('EmpwLoss', mean_loss_empw)
            self.__empw_params = self.empw.get_params()



        

    def train(self):
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        if self.init_pol_params is not None:
            self.policy.set_param_values(self.init_pol_params)

        if self.init_qvar_params is not None:
            self.qvar_model.set_params(self.init_qvar_params)

        if self.init_irl_params is not None:
            self.irl_model.set_params(self.init_irl_params)

        if self.init_empw_params is not None:
            self.empw.set_params(self.init_empw_params)

        self.start_worker()
        start_time = time.time()

        returns = []
        rew = [] # stores score at each step
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()

            with logger.prefix('itr #%d | ' % itr):
                logger.log("Obtaining samples...")
                paths = self.obtain_samples(itr)

                logger.log("Processing samples...")
                paths,r = self.compute_irl(paths, itr=itr)
                rew.append(r)
                returns.append(self.log_avg_returns(paths))
                self.compute_qvar(paths, itr=itr)
                self.compute_empw(paths, itr=itr)
                samples_data = self.process_samples(itr, paths)
                logger.log("Logging diagnostics...")
                self.log_diagnostics(paths)
                logger.log("Optimizing policy...")
                self.optimize_policy(itr, samples_data)
                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                if self.store_paths:
                    params["paths"] = samples_data["paths"]
                logger.save_itr_params(itr, params)
                logger.log("Saved")
                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    self.update_plot()
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")
            if itr%self.target_empw_update==0 and self.train_empw:  #reward 5
                print('updating target empowerment parameters')
                self.tempw.set_params(self.__empw_params)


        #pickle.dump(rew, open("rewards.p", "wb" )) # uncomment to store rewards in every iteration
        self.shutdown_worker()
        return

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError


    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)
