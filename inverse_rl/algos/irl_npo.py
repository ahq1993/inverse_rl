from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from inverse_rl.algos.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from inverse_rl.algos.irl_batch_polopt import IRLBatchPolopt
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf
import numpy as np
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer


class IRLNPO(IRLBatchPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            entropy_weight=1.0,
            lambda_i=1.0,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict(name='lbfgs')
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        self.pol_ent_wt = entropy_weight
        self.lambda_i = lambda_i
        super(IRLNPO, self).__init__(**kwargs)

    @overrides
    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )

        nobs_var = self.env.observation_space.new_tensor_variable(
            'nobs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = tensor_utils.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=tf.float32,
        )

        empw_var = tensor_utils.new_tensor(
            'empowerment',
            ndim=2 + is_recurrent,
            dtype=tf.float32,
        )


        input_list = [
            obs_var,
            nobs_var,
            action_var,
            advantage_var,
			empw_var,
        ]

        dist = self.policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        #dist_info_vars["mean"]=dist_info_vars["mean"]+empw_var
        q_input = tf.concat([obs_var,nobs_var],axis=1)
        q_dist_info_vars = self.qvar_model.dist_info_sym(q_input, state_info_vars)

        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)

        if self.pol_ent_wt > 0:
            if 'log_std' in dist_info_vars:
                log_std = dist_info_vars['log_std']
                ent = tf.reduce_sum(log_std + tf.log(tf.sqrt(2 * np.pi * np.e)), reduction_indices=-1)
            elif 'prob' in dist_info_vars:
                prob = dist_info_vars['prob']
                ent = -tf.reduce_sum(prob*tf.log(prob), reduction_indices=-1)
            else:
                raise NotImplementedError()
            ent = tf.stop_gradient(ent)
            adv = advantage_var + self.pol_ent_wt*ent 
        else:
            adv = advantage_var


        if is_recurrent:
            mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
            surr_loss = - tf.reduce_sum(lr * adv * valid_var) / tf.reduce_sum(valid_var)
        else:
            mean_kl = tf.reduce_mean(kl)
            surr_loss = - tf.reduce_mean(lr * adv)


        if self.train_empw:
            print("training empowerment========================================")
            pred = dist.log_likelihood(dist.sample(dist_info_vars),dist_info_vars)+empw_var
            target = dist.log_likelihood(dist.sample(q_dist_info_vars),q_dist_info_vars)
            surr_loss = surr_loss+self.lambda_i*tf.losses.mean_squared_error(predictions=pred,labels=target)




        input_list += state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        return dict()

 
    @overrides
    def optimize_policy(self, itr, samples_data):

        all_input_values = tuple(ext.extract(
            samples_data,
            "observations","observations_next", "actions", "advantages",
        ))


        obs = samples_data["observations"]
        empwr = self.empw.eval(obs)
        ep=[]
        for i in range(0,len(empwr)):
            ep.append(empwr[i][0])
        all_input_values+=(np.array(ep).reshape(-1,1)),
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        logger.log("Computing loss before")
        loss_before = self.optimizer.loss(all_input_values)
        logger.log("Computing KL before")
        mean_kl_before = self.optimizer.constraint_val(all_input_values)
        logger.log("Optimizing")
        self.optimizer.optimize(all_input_values)
        logger.log("Computing KL after")
        mean_kl = self.optimizer.constraint_val(all_input_values)
        logger.log("Computing loss after")
        loss_after = self.optimizer.loss(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy_params=self.policy.get_param_values(),
            irl_params=self.get_irl_params(),
            empw_params=self.get_empw_params(),
            qvar_params=self.get_qvar_params(),
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
