import tensorflow as tf
import numpy as np
from sandbox.rocky.tf.spaces.box import Box

from inverse_rl.models.fusion_manager import RamFusionDistr
from inverse_rl.models.imitation_learning import SingleTimestepIRL
from inverse_rl.models.architectures import relu_net_dropout,relu_net
from inverse_rl.utils import TrainingIterator



class Empowerment(SingleTimestepIRL):
    """ 


    Args:
        fusion (bool): Use trajectories from old iterations to train.
        state_only (bool): Fix the learned reward to only depend on state.
        max_itrs (int): Number of training iterations to run per fit step.
    """
    def __init__(self, env,
                 emp_fn_arch=relu_net,#_dropout,
                 scope='efn',
                 max_itrs=100,
                 fusion=False,
                 name='empowerment'):
        super(Empowerment, self).__init__()
        env_spec = env.spec

        if fusion:
            self.fusion = RamFusionDistr(100, subsample_ratio=0.5)
        else:
            self.fusion = None
        self.dO = env_spec.observation_space.flat_dim
        self.dU = env_spec.action_space.flat_dim
        assert isinstance(env.action_space, Box)
        assert emp_fn_arch is not None
       	self.max_itrs=max_itrs

        # build energy model
        with tf.variable_scope(name) as _vs:
            # Should be batch_size x T x dO/dU
            self.obs_t = tf.placeholder(tf.float32, [None, self.dO], name='obs')
            self.act_qvar = tf.placeholder(tf.float32, [None, 1], name='act_qvar')
            self.act_policy = tf.placeholder(tf.float32, [None, 1], name='act_policy') 
            self.lr = tf.placeholder(tf.float32, (), name='lr')

            # empowerment function 
            with tf.variable_scope(scope):
                self.empwerment = emp_fn_arch(self.obs_t, dout=1)

            cent_loss = tf.losses.mean_squared_error(predictions=(self.empwerment+self.act_policy),labels=self.act_qvar) 

            self.loss_emp = cent_loss
            tot_loss_emp = self.loss_emp
            self.step_emp = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(tot_loss_emp)

            self._make_param_ops(_vs)

    def fit(self, paths,irl_model=None, tempw=None, policy=None, qvar_model=None, batch_size=32, logger=None, lr=1e-3,**kwargs):

        if self.fusion is not None:
            old_paths = self.fusion.sample_paths(n=len(paths))
            self.fusion.add_paths(paths)
            paths = paths+old_paths

        #self._insert_next_state(paths)
        obs,ac, obs_next= self.extract_paths(paths,keys=('observations', 'actions', 'observations_next'))

        for it in TrainingIterator(self.max_itrs, heartbeat=1):

            nobs_batch, obs_batch, ac_batch = self.sample_batch(obs_next, obs, ac)
            dist_info_vars = policy.dist_info_sym(obs_batch, None)

            dist_s =policy.distribution.log_likelihood(policy.distribution.sample(dist_info_vars),dist_info_vars)


            q_input = tf.concat([obs_batch,nobs_batch],axis=1)
            q_dist_info_vars = qvar_model.dist_info_sym(q_input, None)

            q_dist_s =policy.distribution.log_likelihood(policy.distribution.sample(q_dist_info_vars),q_dist_info_vars)

            # Build feed dict
            feed_dict = {
                self.obs_t: obs_batch,
                self.act_qvar:q_dist_s.eval(),
                self.act_policy:dist_s.eval(),
                self.lr: lr
                }

            loss_emp, _ = tf.get_default_session().run([self.loss_emp, self.step_emp], feed_dict=feed_dict)
            it.record('loss_emp', loss_emp)
            if it.heartbeat:
                print(it.itr_message())
                mean_loss_emp = it.pop_mean('loss_emp')
                print('\tLoss_emp:%f' % mean_loss_emp)


        return mean_loss_emp

    def eval(self, obs, **kwargs):
        """
        Return bonus
        """
 
        empw = tf.get_default_session().run(self.empwerment,
                                          feed_dict={self.obs_t: obs})
        return empw




