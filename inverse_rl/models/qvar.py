import tensorflow as tf
import numpy as np
from sandbox.rocky.tf.spaces.box import Box

from inverse_rl.models.fusion_manager import RamFusionDistr
from inverse_rl.models.imitation_learning import SingleTimestepIRL
from inverse_rl.models.architectures import relu_net, linear_net
from inverse_rl.utils import TrainingIterator



class Qvar(SingleTimestepIRL):
    """ 


    Args:
        fusion (bool): Use trajectories from old iterations to train.
        state_only (bool): Fix the learned reward to only depend on state.
        score_discrim (bool): Use log D - log 1-D as reward (if true you should not need to use an entropy bonus)
        max_itrs (int): Number of training iterations to run per fit step.
    """
    def __init__(self, env,
                 expert_trajs=None,
                 qvar=None,
                 score_discrim=False,
                 discount=1.0,
                 state_only=False,
                 max_itrs=100,
                 fusion=False,
                 name='qvar'):
        super(Qvar, self).__init__()
        env_spec = env.spec
        if qvar is not None:
            self.qvar = qvar

        if fusion:
            self.fusion = RamFusionDistr(100, subsample_ratio=0.5)
        else:
            self.fusion = None
        self.dO = env_spec.observation_space.flat_dim
        self.dU = env_spec.action_space.flat_dim
        assert isinstance(env.action_space, Box)
        self.set_demos(expert_trajs)
        self.max_itrs=max_itrs

        # build energy model
        with tf.variable_scope(name) as _vs:
            # Should be batch_size x T x dO/dU
            self.act_t = tf.placeholder(tf.float32, [None, self.dU], name='act')
            self.obs_t = tf.placeholder(tf.float32, [None, self.dO], name='obs')
            self.nobs_t = tf.placeholder(tf.float32, [None, self.dO], name='nobs')
            self.lr = tf.placeholder(tf.float32, (), name='lr') 

            with tf.variable_scope('q_var') as dvs:
                q_input = tf.concat([self.obs_t,self.nobs_t],axis=1)
                self.act_predicted=self.qvar.dist_info_sym(q_input,None)

            self.loss_q = tf.losses.mean_squared_error(predictions=self.act_predicted["mean"],labels=self.act_t)
            tot_loss_q = self.loss_q


            self.step_q = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(tot_loss_q)
            self._make_param_ops(_vs)

    def fit(self, paths, batch_size=32, logger=None, lr=1e-3,**kwargs):

        if self.fusion is not None:
            old_paths = self.fusion.sample_paths(n=len(paths))
            self.fusion.add_paths(paths)
            paths = paths+old_paths


        obs, obs_next, acts = \
            self.extract_paths(paths,
                               keys=('observations', 'observations_next', 'actions'))
        '''expert_obs, expert_obs_next, expert_acts = \
            self.extract_paths(self.expert_trajs,
                               keys=('observations', 'observations_next', 'actions'))'''


        # Train discriminator
        for it in TrainingIterator(self.max_itrs, heartbeat=5):
            nobs_batch, obs_batch, act_batch = \
                self.sample_batch(obs_next, obs, acts, batch_size=batch_size)


            feed_dict = {
                self.act_t: act_batch,
                self.obs_t: obs_batch,
                self.nobs_t: nobs_batch,
                self.lr: lr
                }

            loss_q, _ = tf.get_default_session().run([self.loss_q, self.step_q], feed_dict=feed_dict)
            it.record('loss_q', loss_q)
            if it.heartbeat:
                mean_loss_q = it.pop_mean('loss_q')
                print('\tLoss_q:%f' % mean_loss_q)


        return mean_loss_q

    def dist_info_sym(self, q_input,state_info_vars):
        return self.qvar.dist_info_sym(q_input,None)

    '''def set_params(self, params):
        return self.qvar.set_param_values(params)'''

