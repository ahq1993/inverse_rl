import tensorflow as tf
import numpy as np
from sandbox.rocky.tf.spaces.box import Box

from inverse_rl.models.fusion_manager import RamFusionDistr
from inverse_rl.models.imitation_learning import SingleTimestepIRL
from inverse_rl.models.architectures import relu_net,relu_net_dropout, linear_net
from inverse_rl.utils import TrainingIterator



class EAIRL(SingleTimestepIRL):
    """ 


    Args:
        fusion (bool): Use trajectories from old iterations to train.
        state_only (bool): Fix the learned reward to only depend on state.
        score_discrim (bool): Use log D - log 1-D as reward (if true you should not need to use an entropy bonus)
        max_itrs (int): Number of training iterations to run per fit step.
    """
    def __init__(self, env,
                 expert_trajs=None,
                 reward_arch=relu_net,
                 reward_arch_args=None,
                 score_discrim=False,
                 discount=1.0,
                 state_only=False,
                 max_itrs=100,
                 fusion=False,
                 name='eairl'):
        super(EAIRL, self).__init__()
        env_spec = env.spec
        if reward_arch_args is None:
            reward_arch_args = {}


        if fusion:
            self.fusion = RamFusionDistr(100, subsample_ratio=0.5)
        else:
            self.fusion = None
        self.dO = env_spec.observation_space.flat_dim
        self.dU = env_spec.action_space.flat_dim
        assert isinstance(env.action_space, Box)
        self.score_discrim = score_discrim
        self.gamma = discount
        self.set_demos(expert_trajs)
        self.state_only=state_only
        self.max_itrs=max_itrs

        # build energy model
        with tf.variable_scope(name) as _vs:
            # Should be batch_size x T x dO/dU
            self.obs_t = tf.placeholder(tf.float32, [None, self.dO], name='obs')
            self.nobs_t = tf.placeholder(tf.float32, [None, self.dO], name='nobs')
            self.act_t = tf.placeholder(tf.float32, [None, self.dU], name='act')
            self.nact_t = tf.placeholder(tf.float32, [None, self.dU], name='nact')
            self.labels = tf.placeholder(tf.float32, [None, 1], name='labels')
            self.lprobs = tf.placeholder(tf.float32, [None, 1], name='log_probs')
            self.lr = tf.placeholder(tf.float32, (), name='lr')
            self.vs = tf.placeholder(tf.float32, [None, 1], name='vs')
            self.vsp = tf.placeholder(tf.float32, [None, 1], name='vsp')

            with tf.variable_scope('discrim') as dvs:
                rew_input = self.obs_t
                if not self.state_only:
                    rew_input = tf.concat([self.obs_t, self.act_t], axis=1)
                with tf.variable_scope('reward'):
                    self.reward = reward_arch(rew_input, dout=1, **reward_arch_args)
                    


                # Define log p_tau(a|s) = r + gamma * V(s') - V(s)
                self.qfn = self.reward + self.gamma*self.vsp
                log_p_tau = self.reward  + self.gamma*self.vsp-self.vs


            log_q_tau = self.lprobs

            log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)
            self.discrim_output = tf.exp(log_p_tau-log_pq)
            cent_loss = -tf.reduce_mean(self.labels*(log_p_tau-log_pq) + (1-self.labels)*(log_q_tau-log_pq))

            self.loss_irl = cent_loss
            tot_loss_irl = self.loss_irl
            self.step_irl = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(tot_loss_irl)

            self._make_param_ops(_vs)

    def fit(self, paths, policy=None,empw_model=None,t_empw_model=None, batch_size=32, logger=None, lr=1e-3,**kwargs):

        if self.fusion is not None:
            old_paths = self.fusion.sample_paths(n=len(paths))
            self.fusion.add_paths(paths)
            paths = paths+old_paths

        # eval samples under current policy
        self._compute_path_probs(paths, insert=True)

        # eval expert log probs under current policy
        self.eval_expert_probs(self.expert_trajs, policy, insert=True)

        self._insert_next_state(paths)
        self._insert_next_state(self.expert_trajs)
        obs, obs_next, acts, acts_next, path_probs = \
            self.extract_paths(paths,
                               keys=('observations', 'observations_next', 'actions', 'actions_next', 'a_logprobs'))
        expert_obs, expert_obs_next, expert_acts, expert_acts_next, expert_probs = \
            self.extract_paths(self.expert_trajs,
                               keys=('observations', 'observations_next', 'actions', 'actions_next', 'a_logprobs'))


        # Train discriminator
        for it in TrainingIterator(self.max_itrs, heartbeat=5):
            nobs_batch, obs_batch, nact_batch, act_batch, lprobs_batch = \
                self.sample_batch(obs_next, obs, acts_next, acts, path_probs, batch_size=batch_size)

            nexpert_obs_batch, expert_obs_batch, nexpert_act_batch, expert_act_batch, expert_lprobs_batch = \
                self.sample_batch(expert_obs_next, expert_obs, expert_acts_next, expert_acts, expert_probs, batch_size=batch_size)

            # Build feed dict
            labels = np.zeros((batch_size*2, 1))
            labels[batch_size:] = 1.0
            obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
            nobs_batch = np.concatenate([nobs_batch, nexpert_obs_batch], axis=0)
            act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)
            nact_batch = np.concatenate([nact_batch, nexpert_act_batch], axis=0)
            lprobs_batch = np.expand_dims(np.concatenate([lprobs_batch, expert_lprobs_batch], axis=0), axis=1).astype(np.float32)
            vs=empw_model.eval(obs_batch)
            vsp=t_empw_model.eval(nobs_batch)
            feed_dict = {
                self.act_t: act_batch,
                self.obs_t: obs_batch,
                self.nobs_t: nobs_batch,
                self.nact_t: nact_batch,
                self.labels: labels,
                self.lprobs: lprobs_batch,
                self.lr: lr,
                self.vs: vs,
                self.vsp: vsp
                }


            loss_irl, _ = tf.get_default_session().run([self.loss_irl, self.step_irl], feed_dict=feed_dict)
            it.record('loss_irl', loss_irl)

            if it.heartbeat:
                print(it.itr_message())
                mean_loss_irl = it.pop_mean('loss_irl')
                print('\tLoss_irl:%f' % mean_loss_irl)

         

        return mean_loss_irl

    def next_state(self, paths,**kwargs):
        self._insert_next_state(paths)


    def eval(self, paths,empw_model=None,t_empw_model=None, **kwargs):
        """
        Return bonus
        """
        if self.score_discrim:
            self._compute_path_probs(paths, insert=True)
            obs, obs_next, acts, path_probs = self.extract_paths(paths, keys=('observations', 'observations_next', 'actions', 'a_logprobs'))
            path_probs = np.expand_dims(path_probs, axis=1)
            vs=empw_model.eval(obs).reshape(-1,1)
            vsp=empw_model.eval(obs_next).reshape(-1,1)
            path_probs = np.expand_dims(path_probs, axis=1)
            scores = tf.get_default_session().run(self.discrim_output,
                                              feed_dict={self.act_t: acts, self.obs_t: obs,
                                                         self.nobs_t: obs_next,
                                                         self.lprobs: path_probs.reshape(-1,1), self.vs:vs, self.vsp:vsp})
            score = np.log(scores) - np.log(1-scores)
            score = score[:,0]
        else:
            obs, acts = self.extract_paths(paths)
            reward = tf.get_default_session().run(self.reward,
                                              feed_dict={self.act_t: acts, self.obs_t: obs})
            score = reward[:,0]
        return self.unpack(score, paths)

    def eval_single(self, obs, acts):
        reward = tf.get_default_session().run(self.reward,
                                              feed_dict={self.act_t: acts, self.obs_t: obs})
        score = reward[:, 0]
        return score

    def debug_eval(self, paths, **kwargs):
        obs, acts = self.extract_paths(paths)
        reward, v, qfn = tf.get_default_session().run([self.reward, self.value_fn,
                                                            self.qfn],
                                                      feed_dict={self.act_t: acts, self.obs_t: obs})
        return {
            'reward': reward,
            'value': v,
            'qfn': qfn,
        }

