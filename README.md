# Variation Inverse Reinforcement Learning
Implementation of [Adversarial Imitation Via Variational Inverse Reinforcement Learning](https://sites.google.com/view/eairl).  

The code is an adaption of [inverse-rl](https://github.com/justinjfu/inverse_rl) repository that contains the implementations of state-of-the-art imitation & inverse reinforcement learning algorithms.

## Requirements
* [Rllab](https://github.com/openai/rllab)
	* Use our base.py by replacing ```from rllab.sampler.base import BaseSampler``` to ```from base import BaseSampler```  in the file ```sandbox/rocky/tf/samplers/vectorized_sampler.py```
	* Include our gaussian_mlp_inverse_policy.py to the folder ```sandbox/rocky/tf/policies/```
* [TensorFlow](https://www.tensorflow.org)
## Examples

### Running the Ant gym environment
1. Collect expert data
	
    ```python ant_data_collect.py```
2. Run Inverse Reinforcement Learning:
	
    ```python ant_irl.py```
    
3. Run transfer learning on disabled-ant
	
    ```python ant_transfer_disabled.py``` 

# Bibliography
```
@inproceedings{
qureshi2018adversarial,
title={Adversarial Imitation via Variational Inverse Reinforcement Learning},
author={Ahmed H. Qureshi and Byron Boots and Michael C. Yip},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=HJlmHoR5tQ},
}
```

