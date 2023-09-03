from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
import time
import numpy as np
import random
import gym
from gym import spaces
import torch as th
from gym_pybullet_drones.examples.cross_rl import rl_ude
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy



class RetrainEnv(gym.Env):
    def __init__(self, data: np.ndarray):
        
        # 定义动作空间
        #Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
        #Box(2,)
        self.action_space = spaces.Box(
            low=np.array([0.1,0.1,0.1,0.1,0.1,0.1]),
            high=np.array([10,10,10,10,10,10]),
            dtype=np.float32
            )
        # self.self.PYB_CLIENT = p.connect(p.GUI if self._render else p.DIRECT)

        # 定义状态空间
        obs_lower_bound = np.array([-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.])
        obs_upper_bound = np.array([np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  600000, 600000, 600000, 600000])
        self.observation_space  =  spaces.Box(low=obs_lower_bound,
                                            high=obs_upper_bound,
                                            dtype=np.float32
                                            )
        
        self.step_num: int = 0

        self.data: np.ndarray = data
        self.data_length: int = self.data.shape[0]

    def step (self, 
              act: Union[None, np.ndarray] = None) -> Tuple[Any, Any, Any, Any]:
        states = self.data[act, 0:20]
        rewards = 0
        done = False
        info = {}
        return states, rewards, done, info
    
    def reset(self):
        states = self.data[0, 0:20]
        rewards = 0
        done = False
        info = {}
        return states
    
    
class RetrainPPO(PPO):
    def __init__(self, 
                 policy: Union[str, Type[ActorCriticPolicy]],
                 env: Union[GymEnv, str],
                 data: np.ndarray,
                 learning_rate: Union[float, Schedule] = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: Union[float, Schedule] = 0.2,
                 clip_range_vf: Union[None, float, Schedule] = None,
                 normalize_advantage: bool = True,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 target_kl: Optional[float] = None,
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = "auto",
                 _init_setup_model: bool = True):
        super(RetrainPPO, self).__init__(policy, env, 
                                         learning_rate, 
                                         n_steps, 
                                         batch_size, 
                                         n_epochs, 
                                         gamma, 
                                         gae_lambda, 
                                         clip_range, 
                                         clip_range_vf,
                                         normalize_advantage,
                                         ent_coef,
                                         vf_coef,
                                         max_grad_norm,
                                         use_sde,
                                         sde_sample_freq,
                                         target_kl,
                                         tensorboard_log,
                                         create_eval_env,
                                         policy_kwargs,
                                         verbose,
                                         seed,
                                         device)
        self.experiment_data = th.from_numpy(data).to(self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                # actions, values, log_probs = self.policy(obs_tensor)
                # Get actions, values, log_probs from data file
                actions = self.experiment_data[self.num_timesteps, 2:8]
                values = self.experiment_data[self.num_timesteps, 0]
                log_probs = self.experiment_data[self.num_timesteps, 1]
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # new_obs, rewards, dones, infos = env.step(clipped_actions)
            new_obs, rewards, dones, infos = env.step(np.array([self.num_timesteps]))

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    
if __name__ == '__main__':
    data = np.vstack((np.loadtxt("ICRA_data.txt"),
                      np.loadtxt("ICRA_data1.txt"),
                      np.loadtxt("ICRA_data2.txt"),
                      np.loadtxt("ICRA_data3.txt"),
                      np.loadtxt("ICRA_data4.txt"),
                      np.loadtxt("ICRA_data5.txt"),
                      np.loadtxt("ICRA_data6.txt"),
                      np.loadtxt("ICRA_data7.txt"),
                      np.loadtxt("ICRA_data8.txt")))
    data = np.vstack((data, data))
    env = RetrainEnv(data)
    model = RetrainPPO("MlpPolicy", env, data[:, 20:28], verbose=1, tensorboard_log="./ppo/")
    model.learn(total_timesteps=4096)
    model.save("ppo_retrain")
    