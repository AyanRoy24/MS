import wandb
import os
import numpy as np
import jax
import jax.random as jr
import functools as ft
import jax.numpy as jnp
import h5py

from time import time
from tqdm import tqdm

from .data import Rollout
from .utils import test_rollout
from ..env import MultiAgentEnv
from ..algo.base import Algorithm
from ..utils.graph import GraphsTuple
from dgppo.env.mpe.base import LocalObservationEncoder


class Trainer:

    def __init__(
            self,
            env: MultiAgentEnv,
            env_test: MultiAgentEnv,
            algo: Algorithm,
            gamma: float,
            n_env_train: int,
            n_env_test: int,
            log_dir: str,
            seed: int,
            params: dict,
            name : str,
            save_log: bool = True,
    ):
        self.env = env
        # self.env.get_graph
        self.env_test = env_test
        self.algo = algo
        self.gamma = gamma
        self.n_env_train = n_env_train
        self.n_env_test = n_env_test
        self.log_dir = log_dir
        self.seed = seed
        self.name = name

        # self.encoder = LocalObservationEncoder(
        #     comm_radius=self.env.params["comm_radius"],
        #     car_radius=self.env.params["car_radius"],
        #     obs_radius=self.env.params["obs_radius"],
        #     max_agents=self.env.num_agents,
        #     max_obs=self.env.params["n_obs"],
        #     include_goal=True
        # )

        if Trainer._check_params(params):
            self.params = params

        # make dir for the models
        if save_log:
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            self.model_dir = os.path.join(log_dir, 'models')
            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)

        wandb.login()
        wandb.init(name=params['run_name'], project='dgppo', group=env.__class__.__name__, dir=self.log_dir)

        self.save_log = save_log

        self.steps = params['training_steps']
        self.eval_interval = params['eval_interval']
        self.eval_epi = params['eval_epi']
        self.save_interval = params['save_interval']

        self.update_steps = 0
        self.key = jax.random.PRNGKey(seed)

    @staticmethod
    def _check_params(params: dict) -> bool:
        assert 'run_name' in params, 'run_name not found in params'
        assert 'training_steps' in params, 'training_steps not found in params'
        assert 'eval_interval' in params, 'eval_interval not found in params'
        assert params['eval_interval'] > 0, 'eval_interval must be positive'
        assert 'eval_epi' in params, 'eval_epi not found in params'
        assert params['eval_epi'] >= 1, 'eval_epi must be greater than or equal to 1'
        assert 'save_interval' in params, 'save_interval not found in params'
        assert params['save_interval'] > 0, 'save_interval must be positive'
        return True

    def train(self):
        # record start time
        start_time = time()

        # preprocess the rollout function
        init_rnn_state = self.algo.init_rnn_state

        def test_fn_single(params, key):
            act_fn = ft.partial(self.algo.act, params=params)
            return test_rollout(
                self.env_test,
                act_fn,
                init_rnn_state,
                key
            )
        # test_rollout shape :- (n_env_test, T, n_agent)
  
        test_fn = lambda params, keys: jax.vmap(ft.partial(test_fn_single, params))(keys)
        test_fn = jax.jit(test_fn)

        # start training
        test_key = jr.PRNGKey(self.seed)
        assert self.n_env_test <= 1_000, 'n_env_test must be less than or equal to 1_000'
        test_keys = jr.split(test_key, 1_000)[:self.n_env_test]

        pbar = tqdm(total=self.steps, ncols=80)
        env_type = self.env_test.__class__.__name__ if hasattr(self.env_test, "__class__") else self.env_test

        
        with h5py.File(self.name, 'a') as f:
            # datasets_created = False
            def append(name, arr):
                arr = np.array(arr)
                arr = np.squeeze(arr) # for example, if arr is (1, 3, 2), it will become (3, 2)
                # if arr.ndim >= 3:
                #         arr = arr.reshape(-1, *arr.shape[2:])    # If arr is still more than 2D and first dim is batch, keep only first batch (optional)
                # if arr.ndim > 2 and arr.shape[0] == self.n_env_test:
                #     arr = arr[0]
                
                # If arr has no batch dimension, add one
                if arr.shape == (): arr = arr[None]
                elif arr.ndim == 1: arr = arr[None, :]
                elif arr.ndim > 1 and arr.shape[0] != 1: arr = arr[None, ...]  # add batch dimension

                shape = arr.shape
                maxshape = (None,) + shape[1:]
                if name in f:
                    dset = f[name]
                    dset.resize(dset.shape[0] + 1, axis=0)
                    # Make sure arr shape matches dset[-1] shape
                    if arr.shape[1:] != dset.shape[1:]:
                        raise ValueError(f"Shape mismatch: arr.shape={arr.shape}, dset.shape={dset.shape}")
                    dset[-1] = arr[0]
                else:
                    f.create_dataset(name, data=arr, maxshape=maxshape, compression='gzip')


            for step in range(0, self.steps): # + 1):
            
                eval_info = {}
                test_rollouts: Rollout = test_fn(self.algo.params, test_keys)
                total_reward = test_rollouts.rewards.sum(axis=-1)
                reward_min, reward_max = total_reward.min(), total_reward.max()
                reward_mean = np.mean(total_reward)
                reward_final = np.mean(test_rollouts.rewards[:, -1])
                cost_mean = jnp.maximum(test_rollouts.costs, 0.0).max(axis=-1).max(axis=-1).sum(axis=-1).mean()
                unsafe_frac = np.mean(test_rollouts.costs.max(axis=-1).max(axis=-2) >= 1e-6) # fraction of environments with unsafe cost
                safe_frac = 1 - unsafe_frac
                eval_info = eval_info | {
                    "eval/reward": reward_mean,
                    "eval/reward_final": reward_final,
                    "eval/cost": cost_mean,
                    "eval/unsafe_frac": unsafe_frac,
                }
                time_since_start = time() - start_time
                eval_verbose = (f'step: {step:3}, time: {time_since_start:5.0f}s, reward: {reward_mean:9.4f}, '
                                f'min/max reward: {reward_min:7.2f}/{reward_max:7.2f}, cost: {cost_mean:8.4f}, '
                                f'unsafe_frac: {unsafe_frac:6.2f}')
                tqdm.write(eval_verbose)
                wandb.log(eval_info, step=self.update_steps)
                append('cost_mean', cost_mean)
                append('reward_mean', reward_mean)
                # append('reward_min', reward_min)
                # append('reward_max', reward_max)
                # append('reward_final', reward_final)
                append('safe_frac', safe_frac)
                # append('unsafe_frac', unsafe_frac)
                # append('step', np.array([step]))
                # append('time', np.array([time_since_start]))
                # append('total_reward', total_reward[None])

                if env_type in ["LidarTarget", "LidarSpread","LidarLine","LidarBicycleTarget"]:
                    agent = test_rollouts.graph.env_states.agent
                    goal = test_rollouts.graph.env_states.goal
                    obstacle = test_rollouts.graph.env_states.obstacle.center if test_rollouts.graph.env_states.obstacle is not None else None
                    n_agent = test_rollouts.next_graph.env_states.agent
                    n_goal = test_rollouts.next_graph.env_states.goal
                    n_obstacle = test_rollouts.next_graph.env_states.obstacle.center if test_rollouts.graph.env_states.obstacle is not None else None
                    
                elif env_type in ["MPETarget", "MPESpread","MPEFormation","MPELine", "MPECorridor", "MPEConnectSpread"]:
                    agent = test_rollouts.graph.env_states.agent
                    goal = test_rollouts.graph.env_states.goal
                    obstacle = test_rollouts.graph.env_states.obs if test_rollouts.graph.env_states.obs is not None else None
                    n_agent = test_rollouts.next_graph.env_states.agent
                    n_goal = test_rollouts.next_graph.env_states.goal
                    n_obstacle = test_rollouts.next_graph.env_states.obs if test_rollouts.graph.env_states.obs is not None else None

                elif env_type == "VMASReverseTransport":
                    agent = test_rollouts.graph.env_states.a_pos
                    goal = test_rollouts.graph.env_states.goal_pos
                    obstacle = test_rollouts.graph.env_states.o_pos

                elif env_type == "VMASWheel":
                    agent = test_rollouts.graph.env_states.a_pos
                    goal = test_rollouts.graph.env_states.goal_angle
                    obstacle = test_rollouts.graph.env_states.avoid_angle

                else:
                    raise ValueError(f"Unknown environment type: {env_type}")
                
                state = np.concatenate([agent, goal, obstacle],axis=-1) if obstacle is not None else np.concatenate([agent, goal],axis=-1)

                next_state = np.concatenate([n_agent, n_goal, n_obstacle],axis=-1) if n_obstacle is not None else np.concatenate([n_agent, n_goal],axis=-1)

                action = test_rollouts.actions
                cost = test_rollouts.costs
                reward = test_rollouts.rewards
                # reward_m = np.tile(reward[...,None], (1,3))
                # append('agent', agent[None])
                # append('goal', goal[None])
                # append('obstacle', obstacle[None] if obstacle is not None else np.array([]))
                append('state', state)
                append('next_state', next_state)
                append('action', action)
                append('cost', cost)
                append('reward', reward)
                # append('reward', reward_m[None])
                # for t in range(128):  # Assuming T=128 for all environments   
                # #     agent = test_rollouts.graph.env_states.agent[0,t]#.mean(axis=0).flatten()#(axis=(0,1))  # Assuming agent is a field in the graph
                # #     goal = test_rollouts.graph.env_states.goal[0,t]#.mean(axis=0).flatten()#(axis=(0,1))  # Assuming goal is a field in the graph
                # #     obstacle = test_rollouts.graph.env_states.obs[0,t]#.mean(axis=0).flatten() if test_rollouts.graph.env_states.obs is not None else None #obstacle.center.mean(axis=0).flatten()     
                # #     globalState = np.concatenate([agent, goal, obstacle]) if obstacle.size > 0 else np.concatenate([agent, goal])

                # #     next_agent = test_rollouts.next_graph.env_states.agent[0,t]#.mean(axis=0).flatten() #(axis=(0,1))  # Assuming agent is a field in the graph
                # #     next_goal = test_rollouts.next_graph.env_states.goal[0,t]
                # #     next_obstacle = test_rollouts.next_graph.env_states.obs[0,t]#.mean(axis=0).flatten() if test_rollouts.graph.env_states.obs is not None else None #obstacle.center.mean(axis=0).flatten()            
                # #     nextGlobalState = (
                # #         np.concatenate([next_agent, next_goal, next_obstacle])
                # #         if next_obstacle.size > 0 else np.concatenate([next_agent, next_goal])
                # #     )

                #     action = test_rollouts.actions[0,t]#.flatten()
                # #     jointReward = test_rollouts.rewards[0,t]
                # #     globalCost = test_rollouts.costs[0,t]
                # #     done = test_rollouts.dones[0,t]


                #     b_graph = test_rollouts.graph    
                #     env_states = b_graph.env_states  # This is a NamedTuple

                # #     # # For each field in env_states, index [env_idx, t]
                #     env_states_single = type(env_states)(
                #         **{k: (getattr(env_states, k)[0,t] if getattr(env_states, k) is not None else None)
                #         for k in env_states._fields}
                #     )
                #     s_graph = GraphsTuple(
                #             n_edge=b_graph.n_edge[0,t],
                #             n_node=b_graph.n_node[0,t],
                #             nodes=b_graph.nodes[0,t],
                #             edges=b_graph.edges[0,t],
                #             states=b_graph.states[0,t],
                #             receivers=b_graph.receivers[0,t],
                #             senders=b_graph.senders[0,t],
                #             node_type=b_graph.node_type[0,t],
                #             env_states=env_states_single,
                #             connectivity=b_graph.connectivity[0,t] if b_graph.connectivity is not None else None,
                #         )
                #     localObs = self.env_test.get_local_observations(s_graph)
                #     # local_obs_enc = self.encoder.encode(s_graph)
                #     localReward = self.env_test.get_reward_local(s_graph,action)
                #     localCost = self.env_test.get_cost_local(s_graph)

                #     n_b_graph = test_rollouts.next_graph#[env_idx, t]
                #     n_env_states = n_b_graph.env_states  # This is a NamedTuple

                # #     # # For each field in env_states, index [env_idx, t]
                #     n_env_states_single = type(n_env_states)(
                #         **{k: (getattr(n_env_states, k)[0,t] if getattr(n_env_states, k) is not None else None)
                #         for k in n_env_states._fields}
                #     )
                #     n_s_graph = GraphsTuple(
                #         n_edge=n_b_graph.n_edge[0,t],
                #         n_node=n_b_graph.n_node[0,t],
                #         nodes=n_b_graph.nodes[0,t],
                #         edges=n_b_graph.edges[0,t],
                #         states=n_b_graph.states[0,t],
                #         receivers=n_b_graph.receivers[0,t],
                #         senders=n_b_graph.senders[0,t],
                #         node_type=n_b_graph.node_type[0,t],
                #         env_states=n_env_states_single,
                #         connectivity=n_b_graph.connectivity[0,t] if n_b_graph.connectivity is not None else None,
                #     )
                    
                # #     # next_agent = n_b_graph.env_states.agent[0,t]#.flatten()
                # #     # next_goal = n_b_graph.env_states.goal[0,t]#.flatten()
                # #     # next_obstacle = n_b_graph.env_states.obs[0,t]#.flatten()
                        

                # #     # nextGlobalState = (
                # #     #     np.concatenate([next_agent, next_goal, next_obstacle])
                # #     #     if next_obstacle.size > 0 else np.concatenate([next_agent, next_goal])
                # #     # )
                #     nextLocalObs = self.env_test.get_local_observations(n_s_graph)
                    
                # #     # append('agent', agent[None])
                # #     # append('goal', goal[None])
                # #     # append('obstacle', obstacle[None])
                # #     append('state', globalState[None])
                #     append('localObs', localObs[None])
                # #     # append_or_create('local_obs_encoded', local_obs_enc)
                # #     append('action', action[None])
                # #     append('next_state', nextGlobalState[None])
                #     append('nextLocalObs', nextLocalObs[None])
                # #     # append_or_create('next_local_obs_encoded', n_local_obs_enc)                            
                # #     append('reward', np.array([jointReward]))
                # #     append('cost', np.array([globalCost]))
                #     append('localReward', localReward[None])
                #     append('localCost', localCost[None])        
                # #     append('done', np.array([done]))
        
                # save the model
                if self.save_log and step % self.save_interval == 0:
                    self.algo.save(os.path.join(self.model_dir), step)

                # collect rollouts
                key_x0, self.key = jax.random.split(self.key)
                key_x0 = jax.random.split(key_x0, self.n_env_train)
                rollouts = self.algo.collect(self.algo.params, key_x0)

                # update the algorithm
                update_info = self.algo.update(rollouts, step)
                wandb.log(update_info, step=self.update_steps)
                self.update_steps += 1

                pbar.update(1)

        # desired_order = [
        #     'step', 'time', 'state', 'action', 'next_state', 'reward', 'cost_mean', 'done',
        #     'total_reward', 'reward_mean', 'reward_min', 'reward_max', 'reward_final',
        #     'cost', 'safe_frac', 'unsafe_frac'
        # ]

        # with h5py.File('MPESpread_dgppo.h5', 'r') as f_in, h5py.File('MPESpread_dgppo_reordered.h5', 'w') as f_out:
        #     for name in desired_order:
        #         if name in f_in:
        #             f_in.copy(name, f_out)


        # # Next state (if not last timestep)
                    # if t < test_rollouts.rewards.shape[1] - 1:
                    #     next_agent = test_rollouts.graph[env_idx, t+1].env_states.agent.flatten()
                    #     next_goal = test_rollouts.graph[env_idx, t+1].env_states.goal.flatten()
                    #     next_obstacle = (test_rollouts.graph[env_idx, t+1].env_states.obs.flatten()
                    #                     if test_rollouts.graph[env_idx, t+1].env_states.obs is not None else np.array([]))
                    #     next_state = (np.concatenate([next_agent, next_goal, next_obstacle])
                    #                 if next_obstacle.size > 0 else np.concatenate([next_agent, next_goal]))
                    # else:
                    #     next_state = state  # or np.zeros_like(state)

                    # if t < test_rollouts.rewards.shape[1] - 1:
     
# evaluate the algorithm
                # for epi in range(self.eval_epi):
                    # eval_keys = jr.split(test_key,self.eval_epi)
                    # eval_keys = jr.split(test_key[epi], 1_000)[:self.n_env_test]
                    # test_rollouts: Rollout = test_fn(self.algo.params, eval_keys)
# eval_keys = jr.split(test_key,self.eval_epi)
                # eval_keys = jr.split(test_key[epi], 1_000)[:self.n_env_test]
                # test_rollouts: Rollout = test_fn(self.algo.params, eval_keys)
              
# Loop over environments
                # for env_idx in range(self.n_env_test):
                    # Loop over timesteps
                # for t in range(test_rollouts.rewards.shape[1]):  # T
              

            # local_obs = []
            # n_local_obs = []
            # local_cost = []
            # local_reward = []


       # self.encoder = LocalObservationEncoder(
        #     comm_radius=self.env.params["comm_radius"],
        #     car_radius=self.env.params["car_radius"],
        #     obs_radius=self.env.params["obs_radius"],
        #     max_agents=self.env.num_agents,
        #     max_obs=self.env.params["n_obs"],
        #     include_goal=True
        # )
'''
trainer local
            

'''               
'''
                    if not datasets_created:
                        dataset_specs = [
                            ('step', ()),  # scalar
                            ('time', ()),  # scalar
                            ('state', state.shape),  # shape from first eval
                            ('action', action.shape),
                            ('next_state', next_state.shape),
                            ('reward', reward.shape),
                            ('cost_mean', ()),  # scalar
                            ('done', dones.shape),
                            ('total_reward', total_reward.shape),
                            ('reward_mean', ()),  # scalar
                            ('reward_min', ()),  # scalar
                            ('reward_max', ()),  # scalar
                            ('reward_final', ()),  # scalar
                            ('cost', costs.shape),
                            ('safe_frac', ()),  # scalar
                            ('unsafe_frac', ()),  # scalar
                        ]
                        # Only create if not already present
                        for name, shape in dataset_specs:
                            if name not in f:
                                # All datasets will have shape (0, ...) to start (0 along first axis)
                                maxshape = (None,) + (shape if isinstance(shape, tuple) else (shape,))
                                # For scalars, shape is ()
                                if shape == () or shape == ():
                                    f.create_dataset(name, shape=(0,), maxshape=(None,), dtype='float32', compression='gzip')
                                else:
                                    f.create_dataset(name, shape=(0,) + shape, maxshape=maxshape, dtype='float32', compression='gzip')
                        datasets_created = True

                    # --- 2. Append function: only append, never create ---
                    def append(name, arr):
                        arr = np.array(arr)
                        # If arr is scalar, make it (1,)
                        if arr.shape == ():
                            arr = arr[None]
                        elif arr.ndim == 1:
                            arr = arr[None, :]
                        elif arr.ndim > 1 and arr.shape[0] != 1:
                            arr = arr[None, ...]
                        dset = f[name]
                        dset.resize(dset.shape[0] + 1, axis=0)
                        if arr.shape[1:] != dset.shape[1:]:
                            raise ValueError(f"Shape mismatch: arr.shape={arr.shape}, dset.shape={dset.shape}")
                        dset[-1] = arr[0]

'''


                    # n_env_test, T = test_rollouts.graph.env_states.shape[:2]
                    # T = 128
                    # local_obs = []
                    # n_local_obs = []
                    # local_cost = []
                    # local_reward = []
                    # for i in range(self.n_env_test):
                    #     for t in range(T):
                    #         env_state = test_rollouts.graph.env_states[i, t]
                    #         n_env_state = test_rollouts.next_graph.env_states[i, t]
                    #         graph = self.env_test.get_graph(env_state)
                    #         n_graph = self.env_test.get_graph(n_env_state)
                    #         local_obs.append(self.env_test.get_local_observations(graph))
                    #         n_local_obs.append(self.env_test.get_local_observations(n_graph))
                    #         local_cost.append(self.env_test.get_cost_local(graph))
                    #         local_reward.append(self.env_test.get_reward_local(graph, action[i, t]))

                    # local_obs = np.array(local_obs).reshape(self.n_env_test, T, -1, local_obs[0].shape[-1])
                    # n_local_obs = np.array(n_local_obs).reshape(self.n_env_test, T, -1, n_local_obs[0].shape[-1])
                    # local_cost = np.array(local_cost).reshape(self.n_env_test, T, -1)
                    # local_reward = np.array(local_reward).reshape(self.n_env_test, T, -1)

                    # batched_graph = test_rollouts.graph
                    # batch_size = batched_graph.states.shape[0]
                    # print('states shape', batched_graph.states.shape)  # or self.n_env_test
                    # single_graphs = unbatch_graphs_tuple(batched_graph, batch_size)
                    # local_obs = np.stack([self.env_test.get_local_observations(g) for g in single_graphs])
                    # local_obs_enc = np.stack([self.encoder.encode(g) for g in single_graphs])
                    # local_reward = np.stack([self.env_test.get_reward_local(g, a) for g, a in zip(single_graphs, action)])
                    # local_cost = np.stack([self.env_test.get_cost_local(g) for g in single_graphs])


                    # for i in range(n_env_test):
                    #     for t in range(T):
                    #         graph = test_rollouts.graph[i, t]
                    #         n_graph = test_rollouts.next_graph[i, t]  # This is a GraphsTuple for env i at time t
                    #         # env_state = graph.env_states       # This is the MPEEnvState for that step
                    #         local_obs.append(self.env_test.get_local_observations(graph))
                    #         n_local_obs.append(self.env_test.get_local_observations(n_graph))
                    #         local_cost.append(self.env_test.get_cost_local(graph))
                    #         local_reward.append(self.env_test.get_reward_local(graph, action[i, t]))


'''
line 114
Step-by-step:

jnp.maximum(test_rollouts.costs, 0.0)

Ensures all costs are non-negative (clips negative values to 0).
.max(axis=-1)

Takes the maximum cost across agents for each timestep and environment.
Shape: (n_env_test, T)
.max(axis=-1)

Takes the maximum cost across timesteps for each environment.
Shape: (n_env_test,)
.sum(axis=-1)

Sums the maximum costs across all environments.
Shape: () (scalar)
.mean()

Takes the mean (though after .sum() this is redundant unless you want the mean across something else; check if this is intended).
'''


'''
line 115
Step-by-step:

test_rollouts.costs.max(axis=-1)

Maximum cost per agent at each timestep and environment.
Shape: (n_env_test, T)
.max(axis=-2)

Maximum cost per environment across all timesteps.
Shape: (n_env_test,)
>= 1e-6

Boolean array: True if the maximum cost in that environment is at least 1e-6.
np.mean(...)

Fraction of environments where the maximum cost is at least 1e-6.
'''

'''
trainer global


            # self.eval_epi = int((1e6 * self.eval_interval) / self.steps)
            # print(f"Number of evaluation episodes: {self.eval_epi}")
            for step in range(0, self.steps + 1):
                # evaluate the algorithm
                if step % self.eval_interval == 0:
                    eval_info = {}
                    eval_keys = jr.split(test_key,self.eval_epi)
                    all_rewards, all_costs, all_unsafe = [], [], []
                    for epi in range(self.eval_epi):
                        eval_keys = jr.split(eval_keys[epi], 1_000)[:self.n_env_test]
                        test_rollouts: Rollout = test_fn(self.algo.params, eval_keys) #test_keys)
                        total_reward = test_rollouts.rewards.sum(axis=-1)
                        reward_min, reward_max = total_reward.min(), total_reward.max()
                        reward_mean = np.mean(total_reward)
                        reward_final = np.mean(test_rollouts.rewards[:, -1])
                       
                        
                        agent = test_rollouts.graph.env_states.agent.mean(axis=0).flatten()#(axis=(0,1))  # Assuming agent is a field in the graph
                        goal = test_rollouts.graph.env_states.goal.mean(axis=0).flatten()#(axis=(0,1))  # Assuming goal is a field in the graph
                        obstacle = test_rollouts.graph.env_states.obs.mean(axis=0).flatten() if test_rollouts.graph.env_states.obs is not None else None #obstacle.center.mean(axis=0).flatten() 
                        state = np.concatenate([agent, goal, obstacle]) if obstacle is not None else np.concatenate([agent, goal])
            
                        action = test_rollouts.actions.mean(axis=0)#(axis=(0,1)) # Ensure actions are numpy array

                        next_agent = test_rollouts.next_graph.env_states.agent.mean(axis=0).flatten() #(axis=(0,1))  # Assuming agent is a field in the graph
                        next_goal = test_rollouts.next_graph.env_states.goal.mean(axis=0).flatten() #(axis=(0,1))  # Assuming goal is a field in the graph
                        next_obstacle = test_rollouts.next_graph.env_states.obs.mean(axis=0).flatten() if test_rollouts.graph.env_states.obs is not None else None #obstacle.center.mean(axis=0).flatten() 
                        next_state = np.concatenate([next_agent, next_goal, next_obstacle]) if next_obstacle is not None else np.concatenate([next_agent, next_goal])
            
                        reward = test_rollouts.rewards.mean(axis=0)#(axis=(0, 1))
                        costs = test_rollouts.costs.mean(axis=0)#(axis=(0, 1))

                        dones = test_rollouts.dones #.mean(axis=0)#(axis=(0, 1))

                        # Mean reward across all environments and timesteps
                        # For each environment, finds the largest cost encountered by any agent at any timestep.
                        # Sums these maximum costs across all environments.
                        # Averages the sum (though this may be redundant).
                        cost = jnp.maximum(test_rollouts.costs, 0.0).max(axis=-1).max(axis=-1).sum(axis=-1).mean()
                        # Computes the fraction of test environments where any agent ever incurred a non-trivial cost (i.e., the episode was "unsafe" at least once).
                        unsafe_frac = np.mean(test_rollouts.costs.max(axis=-1).max(axis=-2) >= 1e-6)
                        safe_frac = 1 - unsafe_frac
                        all_rewards.append(reward_mean)
                        all_costs.append(cost)
                        all_unsafe.append(unsafe_frac)

                        step_np = np.array([step])
                        episode_np = np.array([epi])
                        timestep_np = np.array([step + epi])
                        state_np = np.array(state)
                        action_np = np.array(action)
                        next_state_np = np.array(next_state)
                        reward_np = np.array(reward)
                        cost_np = np.array(costs)
                        done_np = np.array(dones)
                        safe_frac_np = np.array(safe_frac)

                        append_or_create('step', step_np)
                        append_or_create('episode', episode_np)
                        append_or_create('timestep', timestep_np)
                        append_or_create('state', state_np)
                        append_or_create('action', action_np)
                        append_or_create('next_state', next_state_np)
                        append_or_create('reward', reward_np)
                        append_or_create('cost', cost_np)
                        append_or_create('done', done_np)
                        append_or_create('safe_frac', safe_frac_np)


                                        
                    reward_mean = np.mean(all_rewards)
                    reward_min = np.min(all_rewards)
                    reward_max = np.max(all_rewards)
                    cost = np.mean(all_costs)
                    unsafe_frac = np.mean(all_unsafe)
                    eval_info = eval_info | {
                        "eval/reward": reward_mean,
                        "eval/reward_final": reward_final,
                        "eval/cost": cost,
                        "eval/unsafe_frac": unsafe_frac,
                    }
                    time_since_start = time() - start_time
                    eval_verbose = (f'step: {step:3}, time: {time_since_start:5.0f}s, reward: {reward_mean:9.4f}, '
                                    f'min/max reward: {reward_min:7.2f}/{reward_max:7.2f}, cost: {cost:8.4f}, '
                                    f'unsafe_frac: {unsafe_frac:6.2f}')
                   
                
                    tqdm.write(eval_verbose)
                    wandb.log(eval_info, step=self.update_steps)

                # save the model
                if self.save_log and step % self.save_interval == 0:
                    self.algo.save(os.path.join(self.model_dir), step)

                # collect rollouts
                key_x0, self.key = jax.random.split(self.key)
                key_x0 = jax.random.split(key_x0, self.n_env_train)
                rollouts = self.algo.collect(self.algo.params, key_x0)

                # update the algorithm
                update_info = self.algo.update(rollouts, step)
                wandb.log(update_info, step=self.update_steps)
                # if 'eval/safe_data' in update_info:
                #     wandb.log({'eval/safe_data': float(update_info['eval/safe_data'])}, step=self.update_steps)
                
                self.update_steps += 1


                pbar.update(1)



                    # # append_or_create('step', step_np)
                    # # append_or_create('time', time_np)
                    # # append_or_create('reward', reward_np)
                    # # append_or_create('reward_min', reward_min_np)
                    # # append_or_create('reward_max', reward_max_np)
                    # # append_or_create('cost', cost_np)
                    # # append_or_create('action', action_np)
                    # # append_or_create('agent', agent_np)
                    # # # append_or_create('unsafe_frac', unsafe_frac_np)
                    # # append_or_create('goal', goal_np)
                    # # append_or_create('obstacle', obstacle_np)
                    # # if obstacle is not None:
                    
                    # # append_or_create('action', test_rollouts.actions)
                    
                    
                    # # graphs_tuple = self.env.get_graph(test_rollouts.graph)
                    # # save_graphs_tuple_state(f, f'env_state/step_{step}', graphs_tuple)                    
                    
                    # # other_rollout_fields = {
                    # #     # "graphs": test_rollouts.graph,
                    # #     "actions": test_rollouts.actions,
                    # #     "actor_rnn_states": test_rollouts.rnn_states,  # or test_rollouts.actor_rnn_states if that's the field name
                    # #     "dones": test_rollouts.dones,
                    # #     "log_pis": test_rollouts.log_pis,
                    # #     # "next_graphs": test_rollouts.next_graph,
                    # # }

                    
                    # # if test_rollouts.log_pis is not None:
                    # #     other_rollout_fields["log_pis"] = np.array(test_rollouts.log_pis)

                    # def append_or_create_nd(name, arr):
                    #     arr_np = np.array(arr)
                    #     if name in f:
                    #         dset = f[name]
                    #         dset.resize(dset.shape[0] + 1, axis = 0) #+ dset.shape[1:], axis=0)
                    #         dset[-1] = arr_np
                    #     else:
                    #         f.create_dataset(name, data=arr_np[None], maxshape=(None,) + arr_np.shape, compression='gzip')

                    # def save_env_state_h5(env_state, h5_path, step):
                    #     # with h5py.File(h5_path, 'a') as f:
                    #     grp = f.require_group(f'env_state/step_{step}')
                    #     # Save agent and goal states
                    #     grp.create_dataset('agent', data=np.array(env_state.agent))
                    #     grp.create_dataset('goal', data=np.array(env_state.goal))
                    #     # Save obstacle attributes if obstacles exist
                    #     if env_state.obstacle is not None:

                    #         obs_grp = grp.require_group('obstacle')
                    #         for field in env_state.obstacle._fields:
                    #             arr = getattr(env_state.obstacle, field)
                    #             obs_grp.create_dataset(field, data=np.array(arr))
                    
                    # def append_graphs_tuple_group(group, name, graphs_tuple):
                    #     grp = group.require_group(name)
                    #     for field in graphs_tuple._field_names:
                    #         arr = getattr(graphs_tuple, field)
                    #         if arr is None:
                    #             continue
                    #         # Only save if arr is a numpy/jax array (has 'shape' attribute)
                    #         if not hasattr(arr, 'shape'):
                    #             print(f"Skipping field '{field}' in '{name}' because it is not an array (type={type(arr)}).")
                    #             continue
                    #         print(f"{name}.{field}: type={type(arr)}, shape={getattr(arr, 'shape', None)}")
                    #         if isinstance(arr, list):
                    #             try:
                    #                 arr_np = np.stack(arr)
                    #             except Exception:
                    #                 print(f"Skipping field '{field}' in '{name}' because it is a list of non-uniform arrays/objects.")
                    #                 continue
                    #         else:
                    #             arr_np = np.array(arr)
                    #         if field in grp:
                    #             dset = grp[field]
                    #             dset.resize(dset.shape[0] + 1, axis =0) # + dset.shape[1:], axis=0)
                    #             dset[-1] = arr_np
                    #         else:
                    #             grp.create_dataset(field, data=arr_np[None], maxshape=(None,) + arr_np.shape, compression='gzip')
                    
                    # def save_graphs_tuple_state(f, group_name, graphs_tuple):
                    #     grp = f.require_group(group_name)
                    #     # Save agent and goal
                    #     grp.create_dataset('agent', data=np.array(graphs_tuple.env_states.agent))
                    #     grp.create_dataset('goal', data=np.array(graphs_tuple.env_states.goal))
                    #     # Save obstacle fields if present
                    #     if graphs_tuple.env_states.obstacle is not None:
                    #         obs_grp = grp.require_group('obstacle')
                    #         for field in graphs_tuple.env_states.obstacle._fields:
                    #             arr = getattr(graphs_tuple.env_states.obstacle, field)
                    #             obs_grp.create_dataset(field, data=np.array(arr))
                    
                    # # for key, value in other_rollout_fields.items():
                    # #     if isinstance(value, list):
                    # #         value = np.stack(value)
                    # #     append_or_create_nd(key, value)
                    
                    # # Usage in your loop:
                    # # append_graphs_tuple_group(f, 'graph', test_rollouts.graph)
                    # # append_graphs_tuple_group(f, 'next_graph', test_rollouts.next_graph)
                    
                    # # for t, graph in enumerate(test_rollouts.graph[epi]):
                    # #     env_state = graph.env_states
                    # #     grp = f.require_group(f'episode_{epi}/t{t}')
                    # #     grp.create_dataset('agent', data=np.array(env_state.agent))
                    # #     grp.create_dataset('goal', data=np.array(env_state.goal))
                    # #     if env_state.obstacle is not None:
                    # #         obs_grp = grp.require_group('obstacle')
                    # #         for field in env_state.obstacle._fields:
                    # #             arr = getattr(env_state.obstacle, field)
                    # #             obs_grp.create_dataset(field, data=np.array(arr))

                    # def append_or_create_env_state(name, arr):
                    #     arr_np = np.array(arr)
                    #     if name in f:
                    #         dset = f[name]
                    #         dset.resize(dset.shape[0] + 1, axis=0)
                    #         dset[-1] = arr_np
                    #     else:
                    #         f.create_dataset(name, data=arr_np[None], maxshape=(None,) + arr_np.shape, compression='gzip')

                    # # In your evaluation loop, after getting test_rollouts:
                    # # for t, graph in enumerate(test_rollouts.graph):  # assuming n_env_test == 1
                    # #     env_state = graph.env_states  # This is a LidarEnvState
                    # #     append_or_create_env_state('agent', env_state.agent)
                    # #     append_or_create_env_state('goal', env_state.goal)
                    # #     if env_state.obstacle is not None:
                    # #         for field in env_state.obstacle._fields:
                    # #             arr = getattr(env_state.obstacle, field)
                    # #             append_or_create_env_state(f'obstacle_{field}', arr)
                                

                # if step % self.eval_interval == 0:
                #     per_agent_keys = [
                #         'bTa_is_safe', 'bTah_Vh', 'bTah_Acbf', 'bTah_Qh', 'bTa_Al'
                #         # add more keys as needed for your use case
                #     ]
                #     # Save update_info to HDF5: mean for all except bTa_is_safe (save as array), skip eval/safe_data
                    

                #     # for key, arr in update_info.items():
                #     #     append_or_create_nd(key, arr)

                    # for key, arr in update_info.items():
                    #     if key == 'eval/safe_data':
                    #         continue  # Don't save this field
                    #     arr_np = np.array(arr)
                    #     if key in per_agent_keys:
                    #         if arr_np.ndim >= 3:
                    #             arr_mean = np.mean(arr_np, axis=0)  # mean over batch
                    #         else:
                    #             arr_mean = arr_np
                    #         append_or_create_nd(key, arr_mean)  # Save full array
                    #     else:
                    #         # Save mean as a scalar
                    #         arr_mean = np.mean(arr_np)
                    #         append_or_create(key, np.array([arr_mean]))
                    

                     # ...existing code...
                    # Prepare data for HDF5
                    # step_np = np.array([step])
                    # time_np = np.array([time_since_start])
                    # reward_np = np.array([reward_mean])
                    # reward_min_np = np.array([reward_min])
                    # reward_max_np = np.array([reward_max])
                    # cost_np = np.array([cost])
                    # unsafe_frac_np = np.array([unsafe_frac])
                    # action_np = np.array([action])
                    # agent_np = np.array([agent])
                    # goal_np = np.array([goal])
                    # if obstacle is not None:
                    #     obstacle_np = np.array([obstacle])
                    # else:
                    #     obstacle_np = None
                    # Ensure action is a numpy array
                   

                    # def append_or_create(name, arr):
                    #     arr = np.array(arr)
                    #     shape = arr.shape
                    #     maxshape = (None,) + shape[1:] 
                    #     if name in f:
                    #         dset = f[name]
                    #         # dset.resize((dset.shape[0] + 1,), axis=0)
                    #         dset.resize(dset.shape[0] + 1, axis=0)
                    #         dset[-1] = arr
                    #     else:
                    #         f.create_dataset(name, data=arr, maxshape=maxshape,compression='gzip')




                # for t, graph in enumerate(test_rollouts.graph):
                #     env_state = graph.env_states  # This is a LidarEnvState
                #     agent = np.array(env_state.agent)
                #     goal = np.array(env_state.goal)
                #     # For obstacle, save each field
                #     if env_state.obstacle is not None:
                #         for field in env_state.obstacle._fields:
                #             arr = getattr(env_state.obstacle, field)
                            # Save arr to HDF5 as needed



                    # for key, arr in update_info.items():
                    #     # Only save arrays that are 1D or can be stacked along axis 0
                    #     arr_np = np.array(arr)
                    #     if arr_np.ndim == 0:
                    #         arr_np = arr_np[None]
                    #     append_or_create(key, arr_np)

'''   

'''
for i in range(self.n_env_test):
                    for t in range(128):
                        if step % self.eval_interval == 0:
                            
# NEW CODE
                            agent = test_rollouts.graph.env_states.agent#.mean(axis=0).flatten()#(axis=(0,1))  # Assuming agent is a field in the graph
                            goal = test_rollouts.graph.env_states.goal#.mean(axis=0).flatten()#(axis=(0,1))  # Assuming goal is a field in the graph
                            obstacle = test_rollouts.graph.env_states.obs#.mean(axis=0).flatten() if test_rollouts.graph.env_states.obs is not None else None #obstacle.center.mean(axis=0).flatten() 
                            state = np.concatenate([agent, goal, obstacle]) if obstacle.size > 0 else np.concatenate([agent, goal])

                            next_agent = test_rollouts.next_graph.env_states.agent#.mean(axis=0).flatten() #(axis=(0,1))  # Assuming agent is a field in the graph
                            next_goal = test_rollouts.next_graph.env_states.goal#.mean(axis=0).flatten() #(axis=(0,1))  # Assuming goal is a field in the graph
                            next_obstacle = test_rollouts.next_graph.env_states.obs#.mean(axis=0).flatten() if test_rollouts.graph.env_states.obs is not None else None #obstacle.center.mean(axis=0).flatten() 
                            next_state = np.concatenate([next_agent, next_goal, next_obstacle]) if next_obstacle.size > 0 else np.concatenate([next_agent, next_goal])

                            action = test_rollouts.actions[i, t]#.mean(axis=0)#(axis=(0,1)) # Ensure actions are numpy array

                            reward = test_rollouts.rewards[i, t]#.mean(axis=0)#(axis=(0, 1))
                            costs = test_rollouts.costs[i, t] #.mean(axis=0)#(axis=(0, 1))

                            dones = test_rollouts.dones[i, t] #.mean(axis=0)#(axis=(0, 1))

# Local save code
                            c_env_states = test_rollouts.graph.env_states
                            n_env_states = test_rollouts.next_graph.env_states
                            env_states_single = type(c_env_states)(
                                **{k: (getattr(c_env_states, k)[i, t] if getattr(c_env_states, k) is not None else None)
                                for k in c_env_states._fields}
                            )
                            single_graph = GraphsTuple(
                                n_node=test_rollouts.graph.n_node[i],
                                n_edge=test_rollouts.graph.n_edge[i],
                                nodes=test_rollouts.graph.nodes[i, t],
                                edges=test_rollouts.graph.edges[i, t],
                                states=test_rollouts.graph.states[i, t],
                                receivers=test_rollouts.graph.receivers[i, t],
                                senders=test_rollouts.graph.senders[i, t],
                                node_type=test_rollouts.graph.node_type[i, t],
                                env_states=env_states_single,#[i], #test_rollouts.graph.env_states[i, t ],  # If not batched, else index if batched
                                connectivity=test_rollouts.graph.connectivity[i, t] if test_rollouts.graph.connectivity is not None else None,
                            )
                            n_env_states_single = type(n_env_states)(
                                **{k: (getattr(n_env_states, k)[i, t] if getattr(n_env_states, k) is not None else None)
                                for k in n_env_states._fields}
                            )
                            n_single_graph = GraphsTuple(
                                n_node=test_rollouts.next_graph.n_node[i],
                                n_edge=test_rollouts.next_graph.n_edge[i],
                                nodes=test_rollouts.next_graph.nodes[i, t],
                                edges=test_rollouts.next_graph.edges[i, t],
                                states=test_rollouts.next_graph.states[i, t],
                                receivers=test_rollouts.next_graph.receivers[i, t],
                                senders=test_rollouts.next_graph.senders[i, t],
                                node_type=test_rollouts.next_graph.node_type[i, t],
                                env_states=n_env_states_single,#[i], #test_rollouts.next_graph.env_states[i, t],  # If not batched, else index if batched
                                connectivity=test_rollouts.next_graph.connectivity[i, t] if test_rollouts.graph.connectivity is not None else None,
                            )

                            # local_obs.append(self.env_test.get_local_observations(single_graph))
                            # n_local_obs.append(self.env_test.get_local_observations(n_single_graph))
                            # local_cost.append(self.env_test.get_cost_local(single_graph))
                            # local_reward.append(self.env_test.get_reward_local(single_graph, action[i, t]))
                    # local_obs = np.array(local_obs).reshape(n_env_test, T, -1, local_obs[0].shape[-1])
                    # n_local_obs = np.array(n_local_obs).reshape(n_env_test, T, -1, n_local_obs[0].shape[-1])
                    # local_cost = np.array(local_cost).reshape(n_env_test, T, -1)
                    # local_reward = np.array(local_reward).reshape(n_env_test, T, -1)              

                            local_obs = self.env_test.get_local_observations(single_graph)
                            n_local_obs = self.env_test.get_local_observations(n_single_graph)
                            local_cost = self.env_test.get_cost_local(single_graph)
                            local_reward = self.env_test.get_reward_local(single_graph, action)

                            append('step', step)
                            append('time', time_since_start)
                            append('agent', agent)
                            append('goal', goal)
                            append('obstacle', obstacle)
                            append('state', state)                    
                            append('action', action)
                            append('local_obs', local_obs)
                            # # append('local_obs_enc', local_obs_enc)
                            append('local_reward', local_reward)
                            append('local_cost', local_cost)
                            append('n_local_obs', n_local_obs)
                            # append('n_local_obs_enc', n_local_obs_enc)
                            append('next_state', next_state)
                            append('reward', reward)
                            append('cost_mean', cost)
                            append('done', dones)
                            append('total_reward', total_reward)
                            append('reward_mean', reward_mean)
                            append('reward_min', reward_min)
                            append('reward_max', reward_max)
                            append('reward_final', reward_final)
                            append('cost', costs)
                            append('safe_frac_og', safe_frac)
                            append('unsafe_frac', unsafe_frac)

'''