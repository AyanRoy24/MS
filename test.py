import argparse
import datetime
import functools as ft
import os
import pathlib
import h5py

import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml

from dgppo.algo import make_algo
from dgppo.env import make_env
from dgppo.trainer.utils import test_rollout
from dgppo.utils.graph import GraphsTuple
from dgppo.utils.utils import jax_jit_np, jax_vmap
from dgppo.utils.typing import Array


def test(args):
    print(f"> Running test.py {args}")

    stamp_str = datetime.datetime.now().strftime("%m%d-%H%M")

    # set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)
    np.random.seed(args.seed)

    # load config
    with open(os.path.join(args.path, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)

    # create environments
    num_agents = config.num_agents if args.num_agents is None else args.num_agents
    env = make_env(
        env_id=config.env if args.env is None else args.env,
        num_agents=num_agents,
        num_obs=config.obs if args.obs is None else args.obs,
        max_step=args.max_step,
        full_observation=args.full_observation,
    )

    # create algorithm
    path = args.path
    model_path = os.path.join(path, "models")
    if args.step is None:
        models = os.listdir(model_path)
        step = max([int(model) for model in models if model.isdigit()])
    else:
        step = args.step
    print("step: ", step)

    algo = make_algo(
        algo=config.algo,
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        cost_weight=config.cost_weight,
        actor_gnn_layers=config.actor_gnn_layers,
        Vl_gnn_layers=config.Vl_gnn_layers,
        Vh_gnn_layers=config.Vh_gnn_layers if hasattr(config, "Vh_gnn_layers") else 1,
        lr_actor=config.lr_actor,
        lr_Vl=config.lr_Vl,
        max_grad_norm=2.0,
        seed=config.seed,
        use_rnn=config.use_rnn,
        rnn_layers=config.rnn_layers,
        use_lstm=config.use_lstm,
    )
    algo.load(model_path, step)
    if args.stochastic:
        def act_fn(x, z, rnn_state, key):
            action, _, new_rnn_state = algo.step(x, z, rnn_state, key)
            return action, new_rnn_state
        act_fn = jax.jit(act_fn)
    else:
        act_fn = algo.act
    act_fn = jax.jit(act_fn)
    init_rnn_state = algo.init_rnn_state

    # set up keys
    test_key = jr.PRNGKey(args.seed)
    # test_keys = jr.split(test_key, 1_000)[: args.epi]
    test_keys = jr.split(test_key, args.epi)
    test_keys = test_keys[args.offset:]

    # create rollout function
    rollout_fn = ft.partial(test_rollout,
                            env,
                            act_fn,
                            init_rnn_state,
                            stochastic=args.stochastic)
    rollout_fn = jax_jit_np(rollout_fn)

    def unsafe_mask(graph_: GraphsTuple) -> Array:
        cost = env.get_cost(graph_)
        return jnp.any(cost >= 0.0, axis=-1) # cost  = T, num_agents, cost_dim

    is_unsafe_fn = jax_jit_np(jax_vmap(unsafe_mask))

    # test results
    rewards = []
    costs = []
    rollouts = []
    is_unsafes = []
    rates = []

    # test
   
    with h5py.File(args.name, 'a') as f:

        def append(name, arr):
            arr = np.array(arr)
            # If arr has no batch dimension, add one
            if arr.shape == ():  # scalar
                arr = arr[None]
            elif arr.ndim == 1:  # vector
                arr = arr[None, :]
            elif arr.ndim > 1 and arr.shape[0] != 1:
                arr = arr[None, ...]  # add batch dimension

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

        local_cost_all = []
        local_obs_all = []
        local_reward_all = []
        next_state_all = []
        for i_epi in range(args.epi):
            # 128 steps per episode
            key_x0, _ = jr.split(test_keys[i_epi], 2)
            rollout = rollout_fn(key_x0)
            is_unsafes.append(is_unsafe_fn(rollout.graph))
            # is_safe = ~is_unsafe_fn(rollout.graph)
            # safe_frac_per_T = is_safe.mean(axis=1)

            epi_reward = rollout.rewards.sum()
            epi_cost = rollout.costs.max()
            rewards.append(epi_reward)
            costs.append(epi_cost)
            rollouts.append(rollout)
            safe_rate = 1 - is_unsafes[-1]#.max(axis=0)#.mean() # for last episode only
            safe_rate_scalar = np.mean(safe_rate) * 100  # or safe_rate.item() if shape is ()

            print(f"epi: {i_epi}, reward: {epi_reward:.3f}, cost: {epi_cost:.3f}, safe rate: {safe_rate_scalar * 100:.3f}%")
            
            rates.append(np.array(safe_rate))

            append('Rewards', rewards)
            append('Costs', costs)
            append('Rates', rates)
            
            agent = rollout.graph.env_states.agent#.mean(axis=0).flatten()  # Assuming agent is a field in the graph
            # print('agent: ', agent.shape)
            goal = rollout.graph.env_states.goal#.mean(axis=0).flatten()  # Assuming goal is a field in the graph
            # print('goal: ', goal.shape)
            obstacle = rollout.graph.env_states.obs#.mean(axis=0).flatten() if rollout.graph.env_states.obs is not None else None #obstacle.center.mean(axis=0).flatten() 
            # print('obstacle: ', obstacle.shape if obstacle is not None else 'None')
            state = np.concatenate([agent, goal, obstacle],axis=-2) if obstacle is not None else np.concatenate([agent, goal],axis=-2)
            
            action = rollout.actions#.mean(axis=0) # Ensure actions are numpy array
            
            n_agent = rollout.next_graph.env_states.agent#.mean(axis=0).flatten()  # Assuming agent is a field in the graph
            n_goal = rollout.next_graph.env_states.goal#.mean(axis=0).flatten()  # Assuming goal is a field in the graph
            n_obstacle = rollout.next_graph.env_states.obs#.mean(axis=0).flatten() if rollout.graph.env_states.obs is not None else None #obstacle.center.mean(axis=0).flatten() 
            next_state = np.concatenate([n_agent, n_goal, n_obstacle],axis=-2) if obstacle is not None else np.concatenate([agent, goal],axis=-2)

            cost = rollout.costs#.mean(axis=0)  # Assuming cost is a field in the graph
            reward = rollout.rewards#.mean(axis=0)  # Assuming reward is a field in the graph
            done = rollout.dones
            # Store episode data in test.h5
            # unsafe_frac = np.mean(rollout.costs.max(axis=-1).max(axis=-2) >= 1e-6)
            unsafe_frac = rollout.costs.max(axis=-1).max(axis=-2) >= 1e-6
            safe_frac = 1 - unsafe_frac

            local_obs_epi = []
            local_reward_epi = []
            local_cost_epi = []
            # next_state_epi = []
            # for t in range(rollout.rewards.shape[0]):  # T
            for t in range(128): 
                # Reconstruct env_states for this timestep
                env_states = rollout.graph.env_states
                env_states_single = type(env_states)(
                    **{k: (getattr(env_states, k)[t] if getattr(env_states, k) is not None else None)
                    for k in env_states._fields}
                )
                # Reconstruct GraphsTuple for this timestep
                s_graph = GraphsTuple(
                    n_edge=rollout.graph.n_edge[t],
                    n_node=rollout.graph.n_node[t],
                    nodes=rollout.graph.nodes[t],
                    edges=rollout.graph.edges[t],
                    states=rollout.graph.states[t],
                    receivers=rollout.graph.receivers[t],
                    senders=rollout.graph.senders[t],
                    node_type=rollout.graph.node_type[t],
                    env_states=env_states_single,
                    connectivity=rollout.graph.connectivity[t] if rollout.graph.connectivity is not None else None,
                )
                # Get local obs, reward, cost
                local_obs = env.get_local_observations(s_graph)
                local_reward = env.get_reward_local(s_graph, rollout.actions[t])
                local_cost = env.get_cost_local(s_graph)
                # Save to HDF5
                local_obs_epi.append(local_obs)
                local_reward_epi.append(local_reward)
                local_cost_epi.append(local_cost)
            
            local_obs_epi = np.stack(local_obs_epi)         # (128, num_agents, 16)
            local_reward_epi = np.stack(local_reward_epi)   # (128, num_agents)
            local_cost_epi = np.stack(local_cost_epi) 
            # next_state_epi = np.stack([next_state] * local_obs_epi.shape[0])  # (128, num_agents, 4)    
            
            local_obs_all.append(local_obs_epi)
            local_reward_all.append(local_reward_epi)
            local_cost_all.append(local_cost_epi)
            # next_state_all.append(next_state)

            
        
        local_obs_all = np.stack(local_obs_all)         # (100, 128, 3, 16)
        local_reward_all = np.stack(local_reward_all)   # (100, 128, 3)
        local_cost_all = np.stack(local_cost_all)       # (100, 128, 3, 2)
        # next_state_all = np.stack(next_state_all)
    
        
        append('state', np.array(state))
        append('next_state', np.array(next_state))
        append('action', np.array(action))
        append('reward', np.array([reward]))
        append('cost', np.array([cost]))
        append('done', np.array([done])) #, dtype=np.uint8))

        append('safe_rate', np.array([safe_rate]))
        append('safe_frac', np.array([safe_frac]))
       

        append('local_obs', local_obs_all)
        append('local_reward', local_reward_all)
        append('local_cost', local_cost_all)
        # append('next_state', next_state_all) #np.array(next_state))
        append('reward_mean', np.mean(rewards))
        append('cost_mean', np.mean(costs))
        is_unsafe = np.max(np.stack(is_unsafes), axis=1)
        safe_mean, safe_std = (1 - is_unsafe).mean(), (1 - is_unsafe).std()
        append('safe_mean', np.array([safe_mean]))
        
                
    # is_unsafe = np.max(np.stack(is_unsafes), axis=1)
    # safe_mean, safe_std = (1 - is_unsafe).mean(), (1 - is_unsafe).std()
    # append('safe_mean', np.array([safe_mean]))
    
    print(
        f"reward: {np.mean(rewards):.3f}, min/max reward: {np.min(rewards):.3f}/{np.max(rewards):.3f}, "
        f"cost: {np.mean(costs):.3f}, min/max cost: {np.min(costs):.3f}/{np.max(costs):.3f}, "
        f"safe_rate: {safe_mean * 100:.3f}%"
    )

    # save results
    if args.log:
        with open(os.path.join(path, "test_log.csv"), "a") as f:
            f.write(f"{env.num_agents},{args.epi},{env.max_episode_steps},"
                    f"{env.area_size},{env.params['n_obs']},"
                    f"{safe_mean * 100:.3f},{safe_std * 100:.3f}\n")

    # make video
    if args.no_video:
        return

    videos_dir = pathlib.Path(path) / "videos" / f"{step}"
    videos_dir.mkdir(exist_ok=True, parents=True)
    for ii, (rollout, Ta_is_unsafe) in enumerate(zip(rollouts, is_unsafes)):
        safe_rate = rates[ii] * 100
        video_name = f"n{num_agents}_epi{ii:02}_reward{rewards[ii]:.3f}_cost{costs[ii]:.3f}_sr{safe_rate:.0f}"
        viz_opts = {}
        video_path = videos_dir / f"{stamp_str}_{video_name}.mp4"
        env.render_video(rollout, video_path, Ta_is_unsafe, viz_opts, dpi=args.dpi)


def main():
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument("--path", type=str, required=True)

    # custom arguments
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("--epi", type=int, default=5)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--steps", type=str, default=None, help="Comma-separated list of checkpoint steps to test")
    parser.add_argument("--mult", type=int, default=None, help="Test all checkpoints that are a multiple of this number")

    parser.add_argument("--obs", type=int, default=None)
    parser.add_argument("--stochastic", action="store_true", default=False)
    parser.add_argument("--full-observation", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--log", action="store_true", default=False)

    # default arguments
    parser.add_argument("-n", "--num-agents", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    # test(args)
    
    if args.steps is not None:
        # Run test for each checkpoint in the provided list
        steps = [int(s) for s in args.steps.split(",") if s.strip().isdigit()]
        for step in steps:
            print(f"\n=== Running test for checkpoint step: {step} ===\n")
            args.step = step
            test(args)
    elif args.step is not None:
        test(args)
    elif args.mult is not None:
        # Run test for each checkpoint that is a multiple of step_mult
        import os
        model_path = os.path.join(args.path, "models")
        steps = sorted([int(f) for f in os.listdir(model_path) if f.isdigit() and int(f) != 0])
        steps = [s for s in steps if s % args.mult == 0]
        for step in steps:
            print(f"\n=== Running test for checkpoint step: {step} ===\n")
            args.step = step
            test(args)
    else:
        # Run test for each checkpoint in models folder
        import os
        model_path = os.path.join(args.path, "models")
        steps = sorted([int(f) for f in os.listdir(model_path) if f.isdigit() and int(f) != 0])
        for step in steps:
            print(f"\n=== Running test for checkpoint step: {step} ===\n")
            args.step = step
            test(args)

if __name__ == "__main__":
    # with ipdb.launch_ipdb_on_exception():
    main()



'''
T, num_agents, 2 position 2 velocity as per state_dim property in mpe/base
agent:  (128, 3, 4)
goal:  (128, 3, 4)
obstacle:  (128, 3, 4)
'''

'''
Costs: shape=(200, 1), dtype=float32
Rates: shape=(200, 128, 3), dtype=int64
Rewards: shape=(200, 1), dtype=float32
action: shape=(200, 128, 3, 2), dtype=float32
cost: shape=(200, 128, 3, 2), dtype=float32
cost_mean: shape=(200,), dtype=float32
done: shape=(200, 128), dtype=bool
local_cost: shape=(200, 128, 3, 2), dtype=float32
local_obs: shape=(200, 128, 3, 8, 2), dtype=float32
local_reward: shape=(200, 128, 3), dtype=float32
next_state: shape=(200, 128, 9, 4), dtype=float32
reward: shape=(200, 128, 3), dtype=float32
reward_mean: shape=(200,), dtype=float32
safe_frac: shape=(200, 3), dtype=int64
safe_mean: shape=(200, 1), dtype=float64
safe_rate: shape=(200, 128, 3), dtype=int64
state: shape=(200, 128, 9, 4), dtype=float32
'''

    
        
        # append('epi', np.array([i_epi]))
        # append('step', np.array([step]))
        # timestep = step + i_epi
        # append('timestep', np.array([timestep]))
        # append('agent', np.array(agent))
        # append('goal', np.array(goal))
                    
        # if obstacle is not None:
        #     append('obstacle', np.array(obstacle))
        # else:
        #     append('obstacle', np.array([]))



            # Vh_values = []
            # Vl_values = []
            # rnn_state = algo.init_rnn_state
        
            # for t in range(rollout.graph.n_node.shape[0]):       
            #     # graph_t = jax.tree_map(lambda x: x[t], rollout.graph)
            #     # graph_t = jax.tree.map(lambda x: x[t], rollout.graph)
            #     graph_t = jax.tree.map(lambda x: jnp.asarray(x[t]), rollout.graph)
            #     # graph_t_mean = jax.tree.map(lambda x: x.mean(axis=0, keepdims=True) if hasattr(x, 'mean') else x, graph_t)
            #     # graph_t_mean = jax.tree.map(
            #     #     lambda x: x.mean(axis=0, keepdims=True) if hasattr(x, 'mean') and x.ndim > 0 and x.shape[0] > 1 else x,
            #     #     graph_t
            #     # )
            #     # Prepare rnn_state with batch size 1
            #     # rnn_state_t = rnn_state
            #     # if isinstance(rnn_state, (tuple, list)):
            #     #     rnn_state_t = tuple(
            #     #         s[:, :1, ...] if s.ndim > 1 else s for s in rnn_state
            #     #     )
            #     # elif hasattr(rnn_state, 'ndim') and rnn_state.ndim > 1:
            #     #     rnn_state_t = rnn_state[:, :1, ...]
            #     # Now call the value functions
            #     Vh_t = algo.get_Vh(graph_t, rnn_state)  # shape: (1, n_cost)
            #     # Vl_t = algo.Vl.get_value(algo.Vl_train_state.params, graph_t_mean, rnn_state_t)[0]  # shape: (1, 1)
            #     Vh_values.append(np.array(Vh_t))
            #     # Vl_values.append(np.array(Vl_t))
            #     # If using RNN, update rnn_state here if needed

            # Vh_values = np.stack(Vh_values)  # shape: (T, n_agents, n_cost)
            # Vl_values = np.stack(Vl_values)  # shape: (T, n_agents, 1)



        # def append_or_create(name, arr):
        #     if name in f:
        #         dset = f[name]
        #         dset.resize(dset.shape[0] + 1, axis=0)
        #         dset[-1] = arr
        #     else:
        #         f.create_dataset(name, data=arr, maxshape=(None,), compression='gzip')


    # if args.step is not None:
    #     test(args)
    # else:
    #     # Run test for each checkpoint in models folder
    #     import os
    #     model_path = os.path.join(args.path, "models")
    #     steps = sorted([int(f) for f in os.listdir(model_path) if f.isdigit() and int(f) != 0])
    #     for step in steps:
    #         print(f"\n=== Running test for checkpoint step: {step} ===\n")
    #         args.step = step
    #         test(args)
