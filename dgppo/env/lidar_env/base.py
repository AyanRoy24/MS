import pathlib
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import functools as ft

from typing import NamedTuple, Tuple, Optional
from abc import ABC, abstractmethod

from jaxtyping import Float

from ...trainer.data import Rollout
from ...utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ...utils.typing import Action, Array, Cost, Done, Info, Pos2d, Reward, State, AgentState
from ...utils.utils import merge01, jax_vmap
from ..base import MultiAgentEnv
from dgppo.env.obstacle import Obstacle, Rectangle
from dgppo.env.plot import render_lidar
from dgppo.env.utils import get_lidar, get_node_goal_rng


class LidarEnvState(NamedTuple):
    agent: State
    goal: State
    obstacle: Obstacle

    @property
    def n_agent(self) -> int:
        return self.agent.shape[0]


LidarEnvGraphsTuple = GraphsTuple[State, LidarEnvState]


class LidarEnv(MultiAgentEnv, ABC):

    AGENT = 0
    GOAL = 1
    OBS = 2

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.3],
        "n_obs": 3,
        "default_area_size": 1.5,
        "dist2goal": 0.01,
        "top_k_rays": 8,
    }

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 128,
            dt: float = 0.03,
            params: dict = None
    ):
        area_size = LidarEnv.PARAMS["default_area_size"] if area_size is None else area_size
        super(LidarEnv, self).__init__(num_agents, area_size, max_step, dt, params)
        self.create_obstacles = jax_vmap(Rectangle.create)
        self.num_goals = self._num_agents

    @property
    def state_dim(self) -> int:
        return 4  # x, y, vx, vy

    @property
    def node_dim(self) -> int:
        return 7  # state dim (4) + indicator: agent: 001, goal: 010, obstacle: 100

    @property
    def edge_dim(self) -> int:
        return 4  # x_rel, y_rel, vx_vel, vy_vel

    @property
    def action_dim(self) -> int:
        return 2  # ax, ay

    @property
    def n_cost(self) -> int:
        return 2

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obs collisions"

    def reset(self, key: Array) -> GraphsTuple:
        # randomly generate obstacles
        n_rng_obs = self._params["n_obs"]
        assert n_rng_obs >= 0
        if n_rng_obs == 0:
            obstacles = None
        else:
            obstacle_key, key = jr.split(key, 2)
            obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=self.area_size)
            length_key, key = jr.split(key, 2)
            obs_len = jr.uniform(
                length_key,
                (self._params["n_obs"], 2),
                minval=self._params["obs_len_range"][0],
                maxval=self._params["obs_len_range"][1],
            )
            theta_key, key = jr.split(key, 2)
            obs_theta = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * np.pi)
            obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)

        # randomly generate agent and goal
        states, goals = get_node_goal_rng(
            key, self.area_size, 2, self.num_agents, 2.2 * self.params["car_radius"], obstacles)
        states = jnp.concatenate(
            [states, jnp.zeros((self.num_agents, self.state_dim - states.shape[1]), dtype=states.dtype)], axis=1)
        goals = jnp.concatenate(
            [goals, jnp.zeros((self.num_goals, self.state_dim - goals.shape[1]), dtype=goals.dtype)], axis=1)

        assert states.shape == (self.num_agents, self.state_dim)
        assert goals.shape == (self.num_goals, self.state_dim)
        env_states = LidarEnvState(states, goals, obstacles)

        # get lidar data
        lidar_data = self.get_lidar_data(states, obstacles)

        return self.get_graph(env_states, lidar_data)

    def get_lidar_data(self, states: State, obstacles: Obstacle) -> Float[Array, "n_agent top_k_rays 2"]:
        lidar_data = None
        if self.params["n_obs"] > 0:
            get_lidar_vmap = jax_vmap(
                ft.partial(
                    get_lidar,
                    obstacles=obstacles,
                    num_beams=self._params["n_rays"],
                    sense_range=self._params["comm_radius"],
                    max_returns=self._params["top_k_rays"],
                )
            )
            lidar_data = get_lidar_vmap(states[:, :2])
            assert lidar_data.shape == (self.num_agents, self._params["top_k_rays"], 2)
        return lidar_data

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        """By default, use double integrator dynamics"""
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = jnp.concatenate([agent_states[:, 2:], action * 10.], axis=1)
        n_state_agent_new = x_dot * self.dt + agent_states
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def step(
            self, graph: LidarEnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[LidarEnvGraphsTuple, Reward, Cost, Done, Info]:
        # get information from graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        obstacles = graph.env_states.obstacle if self.params['n_obs'] > 0 else None

        # calculate next states
        action = self.clip_action(action)
        next_agent_states = self.agent_step_euler(agent_states, action)
        next_state = LidarEnvState(next_agent_states, goals, obstacles)
        lidar_data_next = self.get_lidar_data(next_agent_states, obstacles)
        info = {}

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # compute reward and cost
        reward = self.get_reward(graph, action)
        cost = self.get_cost(graph)
        assert reward.shape == tuple()

        return self.get_graph(next_state, lidar_data_next), reward, cost, done, info

    @abstractmethod
    def get_reward(self, graph: LidarEnvGraphsTuple, action: Action) -> Reward:
        pass

    def get_cost(self, graph: GraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)

        # collision between agents
        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        agent_cost: Array = self.params["car_radius"] * 2 - min_dist

        # collision between agents and obstacles
        if self.params['n_obs'] == 0:
            obs_cost = jnp.zeros((self.num_agents,)).astype(jnp.float32)
        else:
            obs_pos = graph.type_states(type_idx=2, n_type=self._params["top_k_rays"] * self.num_agents)[:, :2]
            obs_pos = jnp.reshape(obs_pos, (self.num_agents, self._params["top_k_rays"], 2))
            dist = jnp.linalg.norm(obs_pos - agent_pos[:, None, :], axis=-1)  # (n_agent, top_k_rays)
            obs_cost: Array = self.params["car_radius"] - dist.min(axis=1)  # (n_agent,)

        cost = jnp.concatenate([agent_cost[:, None], obs_cost[:, None]], axis=1)
        assert cost.shape == (self.num_agents, self.n_cost)

        # add margin
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        cost = jnp.clip(cost, a_min=-1.0, a_max=1.0)

        return cost

    def render_video(
            self,
            rollout: Rollout,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            dpi: int = 100,
            **kwargs
    ) -> None:
        render_lidar(rollout=rollout, video_path=video_path, side_length=self.area_size, dim=2, n_agent=self.num_agents,
                     n_rays=self.params["top_k_rays"] if self.params["n_obs"] > 0 else 0,
                     r=self.params["car_radius"], cost_components=self.cost_components,
                     Ta_is_unsafe=Ta_is_unsafe, viz_opts=viz_opts, n_goal=self.num_goals, dpi=dpi, **kwargs)

    @abstractmethod
    def edge_blocks(self, state: LidarEnvState, lidar_data: Optional[Pos2d] = None) -> list[EdgeBlock]:
        pass

    def get_graph(self, state: LidarEnvState, lidar_data: Pos2d = None) -> GraphsTuple:
        n_hits = self._params["top_k_rays"] * self.num_agents if self.params["n_obs"] > 0 else 0
        n_nodes = self.num_agents + self.num_goals + n_hits

        if lidar_data is not None:
            lidar_data = merge01(lidar_data)

        # node features
        # states
        node_feats = jnp.zeros((self.num_agents + self.num_goals + n_hits, self.node_dim))
        node_feats = node_feats.at[: self.num_agents, :self.state_dim].set(state.agent)
        node_feats = node_feats.at[self.num_agents: self.num_agents + self.num_goals, :self.state_dim].set(state.goal)
        if lidar_data is not None:
            node_feats = node_feats.at[-n_hits:, :2].set(lidar_data)

        # indicators
        node_feats = node_feats.at[: self.num_agents, self.state_dim + 2].set(1.)  # agent
        node_feats = (
            node_feats.at[self.num_agents: self.num_agents + self.num_goals, self.state_dim + 1].set(1.))  # goal
        if n_hits > 0:
            node_feats = node_feats.at[-n_hits:, self.state_dim].set(1.)  # obs feats

        # node type
        node_type = -jnp.ones(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[: self.num_agents].set(LidarEnv.AGENT)
        node_type = node_type.at[self.num_agents: self.num_agents + self.num_goals].set(LidarEnv.GOAL)
        if n_hits > 0:
            node_type = node_type.at[-n_hits:].set(LidarEnv.OBS)

        # edge blocks
        edge_blocks = self.edge_blocks(state, lidar_data)

        # create graph
        states = jnp.concatenate([state.agent, state.goal], axis=0)
        if lidar_data is not None:
            lidar_states = jnp.concatenate(
                [lidar_data, jnp.zeros((n_hits, self.state_dim - lidar_data.shape[1]))], axis=1)
            states = jnp.concatenate([states, lidar_states], axis=0)
        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=states
        ).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.array([0., 0., -0.5, -0.5])
        upper_lim = jnp.array([self.area_size, self.area_size, 0.5, 0.5])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(2) * -1.0
        upper_lim = jnp.ones(2)
        return lower_lim, upper_lim


'''
The role of `n_cost` in the output of `Vh` (the value predicted by `ValueNet`) is to determine **how many separate safety-related quantities (cost dimensions) the Control Barrier Function (CBF) value network predicts for each agent at each timestep**.

### Details:

- In your environment (see base.py), `n_cost` is defined as:
  ```python
  @property
  def n_cost(self) -> int:
      return 2
  ```
  This means there are **two cost components**: typically, `"agent collisions"` and `"obs collisions"`.

- When you instantiate `ValueNet` in `DGPPO`, you set:
  ```python
  self.Vh = ValueNet(
      ...,
      n_out=env.n_cost,
      ...
  )
  ```
  So, `Vh` is configured to output a vector of length `n_cost` for each agent (or per cost dimension, if decomposed).

- **In practice:**  
  For a batch of graphs, the output shape of `Vh` will be something like  
  `(batch_size, time_steps, num_agents, n_cost)`  
  where each entry `[b, t, a, c]` gives the CBF value for agent `a` and cost component `c` at time `t` in batch `b`.

- **Why is this important?**  
  - Each cost dimension can represent a different type of safety constraint (e.g., agent-agent collision, agent-obstacle collision).
  - The CBF value network (`Vh`) must learn to predict the "safety value" for each of these constraints, so the policy can be penalized for violating any of them.

### Summary Table

| Parameter | Role in Vh Output                                  |
|-----------|----------------------------------------------------|
| `n_cost`  | Number of cost/safety dimensions predicted by Vh   |
| Vh shape  | (..., n_agents, n_cost)                            |
| Meaning   | For each agent, Vh predicts a value for each cost  |

---

**In short:**  
`n_cost` controls how many safety-related values the CBF value network (`Vh`) predicts per agent, allowing the agent to learn and enforce multiple safety constraints simultaneously.

'''


'''
If your environment state is split into three components—**agent**, **goal**, and **obstacle**—and you store them separately in your offline dataset, **the policy can still learn effectively** as long as you reconstruct the full state (or observation) in the same way as during online training or environment interaction.

### How does the policy learn from split state components?

1. **Consistent Representation:**  
   - During both online and offline training, your environment and policy expect the state as a combination of agent, goal, and obstacle information.
   - When you store these separately in your offline data, you must **recombine** them into the same structure (e.g., a `LidarEnvState` or a concatenated array) before feeding them to the policy or value network.

2. **Data Loader Responsibility:**  
   - Your offline data loader should read the `agent`, `goal`, and `obstacle` arrays for each timestep and reconstruct the full state object or observation tensor.
   - This reconstructed state is then passed to the policy network as input, just as it would be during online rollouts.

3. **Policy Input:**  
   - The policy does **not** care how the state was stored, only that it receives the correct input format (e.g., a concatenated array, a named tuple, or a graph structure).
   - As long as you reconstruct the state correctly, the policy will learn as expected.

### Example

Suppose your offline HDF5 file has:
- `/state/agent` (N, num_agents, state_dim)
- `/state/goal` (N, num_goals, state_dim)
- `/state/obstacle` (N, ...)

When loading a batch:
```python
agent = h5file['state/agent'][i]
goal = h5file['state/goal'][i]
obstacle = h5file['state/obstacle'][i]
state = LidarEnvState(agent, goal, obstacle)  # or concatenate as needed
# Now pass `state` to the policy
action = policy.apply(params, state)
```

### Why is this modular storage useful?

- **Flexibility:** You can analyze or modify individual components (e.g., swap goals, add obstacles) for data augmentation or ablation studies.
- **Clarity:** Each part of the state is clearly labeled and accessible.
- **Compatibility:** As long as your data loader reconstructs the state as expected, your policy and value networks will work the same as in online RL.

---

**In summary:**  
Storing the state as three separate values (agent, goal, obstacle) does **not** hinder policy learning, as long as you reconstruct the full state in the same format expected by your policy network during training and inference. This is a common and robust practice in offline RL and dataset design.

'''