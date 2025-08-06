import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr

from typing import NamedTuple, Tuple, Optional
from abc import ABC, abstractmethod

from dgppo.env.plot import render_mpe
from dgppo.trainer.data import Rollout
from dgppo.utils.graph import EdgeBlock, GetGraph, GraphsTuple
from dgppo.utils.typing import Action, Array, Cost, Done, Info, Reward, State, AgentState
from jaxtyping import Float
from dgppo.env.base import MultiAgentEnv
from dgppo.env.utils import get_node_goal_rng

# from typing import Optional
# import jax
# import jax.numpy as jnp
from jaxtyping import Array #, Float
from dataclasses import dataclass # for local observation encoder


class MPEEnvState(NamedTuple):
    agent: State
    goal: State
    obs: State

    @property
    def n_agent(self) -> int:
        return self.agent.shape[0]


MPEEnvGraphsTuple = GraphsTuple[State, MPEEnvState]


class MPE(MultiAgentEnv, ABC):

    AGENT = 0
    GOAL = 1
    OBS = 2

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_obs": 3,
        "obs_radius": 0.05,
        "default_area_size": 1.0,
        "dist2goal": 0.01
    }

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 128,
            dt: float = 0.03,
            params: dict = None
    ):
        area_size = MPE.PARAMS["default_area_size"] if area_size is None else area_size
        super(MPE, self).__init__(num_agents, area_size, max_step, dt, params)
        self.num_goals = self._num_agents

    @property
    def state_dim(self) -> int:
        return 4  # x, y, vx, vy

    @property
    def node_dim(self) -> int:
        return 7  # state dim (4) + indicator: agent: 001, goal: 010, obstacle: 100

    @property
    def edge_dim(self) -> int:
        return 4  # x_rel, y_rel, vx_rel, vy_rel

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
        # randomly generate agent and goal
        states, goals = get_node_goal_rng(
            key,
            self.area_size,
            2,
            self.num_agents,
            2 * self.params["car_radius"],
            None
        )

        # randomly generate obstacles
        def get_obs(inp): #inp =input .whr st. inp = (key, obs_candidate), PRNG key
            this_key, _ = inp
            use_key, this_key = jr.split(this_key, 2)
            return this_key, jr.uniform(use_key, (2,),
                                        minval=self.params['car_radius'] * 3,
                                        maxval=self.area_size - self.params['car_radius'] * 3)

        def non_valid_obs(inp):
            _, this_obs = inp
            dist_min_agents = jnp.linalg.norm(states - this_obs, axis=1).min()
            dist_min_goals = jnp.linalg.norm(goals - this_obs, axis=1).min()
            collide_agent = dist_min_agents <= self.params["car_radius"] + self.params["obs_radius"]
            collide_goal = dist_min_goals <= self.params["car_radius"] * 2 + self.params["obs_radius"]
            out_region = (jnp.any(this_obs < self.params["car_radius"] * 3) |
                          jnp.any(this_obs > self.area_size - self.params["car_radius"] * 3))
            return collide_agent | collide_goal | out_region

        def get_valid_obs(carry, inp):
            this_key = inp
            use_key, this_key = jr.split(this_key, 2)
            obs_candidate = jr.uniform(use_key, (2,), minval=0, maxval=self.area_size)
            _, valid_obs = jax.lax.while_loop(non_valid_obs, get_obs, (this_key, obs_candidate))
            return carry, valid_obs

        obs_keys = jr.split(key, self.params["n_obs"])
        _, obs = jax.lax.scan(get_valid_obs, None, obs_keys)

        # add zero velocity
        states = jnp.concatenate([states, jnp.zeros_like(states)], axis=1)
        goals = jnp.concatenate([goals, jnp.zeros_like(goals)], axis=1)
        obs = jnp.concatenate([obs, jnp.zeros_like(obs)], axis=1)

        env_state = MPEEnvState(states, goals, obs)

        return self.get_graph(env_state)

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = jnp.concatenate([agent_states[:, 2:], action * 10.], axis=1)
        n_state_agent_new = x_dot * self.dt + agent_states
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def step(
            self, graph: MPEEnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[MPEEnvGraphsTuple, Reward, Cost, Done, Info]:
        # get information from graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        obstacles = graph.type_states(type_idx=2, n_type=self.params["n_obs"]) if self.params["n_obs"] > 0 else None

        # calculate next graph
        action = self.clip_action(action)
        next_agent_states = self.agent_step_euler(agent_states, action)
        next_env_state = MPEEnvState(next_agent_states, goals, obstacles)
        info = {}

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # calculate reward and cost
        reward = self.get_reward(graph, action)
        cost = self.get_cost(graph)

        return self.get_graph(next_env_state), reward, cost, done, info

    @abstractmethod
    def get_reward(self, graph: MPEEnvGraphsTuple, action: Action) -> Reward:
        pass
    
    def get_local_observations(self, graph: MPEEnvGraphsTuple) -> Float[Array, 'num_agents local_obs_dim']:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        # print('agent_states', agent_states.shape)
        obstacle_states = graph.type_states(type_idx=2, n_type=self.params["n_obs"])
        
        # Positions only (first 2 dims)
        agent_pos = agent_states[:, :2]
        # print('agent_pos', agent_pos.shape)
        agent_vel = agent_states[:, 2:]  # optional
        
        obstacle_pos = obstacle_states[:, :2]
        comm_radius = self.params["comm_radius"]
        
        def obs_for_agent(i, pos_i):
            # Agent-to-agent relative positions (excluding self)
            rel_agent_pos = agent_pos - pos_i  # shape (N, 2)
            agent_dists = jnp.linalg.norm(rel_agent_pos, axis=1)
            agent_mask = (agent_dists < comm_radius) & (agent_dists > 1e-5)
            visible_agents = jnp.where(agent_mask[:, None], rel_agent_pos, 0.0)  # masked relative positions

            # Agent-to-obstacle relative positions
            rel_obs_pos = obstacle_pos - pos_i  # shape (M, 2)
            obs_dists = jnp.linalg.norm(rel_obs_pos, axis=1)
            obs_mask = obs_dists < comm_radius
            visible_obs = jnp.where(obs_mask[:, None], rel_obs_pos, 0.0)

            # Optionally: nearest goal
            goal_pos = graph.type_states(type_idx=1, n_type=self.num_goals)[:, :2]
            rel_goal_pos = goal_pos - pos_i
            goal_dists = jnp.linalg.norm(rel_goal_pos, axis=1)
            nearest_goal = rel_goal_pos[jnp.argmin(goal_dists)]

            # Flatten and concat everything
            # local_obs: shape=(128, 3, 16),
            # got (3, 2), (3, 3, 2), (3, 3, 2), (3, 2).
                # Reshape agent_vel and nearest_goal to (1, 2)
            # agent_vel_pad = agent_vel[i][None, :]      # shape (1, 2)
            # nearest_goal_pad = nearest_goal[None, :]   # shape (1, 2)
            
            obs_vector = jnp.concatenate([
                agent_vel[i][None],                                        # self velocity
                rel_agent_pos, #.flatten(),                 # all agents' relative pos (masked)
                rel_obs_pos, #.flatten(),                   # all obs relative pos (masked)
                nearest_goal[None]                            # relative position to nearest goal
            ]) #,axis=0)
            return obs_vector

        # Run for each agent
        local_obs = jax.vmap(obs_for_agent, in_axes=(0, 0))(jnp.arange(self.num_agents), agent_pos)

        return local_obs  # shape: (num_agents, local_obs_dim)

    
    # 2 (self vel) + 4×2 (relative agent pos) + 3×2 (relative obs pos) + 2 (goal) = 20 
    '''
    num_agents = 4
    num_obs = 3
    goal_dim = 2
    '''
    def get_cost_local(self, graph: MPEEnvGraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        agent_pos = agent_states[:, :2]

        obstacles = graph.type_states(type_idx=2, n_type=self.params["n_obs"])[:, :2]
        comm_radius = self.params["comm_radius"]
        car_radius = self.params["car_radius"]
        obs_radius = self.params["obs_radius"]

        def compute_agent_cost(i, agent_i_pos):
            # Agent-Agent local cost
            diffs = agent_pos - agent_i_pos  # shape (num_agents, 2)
            dists = jnp.linalg.norm(diffs, axis=1)
            mask = (dists > 0) & (dists < comm_radius)  # ignore self, keep only local
            local_dists = jnp.where(mask, dists, 1e6)
            min_dist = jnp.min(local_dists)
            agent_cost = car_radius * 2 - min_dist  # positive if too close

            # Agent-Obstacle local cost
            obs_diffs = obstacles - agent_i_pos  # shape (n_obs, 2)
            obs_dists = jnp.linalg.norm(obs_diffs, axis=1)
            obs_mask = obs_dists < comm_radius
            local_obs_dists = jnp.where(obs_mask, obs_dists, 1e6)
            min_obs_dist = jnp.min(local_obs_dists)
            obs_cost = car_radius + obs_radius - min_obs_dist

            return jnp.array([agent_cost, obs_cost])  # shape (n_cost,)

        costs = jax.vmap(compute_agent_cost, in_axes=(0, 0))(jnp.arange(self.num_agents), agent_pos)

        # Add margin
        eps = 0.5
        costs = jnp.where(costs <= 0.0, costs - eps, costs + eps)
        costs = jnp.clip(costs, a_min=-1.0)

        return costs  # shape (num_agents, 2)


    def get_cost(self, graph: MPEEnvGraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        obstacles = graph.type_states(type_idx=2, n_type=self.params["n_obs"])[:, :2]

        # collision between agents
        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        agent_cost: Array = self.params["car_radius"] * 2 - min_dist

        # collision between agents and obstacles
        if self.params["n_obs"] == 0:
            obs_cost = jnp.zeros(self.num_agents)
        else:
            dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(obstacles, 0), axis=-1)
            min_dist = jnp.min(dist, axis=1)
            obs_cost: Array = self.params["car_radius"] + self.params["obs_radius"] - min_dist

        cost = jnp.concatenate([agent_cost[:, None], obs_cost[:, None]], axis=1)
        assert cost.shape == (self.num_agents, self.n_cost)

        # add margin
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        cost = jnp.clip(cost, a_min=-1.0)

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
        render_mpe(rollout=rollout, video_path=video_path, side_length=self.area_size, dim=2, n_agent=self.num_agents,
                   n_obs=self.params['n_obs'], r=self.params["car_radius"], obs_r=self.params['obs_radius'],
                   cost_components=self.cost_components, Ta_is_unsafe=Ta_is_unsafe, viz_opts=viz_opts,
                   n_goal=self.num_goals, dpi=dpi, **kwargs)

    @abstractmethod
    def edge_blocks(self, state: MPEEnvState) -> list[EdgeBlock]:
        pass

    def get_graph(self, env_state: MPEEnvState) -> MPEEnvGraphsTuple:
        # node features
        # states
        node_feats = jnp.zeros((self.num_agents + self.num_goals + self.params["n_obs"], self.node_dim))
        node_feats = node_feats.at[:self.num_agents, :self.state_dim].set(env_state.agent)
        node_feats = node_feats.at[
                     self.num_agents: self.num_agents + self.num_goals, :self.state_dim].set(env_state.goal)
        if self.params["n_obs"] > 0:
            node_feats = node_feats.at[self.num_agents + self.num_goals:, :self.state_dim].set(env_state.obs)

        # indicators
        node_feats = node_feats.at[:self.num_agents, 6].set(1.0)
        node_feats = node_feats.at[self.num_agents: self.num_agents + self.num_goals, 5].set(1.0)
        if self.params["n_obs"] > 0:
            node_feats = node_feats.at[self.num_agents + self.num_goals:, 4].set(1.0)

        # node type
        node_type = -jnp.ones((self.num_agents + self.num_goals + self.params["n_obs"],), dtype=jnp.int32)
        node_type = node_type.at[:self.num_agents].set(MPE.AGENT)
        node_type = node_type.at[self.num_agents: self.num_agents + self.num_goals].set(MPE.GOAL)
        if self.params["n_obs"] > 0:
            node_type = node_type.at[self.num_agents + self.num_goals:].set(MPE.OBS)

        # edges
        edge_blocks = self.edge_blocks(env_state)

        # create graph
        states = jnp.concatenate([env_state.agent, env_state.goal], axis=0)
        if self.params["n_obs"] > 0:
            states = jnp.concatenate([states, env_state.obs], axis=0)
        return GetGraph(node_feats, node_type, edge_blocks, env_state, states).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.array([0.0, 0.0, -1.0, -1.0])
        upper_lim = jnp.array([self.area_size, self.area_size, 1.0, 1.0])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(2) * -1.0
        upper_lim = jnp.ones(2)
        return lower_lim, upper_lim




@dataclass
class LocalObservationEncoder:
    comm_radius: float
    car_radius: float
    obs_radius: float
    max_agents: int
    max_obs: int
    include_goal: bool = True

    def encode(
        self,
        graph: "MPEEnvGraphsTuple",
    ) -> Float[Array, "num_agents local_obs_dim"]:
        agent_states = graph.type_states(type_idx=0, n_type=None)
        agent_pos = agent_states[:, :2]
        agent_vel = agent_states[:, 2:] if agent_states.shape[1] > 2 else jnp.zeros_like(agent_pos)

        num_agents = agent_pos.shape[0]

        obstacle_states = graph.type_states(type_idx=2, n_type=None)
        obstacle_pos = obstacle_states[:, :2] if obstacle_states.shape[0] > 0 else jnp.zeros((0, 2))

        if self.include_goal:
            goal_states = graph.type_states(type_idx=1, n_type=None)
            goal_pos = goal_states[:, :2] if goal_states.shape[0] > 0 else jnp.zeros((0, 2))

        def pad_and_mask(rel_pos: Array, max_count: int) -> Float[Array, "max_count 2"]:
            count = rel_pos.shape[0]
            padded = jnp.zeros((max_count, 2))
            mask = jnp.zeros((max_count,))
            n = jnp.minimum(count, max_count)
            padded = padded.at[:n].set(rel_pos[:n])
            mask = mask.at[:n].set(1.0)
            return padded, mask

        def single_agent_obs(i, pos_i, vel_i):
            # Agents
            diffs = agent_pos - pos_i
            dists = jnp.linalg.norm(diffs, axis=1)
            mask_agents = (dists < self.comm_radius) & (dists > 1e-5)
            rel_agents = diffs[mask_agents]
            padded_agents, agent_mask = pad_and_mask(rel_agents, self.max_agents)

            # Obstacles
            obs_diffs = obstacle_pos - pos_i
            obs_dists = jnp.linalg.norm(obs_diffs, axis=1)
            mask_obs = obs_dists < self.comm_radius
            rel_obs = obs_diffs[mask_obs]
            padded_obs, obs_mask = pad_and_mask(rel_obs, self.max_obs)

            # Nearest goal (optional)
            if self.include_goal and goal_pos.shape[0] > 0:
                rel_goal_pos = goal_pos - pos_i
                goal_dists = jnp.linalg.norm(rel_goal_pos, axis=1)
                nearest_goal = rel_goal_pos[jnp.argmin(goal_dists)]
            else:
                nearest_goal = jnp.zeros((2,))

            obs_vector = jnp.concatenate([
                vel_i,                                 # (2,)
                padded_agents.flatten(),               # (max_agents * 2,)
                padded_obs.flatten(),                  # (max_obs * 2,)
                agent_mask,                            # (max_agents,)
                obs_mask,                              # (max_obs,)
                nearest_goal if self.include_goal else jnp.zeros((2,))  # (2,)
            ])
            return obs_vector

        local_obs = jax.vmap(single_agent_obs, in_axes=(0, 0, 0))(
            jnp.arange(num_agents), agent_pos, agent_vel
        )

        return local_obs  # shape: (num_agents, local_obs_dim)

'''
usage example:
encoder = LocalObservationEncoder(
    comm_radius=0.5,
    car_radius=0.05,
    obs_radius=0.05,
    max_agents=4,
    max_obs=3,
    include_goal=True
)

local_obs = encoder.encode(graph)  # (num_agents, local_obs_dim)

local_obs_dim = 
    2 (self velocity)
  + 2 * max_agents (rel agent pos)
  + 2 * max_obs (rel obstacle pos)
  + max_agents (agent mask)
  + max_obs (obstacle mask)
  + 2 (nearest goal)

'''