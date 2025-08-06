import jax.numpy as jnp

from typing import Optional

from dgppo.utils.graph import EdgeBlock
from dgppo.utils.typing import Action, Array, Pos2d, Reward, State
from dgppo.env.lidar_env.base import LidarEnv, LidarEnvState, LidarEnvGraphsTuple
from dgppo.utils.utils import jax_vmap


class LidarSpread(LidarEnv):

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
        area_size = LidarSpread.PARAMS["default_area_size"] if area_size is None else area_size
        super(LidarSpread, self).__init__(num_agents, area_size, max_step, dt, params)

    def get_reward_mod(self, graph: LidarEnvGraphsTuple, action: Action) -> Reward:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        agent_pos = agent_states[:, :2]
        goal_pos = goals[:, :2]
        dist2goal = jnp.linalg.norm(jnp.expand_dims(goal_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        # For each agent, find its closest goal
        agent2goal_dist = dist2goal.min(axis=0)  # shape: (num_agents,)
        reward = -agent2goal_dist * 0.01  # shape: (num_agents,)
        reward -= jnp.where(agent2goal_dist > self._params["dist2goal"], 1.0, 0.0) * 0.001
        reward -= (jnp.linalg.norm(action, axis=1) ** 2) * 0.0001
        # Add other penalties per agent as needed
        return reward.astype(jnp.float32)
    
    def get_reward(self, graph: LidarEnvGraphsTuple, action: Action) -> Reward:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        reward = jnp.zeros(()).astype(jnp.float32)

        # each goal finds the nearest agent
        agent_pos = agent_states[:, :2]
        goal_pos = goals[:, :2]
        dist2goal = jnp.linalg.norm(jnp.expand_dims(goal_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1).min(axis=1)
        reward -= dist2goal.mean() * 0.01

        # not reaching goal penalty
        reward -= jnp.where(dist2goal > self._params["dist2goal"], 1.0, 0.0).mean() * 0.001

        # action penalty
        reward -= (jnp.linalg.norm(action, axis=1) ** 2).mean() * 0.0001

        return reward

    def state2feat(self, state: State) -> Array:
        return state

    def edge_blocks(self, state: LidarEnvState, lidar_data: Optional[Pos2d] = None) -> list[EdgeBlock]:
        # agent - agent connection
        agent_pos = state.agent[:, :2]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        edge_feats = (jax_vmap(self.state2feat)(state.agent)[:, None, :] -
                      jax_vmap(self.state2feat)(state.agent)[None, :, :])
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(edge_feats, agent_agent_mask, id_agent, id_agent)

        # agent - goal connection
        id_goal = jnp.arange(self.num_agents, self.num_agents + self.num_goals)
        agent_goal_mask = jnp.ones((self.num_agents, self.num_goals))
        agent_goal_feats = (jax_vmap(self.state2feat)(state.agent)[:, None, :] -
                            jax_vmap(self.state2feat)(state.goal)[None, :, :])
        agent_goal_edges = EdgeBlock(
            agent_goal_feats, agent_goal_mask, id_agent, id_goal
        )

        # agent - obs connection
        agent_obs_edges = []
        n_hits = self._params["top_k_rays"] * self.num_agents
        if lidar_data is not None:
            id_obs = jnp.arange(self.num_agents + self.num_goals, self.num_agents + self.num_goals + n_hits)
            for i in range(self.num_agents):
                id_hits = jnp.arange(i * self._params["top_k_rays"], (i + 1) * self._params["top_k_rays"])
                lidar_feats = agent_pos[i, :] - lidar_data[id_hits, :]
                lidar_dist = jnp.linalg.norm(lidar_feats, axis=-1)
                active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
                agent_obs_mask = jnp.ones((1, self._params["top_k_rays"]))
                agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
                lidar_feats = jnp.concatenate(
                    [lidar_feats, jnp.zeros((lidar_feats.shape[0], self.edge_dim - lidar_feats.shape[1]))], axis=-1)
                agent_obs_edges.append(
                    EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
                )

        return [agent_agent_edges, agent_goal_edges] + agent_obs_edges

'''
Great question!  
In **constrained optimization** (e.g., with Lagrangian or penalty methods), you often see a **trade-off** between reward and safety:  
- As you enforce safety more strictly, the agent must avoid risky (but possibly high-reward) actions, so **reward decreases as safety increases**.

**But in your scatter plot, you see a directly proportional relationship:**  
- **Higher safety → Higher reward**

---

## Why does this happen in your case?

### 1. **Task Structure: Safety and Reward are Aligned**
- In your environment (e.g., LidarSpread), the reward is likely **higher when agents reach goals without collisions or unsafe events**.
- Unsafe behavior (e.g., collisions, leaving the area) is penalized in both the reward and the safety metric.
- So, **being safe is also the way to get a high reward**.

### 2. **Reward Function Design**
- Your reward function penalizes unsafe actions (e.g., distance from goal, collisions, excessive control effort).
- If an agent is unsafe, it gets a penalty, which **reduces the reward**.
- If an agent is always safe, it avoids penalties and can focus on maximizing the positive part of the reward.

### 3. **No Explicit Trade-off**
- In some environments, you can get a higher reward by taking more risk (e.g., going faster but risking a crash).
- In your environment, **unsafe actions are not a shortcut to higher reward**—they are just penalized.
- Thus, the optimal policy is both safe and high-reward.

### 4. **Constraint Not Active**
- In classic constrained RL, the constraint is only "active" if the agent wants to violate it to get more reward.
- If your agent can be both safe and high-reward (because the task is designed that way), the constraint is not binding, and you see a positive correlation.

---

## **Summary Table**

| Scenario                        | Reward vs Safety Relationship         |
|----------------------------------|--------------------------------------|
| Classic constraint trade-off     | Reward ↓ as Safety ↑ (negative corr) |
| Your environment (aligned)       | Reward ↑ as Safety ↑ (positive corr) |

---

## **In short:**
- **In your environment, being safe is the best way to get a high reward.**
- There is no trade-off, so you see a directly proportional (positive) relationship in your scatter plot.

If you want to see a trade-off, you would need a task where taking risks can sometimes increase reward (e.g., faster but riskier maneuvers), and safety constraints force the agent to slow down or avoid those risky actions.
'''

'''
The **action penalty** in the reward calculation serves several important purposes in reinforcement learning environments:

---

### 1. **Encourage Efficient Behavior**
- Penalizing large or unnecessary actions encourages agents to reach their goals using the **least effort** or energy.
- This mimics real-world scenarios where excessive movement, acceleration, or force is costly (e.g., fuel, battery, wear and tear).

### 2. **Promote Smooth and Stable Policies**
- Without an action penalty, agents might learn to make abrupt, jerky, or extreme actions to maximize reward quickly.
- The penalty encourages **smoother, more stable, and realistic trajectories**.

### 3. **Prevent Exploiting the Reward Function**
- Sometimes, agents can exploit the reward structure by taking extreme actions that maximize immediate reward but are unrealistic or unsafe.
- The penalty discourages such behavior.

### 4. **Regularization**
- It acts as a form of **regularization**, helping the policy generalize better and avoid overfitting to specific action patterns.

### 5. **Safety and Physical Constraints**
- In physical systems, large actions may be unsafe or infeasible.
- Penalizing them helps keep the agents behavior within safe and realistic bounds.

---

**In your code:**
```python
reward -= (jnp.linalg.norm(action, axis=1) ** 2) * 0.0001
```
- This penalizes the **squared magnitude** of each agents action, scaled by a small factor.
- It means the agent gets a higher reward for achieving its goal with **smaller, more efficient actions**.

---

**Summary:**  
The action penalty is needed to encourage efficient, smooth, and realistic agent behavior, prevent reward hacking, and promote safety and generalization.
'''


'''
The **action penalty** in your reward function:

```python
reward -= (jnp.linalg.norm(action, axis=1) ** 2).mean() * 0.0001
```

is needed for several reasons:

1. **Encourages Efficient Behavior:**  
   It motivates agents to reach their goals using minimal effort or energy, avoiding unnecessary or excessive movements.

2. **Promotes Smooth and Realistic Actions:**  
   Without this penalty, agents might take abrupt or extreme actions. The penalty encourages smoother, more stable, and realistic trajectories.

3. **Prevents Reward Hacking:**  
   It discourages agents from exploiting the reward function by taking large, unrealistic actions that maximize immediate reward.

4. **Acts as Regularization:**  
   It helps the policy generalize better and prevents overfitting to aggressive action patterns.

5. **Supports Safety and Physical Constraints:**  
   In real-world systems, large actions may be unsafe or infeasible. Penalizing them keeps agent behavior within safe and realistic bounds.

**Summary:**  
The action penalty ensures agents act efficiently, safely, and realistically, leading to better and more robust policies.
'''