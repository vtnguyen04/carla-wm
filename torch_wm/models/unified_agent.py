import torch
import torch.nn as nn
from torch_wm.core.registry import ModuleRegistry
from torch_wm.models.wm_agent import WMAgent

@ModuleRegistry.register('unified_agent')
class UnifiedAgent(WMAgent):
    """
    Unified Framework for World Models in CARLA.
    
    Supports multiple strategies via pluggable modules:
    - TSSM (Transformer State Space Model)
    - V-JEPA (Joint-Embedding Predictive Architecture)
    - TD-MPC (Model Predictive Control with Value Expansion)
    
    The strategy is determined by the 'dynamics_type' and 'agent_strategy' 
    parameters in the YAML configuration.
    """
    
    def __init__(self, env_name, override_config={}, **kwargs):
        # Determine strategy before super().__init__ if needed
        self.strategy_name = override_config.get("agent_strategy", "actor_critic")
        self.use_ema = override_config.get("use_ema", False)
        self.ema_decay = override_config.get("ema_decay", 0.98)
        super().__init__(env_name, override_config, **kwargs)
        
        # Load Strategy-specific components
        self._setup_strategy()
        
        # Setup EMA networks if requested
        if self.use_ema:
            import copy
            self.ema_encoder = copy.deepcopy(self.encoder_network)
            self.ema_dynamics = copy.deepcopy(self.dynamics_model)
            # Freeze EMA parameters
            for p in self.ema_encoder.parameters(): p.requires_grad = False
            for p in self.ema_dynamics.parameters(): p.requires_grad = False

    def _setup_strategy(self):
        """Initialize components based on the selected strategy."""
        if self.strategy_name == "mpc":
            # Setup for TD-MPC style planning
            from torch_wm.modules.planners.cem_planner import CEMPlanner
            # Ensure action_dim is set for the planner
            if "action_dim" not in self.config and "num_actions" in self.config:
                self.config["action_dim"] = self.config["num_actions"]
            self.planner = CEMPlanner(self.config)
        elif self.strategy_name == "vjepa":
            # V-JEPA style joint-embedding (no pixel reconstruction)
            # This is primarily handled by the LossManager (disabling reconstruction loss)
            pass
        
    def act(self, obs, is_first, sample=True):
        """Override act to support MPC planning if enabled."""
        if self.strategy_name == "mpc":
            # Planning-based action selection (TD-MPC)
            return self._act_mpc(obs, is_first)
        return super().act(obs, is_first, sample)

    def update_target_networks(self):
        """Update standard target networks and custom EMA networks."""
        super().update_target_networks()
        
        if self.use_ema:
            # Update EMA Encoder
            for p_t, p_n in zip(self.ema_encoder.parameters(), self.encoder_network.parameters()):
                p_t.mul_(self.ema_decay).add_((1 - self.ema_decay) * p_n.detach())
                
            # Update EMA Dynamics
            for p_t, p_n in zip(self.ema_dynamics.parameters(), self.dynamics_model.parameters()):
                p_t.mul_(self.ema_decay).add_((1 - self.ema_decay) * p_n.detach())

    def _act_mpc(self, obs, is_first):
        """MPC-based action selection using CEM."""
        # Ensure we have what's needed for planning
        if not hasattr(self, "planner"):
            # Fallback to policy if planner not ready
            return super().act(obs, is_first, sample=False)
            
        with torch.no_grad():
            # 1. Prepare initial state
            state = self._current_state if self._current_state is not None else self.dynamics_model.initial(1, 1, device=self.device)
            current_feat = self.dynamics_model.get_feat(state)

            # 2. Check if we have a goal for distance-based planning
            goal_latent = None
            if "goal" in obs:
                goal_embed = self.encoder_network({"camera": obs["goal"]})
                goal_latent = self.dynamics_model.get_feat(goal_embed)

            # 3. Define rollout logic
            def wm_rollout(latent, actions):
                # actions: (H, A) - planning for a single batch item B=1
                # latent: (1, D)
                curr_state = state.copy()
                total_reward = 0
                
                for t in range(actions.shape[1]):
                    # Step dynamics: returns single state dict
                    curr_state = self.dynamics_model.img_step(curr_state, actions[:, t:t+1])
                    
                    if goal_latent is None:
                        # If no goal, we return cumulative reward
                        feat = self.dynamics_model.get_feat(curr_state)
                        reward = self.reward_network(feat).mean()
                        total_reward += reward
                
                if goal_latent is not None:
                    # For goal reaching, return the final feature
                    return self.dynamics_model.get_feat(curr_state)
                else:
                    # For reward maximization
                    return total_reward

            # 4. Plan action sequence
            action_seq = self.planner.plan(
                current_feat,
                wm_rollout,
                goal_latent=goal_latent
            )
            
            # 5. Select first action
            action = action_seq[:, 0:1]
            self._prev_action = action
            
        return action, state
