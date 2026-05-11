import torch
import torch.nn as nn
from torch_wm.core.registry import ModuleRegistry
from torch_wm.modules import MultiLayerPerceptron

@ModuleRegistry.register('expl_disag')
class Disag(nn.Module):
    """
    Disagreement-based exploration (ensemble of predictors).
    
    The ensemble predicts the next latent state (deter).
    The intrinsic reward is the variance of the ensemble predictions.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_models = config.get("disag_models", 5)
        self.hidden_size = config.get("hidden_size", 256)
        self.deter_size = config.get("deter_size", 1024)
        
        # Ensemble of MLPs
        self.models = nn.ModuleList([
            MultiLayerPerceptron(
                dim_input=self.hidden_size + config.get("num_actions", 63),
                dim_layers=[self.hidden_size, self.hidden_size, self.deter_size],
                act_fun=nn.SiLU
            ) for i in range(self.num_models)
        ])
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.get("expl_lr", 1e-4))

    def forward(self, traj_feat, actions):
        """
        Compute intrinsic reward based on ensemble disagreement.
        
        Args:
            traj_feat: (B, L, D) features
            actions: (B, L, A) actions
            
        Returns:
            Disagreement reward: (B, L)
        """
        # Concatenate feature and action
        inputs = torch.cat([traj_feat, actions], dim=-1)
        
        # Get predictions from all models
        preds = []
        for model in self.models:
            preds.append(model(inputs))
            
        preds = torch.stack(preds, dim=0) # (num_models, B, L, D)
        
        # Variance across models
        disagreement = preds.var(dim=0).mean(dim=-1)
        return disagreement

    def train_step(self, traj_feat, actions, targets):
        """Train the ensemble to predict the next state."""
        inputs = torch.cat([traj_feat, actions], dim=-1)
        
        total_loss = 0
        for model in self.models:
            pred = model(inputs)
            loss = nn.functional.mse_loss(pred, targets.detach())
            total_loss += loss
            
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {"loss_disag": total_loss.item()}
