import copy
import torch
import torch.optim as optim
from collections import OrderedDict
from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, CLIENT_PARAMS_KEY, CLIENT_GRAD_KEY, CLIENT_TASK_KEY

from .fedavg_server import Server


class AdaBestServer(Server):
    """
        AdaBestServer: Server that simulates the algorithm of AdaBest as defined in 
        "AdaBest: Minimizing Client Drift in Federated Learning via Adaptive Bias Estimation" 
        https://arxiv.org/abs/2204.13170
        
            Parameters:
            - beta: Hyperparameter for the trade-off of information maintained from the previous
                    estimate of full gradient and the estimation that a new round provides
    """
    def __init__(self, client_model, momentum=0, beta=0, opt_ckpt=None):
        super().__init__(client_model)
        print("beta", beta)
        
        self.server_model = client_model
        self.server_momentum = momentum
        self.beta = beta
        if opt_ckpt is not None:
            self.load_optimizer_checkpoint(opt_ckpt)
        self.round = 0 # Number of rounds (to be sent to the client)
        # Server oracles
        self.historical = { name: 0.0 for name, _ in self.server_model.named_parameters() }
        # Average of client models of previous round
        self.prev_avg_model = { name: 0.0 for name, _ in self.server_model.named_parameters() }

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None, analysis=False):
        # Updating the number of rounds
        self.round += 1
        
        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   CLIENT_PARAMS_KEY: 0,
                   CLIENT_GRAD_KEY: 0,
                   CLIENT_TASK_KEY: {}
                   } for c in clients}

        for c in clients:
            # θ_i^{t,0} ← θ^{t-1}
            c.model.load_state_dict(self.model)
            num_samples, update = c.train(num_epochs, batch_size, minibatch, self.round)
            sys_metrics = self._update_sys_metrics(c, sys_metrics)
            self.updates.append((num_samples, copy.deepcopy(update)))
        
        # sys_metrics = super(AdaBestServer, self).train_model(num_epochs, batch_size, minibatch, clients, analysis)
        #self._save_updates_as_pseudogradients()
        return sys_metrics

    def update_model(self):
        """
        AdaBest algorithm on the server update for the current round.
        Saves the new central model in self.server_model and its state dictionary in self.model
        """
        self.server_model.load_state_dict(self.model)
        # Aggregate received models
        curr_average_model = self._average_updates()
        # Update global model according to chosen optimizer
        self._update_global_model_gradient(curr_average_model)
        
        self.model = copy.deepcopy(self.server_model.state_dict())
        # self.total_grad = self._get_model_total_grad()
        self.updates = []
        return

    def save_model(self, round, ckpt_path, swa_n=None):
        """Saves the servers model and optimizer on checkpoints/dataset/model.ckpt."""
        # Save servers model
        save_info = {'model_state_dict': self.model,
                     'round': round}
        if self.swa_model is not None:
            save_info['swa_model'] = self.swa_model.state_dict()
        if swa_n is not None:
            save_info['swa_n'] = swa_n
        torch.save(save_info, ckpt_path)
        return ckpt_path

    def _update_global_model_gradient(self, average_model):
        """
        Updates the server model using the average of the clients models
        Paramters:
            average_model: Average model, i.e. weighted average of the trained clients' deltas.
        """
        new_model = {}
        for name, parameter in average_model.items():
            # h^t ← β(θ¯^{t−1} − θ¯^t) # Update oracle estimates
            self.historical[name] = self.beta * (self.prev_avg_model[name] - parameter)
            # θ^t ← θ¯^t - h^t # Update cloud (server) model
            new_model[name] = parameter - self.historical[name]
            # θ¯^{t+1} = θ¯^t # Setting the previous avg model as the current one for the next round
            self.prev_avg_model[name] = parameter
        self.server_model.load_state_dict(new_model, strict=False)

        bn_layers = OrderedDict(
            {k: v for k, v in average_model.items() if "running" in k or "num_batches_tracked" in k})
        self.server_model.load_state_dict(bn_layers, strict=False)

    def _get_model_total_grad(self):
        total_norm = 0
        for name, p in self.server_model.named_parameters():
            if p.requires_grad:
                try:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                except Exception:
                    # this param had no grad
                    pass
        total_grad = total_norm ** 0.5
        # print("total grad norm:", total_grad)
        return total_grad
    
    def load_optimizer_checkpoint(self, optimizer_ckpt):
        "Load optimizer state from checkpoint"
        self.server_opt.load_state_dict(optimizer_ckpt)