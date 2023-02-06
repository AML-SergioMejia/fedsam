from copy import deepcopy
import torch

from .client import Client
from .minimizers import AdaBest


class AdaBestClient(Client):
    """
    AdaBestClient: Client that simulates the algorithm of AdaBest as defined in 
    "AdaBest: Minimizing Client Drift in Federated Learning via Adaptive Bias Estimation" 
    https://arxiv.org/abs/2204.13170
    
        Parameters:
        - mu: Regularization factor hyperparameter
    """
    def __init__(self, seed, client_id, lr, weight_decay, batch_size, momentum, train_data, eval_data, model, mu, device=None,
                num_workers=0, run=None, mixup=False, mixup_alpha=1.0):
        super().__init__(seed, client_id, lr, weight_decay, batch_size, momentum, train_data, eval_data, model, device,
                            num_workers, run, mixup, mixup_alpha)
        self.mu = mu
        # Pseudo-gradients that work as local oracles (h_t)
        self.historical = { name: 0.0 for name, _ in self.model.named_parameters() }
        # Last round the client participated in training (t'_i)
        self.last_round = 0
        # Last received server model (\theta^{t-1})
        self.server_model = deepcopy(self.model.state_dict())


    def train(self, num_epochs=1, batch_size=10, minibatch=None, round=1):
        # Save the received server model
        self.server_model = deepcopy(self.model.state_dict())
        num_train_samples, update = super(AdaBestClient, self).train(num_epochs, batch_size, minibatch)

        # After trainig update local gradient estimates
        new_historical = { }
        for name, value in self.model.named_parameters():
            # g^t_i ← θ^{t−1} − θ_i^{t,K}
            local_gradient = self.server_model[name] - value
            # h^t_i ← 1/(t−t′_i) * h_i^{t′}_i + µ * g^t_i
            new_historical[name] = 1 / (round - self.last_round) * self.historical[name] + self.mu * local_gradient
        
        self.last_round = round
        self.historical = new_historical

        return num_train_samples, update

    def run_epoch(self, optimizer, criterion):

        minimizer = AdaBest(optimizer, self.model, self.historical)
        running_loss = 0.0
        i = 0

        for inputs, targets in self.trainloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Step
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            minimizer.step()

            with torch.no_grad():
                running_loss += loss.item()

            i += 1
        if i == 0:
            print("Not running epoch", self.id)
            return 0
        return running_loss / i