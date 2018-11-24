import torch
import torch.nn as nn

class CriticNet(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(CriticNet, self).__init__()

        obs_dim = dim_observation
        act_dim = dim_action

        self.FC1 = nn.Linear(obs_dim, 1024)
        self.FC2 = nn.Linear(1024+act_dim, 512)
        self.FC3 = nn.Linear(512, 300)
        self.FC4 = nn.Linear(300, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        result = torch.relu(self.FC1(obs))
        combined = torch.cat([result, acts], 1)
        out = torch.relu(self.FC2(combined))
        return self.FC4(torch.relu(self.FC3(out)))

class ActorNet(nn.Module):  
    def __init__(self, dim_observation, dim_action):
        super(ActorNet, self).__init__()

        self.FC1 = nn.Linear(dim_observation, 500)
        self.FC2 = nn.Linear(500, 128)
        self.FC3 = nn.Linear(128, dim_action)

    # action output between -1 and 1
    def forward(self, obs):
        result = torch.relu(self.FC1(obs))
        result = torch.relu(self.FC2(result))
        result = torch.tanh(self.FC3(result))
        return result
