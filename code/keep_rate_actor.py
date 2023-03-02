import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyGradient(nn.Module):
    def __init__(self,action_choice):
        super(PolicyGradient,self).__init__()

        with torch.no_grad():
            self.action_choice = action_choice  # [0.6,0.7,0.8,0.9,1.0]

        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(64, 32)
        self.fc_e = nn.Linear(32, len(self.action_choice))


    def forward(self,x):
        out_soft = []
        actions = []
        keep_rate = []
        size_related_keep_rate = []
        for alpha in x:
            if len(alpha) == 16:
                alpha = alpha
                mid = self.fc1(alpha)
                out = self.fc_e(mid)
                soft = F.softmax(out, dim=-1)
                out_soft.append(soft.unsqueeze(0))
                action = np.random.choice(np.arange(len(self.action_choice)), p=soft.cpu().detach().numpy())
                actions.append(action)
                keep_rate.append(self.action_choice[action])
                size_related_keep_rate.append(self.action_choice[action] * 0.5)
            elif len(alpha) == 32:
                alpha = alpha
                mid = self.fc2(alpha)
                out = self.fc_e(mid)
                soft = F.softmax(out, dim=-1)
                out_soft.append(soft.unsqueeze(0))
                action = np.random.choice(np.arange(len(self.action_choice)), p=soft.cpu().detach().numpy())
                actions.append(action)
                keep_rate.append(self.action_choice[action])
                size_related_keep_rate.append(self.action_choice[action] * 0.75)
            else:
                alpha = alpha
                mid = self.fc3(alpha)
                out = self.fc_e(mid)
                soft = F.softmax(out, dim=-1)
                out_soft.append(soft.unsqueeze(0))
                action = np.random.choice(np.arange(len(self.action_choice)), p=soft.cpu().detach().numpy())
                actions.append(action)
                keep_rate.append(self.action_choice[action])
                size_related_keep_rate.append(self.action_choice[action])

        return out_soft, actions, keep_rate, np.mean(size_related_keep_rate)

def learn(out_soft,actions,loss_v,sub_model_size,opt): # average of eposides
        opt.zero_grad()
        object =[]
        beta = 2
        actions = np.array(actions)
        for i in range(len(out_soft)):
            soft_prob = torch.cat(out_soft[i], dim=0)
            log_probs = torch.log(soft_prob)
            loss_e = torch.tensor((-(np.average(loss_v[i])-np.average(loss_v)))).cuda()*log_probs[np.arange(len(actions[i])), actions[i]]
            object.append(loss_e.unsqueeze(0))
        loss = - (torch.cat(object,dim=0).mean() + beta * sub_model_size)
        loss.backward()
        opt.step()
        return loss, sub_model_size