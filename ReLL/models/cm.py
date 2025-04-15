import collections
import math

import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, source_classes, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.source_classes = source_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.knn = 6
        self.lambda0 = 0.55
        self.tau = 0.05
        self.delta = 3.5

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets, epoch=0, isIndex=False, is_source=True):
        if isinstance(targets, torch.Tensor) is False:
            targets = torch.cat(targets)
            targets = targets - 1
            inds = (targets>=0)
            targets = targets[inds]
            inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)

        inputs = F.normalize(inputs, dim=1).cuda()
        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.featPures, self.momentum)
            outputs /= self.temp
            loss = F.cross_entropy(outputs, targets)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)
            outputs /= self.temp
            
            if isIndex:
                if epoch<0:
                    # loss = self.smooth_loss(outputs, targets)
                    loss = F.cross_entropy(outputs, targets)
                else:
                    loss = self.smooth_loss(outputs, targets, epoch)
            else:
                loss = F.cross_entropy(outputs, targets)
                
        if torch.isnan(loss).int().sum() > 0:
            print("get nan loss, it  will be converted to zero")
            device = loss.device
            loss = torch.tensor(0,device=device,dtype=torch.float32)
            
        return loss,outputs

    def smooth_loss(self, inputs, targets, epoch):
        targets = self.adaptive_selection(inputs.detach().clone(), targets.detach().clone(), epoch)
        outputs = F.log_softmax(inputs, dim=1)
        loss = - (targets * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss
        
    def adaptive_selection(self, inputs, targets, t):
        T = 50
        if T > t:
            t = t+1

        targets_onehot = (inputs > self.lambda0/self.tau).float()
        ks = (targets_onehot.sum(dim=1)).float()
        ks1 = ks.cpu()
        ks_mask = (ks > 1).float()
        ks = ks * ks_mask + (1 - ks_mask) * 2

        ratio = torch.log(torch.Tensor([1 + t * (math.exp(1) - 1) / T])).to(self.device)
        ks = ks * (1 + ratio)
        ks = self.delta / ks

        ks = (ks * ks_mask).view(-1,1)
        targets_onehot = targets_onehot * ks

        targets = torch.unsqueeze(targets, 1)
        targets_onehot.scatter_(1, targets, float(1))
         
        return targets_onehot
