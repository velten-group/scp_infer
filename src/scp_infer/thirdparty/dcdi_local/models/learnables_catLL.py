"""
Modified DCDI model to include dropout probabilities in the likelihood function - categorical likelihood
"""
from .base_model import BaseModel
from torch.distributions import Normal, Categorical, MixtureSameFamily
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import sys
import math
sys.path.insert(0, '../')


class LearnableModel_catLL(BaseModel):
    def __init__(self, num_vars, num_layers, hid_dim, num_params,
                 nonlin="leaky-relu", intervention=False,
                 intervention_type="perfect",
                 intervention_knowledge="known", num_regimes=1, size_factors = None):

        super(LearnableModel_catLL, self).__init__(num_vars, num_layers, hid_dim, num_params,
                                             nonlin=nonlin,
                                             intervention=intervention,
                                             intervention_type=intervention_type,
                                             intervention_knowledge=intervention_knowledge,
                                             num_regimes=num_regimes)
        self.reset_params()
        self.size_factors = size_factors

    def compute_log_likelihood(self, x, size_fact, weights, biases, extra_params,
                               detach=False, mask=None, regime=None):
        """
        Return log-likelihood of the model for each example.
            if x == 0, then the log-likelihood is a p_drop for each variable
            if x != 0, then the log-likelihood is given by self.get_distribution()
        WARNING: This is really a joint distribution only if the DAGness constraint on the mask is satisfied.
                 Otherwise the joint does not integrate to one.
        :param x: (batch_size, num_vars)
        :param weights: list of tensor that are coherent with self.weights
        :param biases: list of tensor that are coherent with self.biases
        :param mask: tensor, shape=(batch_size, num_vars)
        :param regime: np.ndarray, shape=(batch_size,)
        :return: (batch_size, num_vars) log-likelihoods
        """
        density_params = self.forward_given_params(
            x, weights, biases, mask, regime)

        if len(extra_params) != 0:
            extra_params = self.transform_extra_params(self.extra_params)
        log_probs = []

        for i in range(self.num_vars):
            density_param = list(torch.unbind(density_params[i], 1))
            if len(extra_params) != 0:
                density_param.extend(list(torch.unbind(extra_params[i], 0)))
                # get the global params, last entry of all i extra_params
                global_params = torch.stack([extra_params[j][-1] for j in range(self.num_vars)])
                #print("fixed_params: ", fixed_params)

            p_drop, conditional = self.get_distribution(density_param, global_params)
            x_d = x[:, i].detach() if detach else x[:, i]
            #print("x_d shape: ", x_d.shape)
            #print("p_drop shape: ", p_drop.shape)
            
            # if x == 0, then the log-likelihood is p_drop for each variable
            zero_log_prob = torch.log(p_drop).unsqueeze(1)
            # if x != 0, then the log-likelihood is given by self.get_distribution() times (1 - p_drop)
            conditional_logprob = conditional.log_prob(x_d).unsqueeze(1)
            #print("conditional_logprob shape: ", conditional_logprob.shape)
            nonzero_log_prob = torch.log(1 - p_drop).unsqueeze(1)
            #print("nonzero_log_prob shape: ", nonzero_log_prob.shape)
            #nonzero_log_prob = conditional.log_prob(x_d).unsqueeze(1) + torch.log(1 - p_drop).unsqueeze(1)
            nonzero_log_prob = nonzero_log_prob + conditional_logprob
            #print("nonzero_log_prob shape: ", nonzero_log_prob.shape)
            # the +torch.log(1 - p_drop) causes NaN values in the means of density_param???????????? WHYYYYYYYY????

            log_probs.append(torch.where(x_d == 0, zero_log_prob, nonzero_log_prob))
            #log_probs.append(conditional.log_prob(x_d).unsqueeze(1))

        return torch.cat(log_probs, 1)

    def get_distribution(self, dp, gp):
        raise NotImplementedError

    def transform_extra_params(self, extra_params):
        raise NotImplementedError

class LearnableModel_NonLinGauss_DropOut_cat(LearnableModel_catLL):
    def __init__(self, num_vars, num_layers, hid_dim, nonlin="leaky-relu",
                 intervention=False,
                 intervention_type="perfect",
                 intervention_knowledge="known",
                 num_regimes=1,
                 log_dropout = False):
        super(LearnableModel_NonLinGauss_DropOut_cat, self).__init__(num_vars, num_layers, hid_dim, 2, nonlin=nonlin,
                                                                 intervention=intervention,
                                                                 intervention_type=intervention_type,
                                                                 intervention_knowledge=intervention_knowledge,
                                                                 num_regimes=num_regimes)
        # extra parameters are log_std, sigmoid_list (k,b,0,...,0 for all variables)
        extra_params = np.ones((self.num_vars, 2))
        np.random.shuffle(extra_params)
        # each element in the list represents a variable, the size of the element is the number of extra_params per var
        # => list of 2 elements: log_std, [k,b,
        self.extra_params = nn.ParameterList()
        for i,extra_param in enumerate(extra_params):
            # self.extra_params.append(nn.Parameter(torch.tensor(np.log(extra_param).reshape(1)).type(torch.Tensor)))
            if i==0:
                self.extra_params.append(nn.Parameter(
                    torch.tensor(np.array([1])).type(torch.Tensor)))
            else:
                self.extra_params.append(nn.Parameter(
                    torch.tensor(np.array([1])).type(torch.Tensor)))
        self.log_dropout = log_dropout

    def get_distribution(self, dp, gp):
        """Modified distribution to include dropout rate linked to the mean of the gaussian

        Args:
            dp (list): list containing: 
                dp[0] (tensor): (bs) tensor - means of the prediction
                dp[1] (tensor): 0-dim tensor - std_devs of the prediction

        Returns:
            tuple:
                torch.distributions.MixtureSameFamily: Mixture distribution of dropout and prediction
                torch.Tensor: dropout probabilities
        """

        # Extract means, standard deviations, and sigmoid parameters
        bs = len(dp[0])
        means = dp[0]           # shape: bs 
        dropout_probs = torch.sigmoid(dp[1])  # shape: bs
        std_devs = dp[2]         # shape: 0-dim
        global_params = gp      # shape: num_vars
        

        return dropout_probs, Normal(means, std_devs)

    def transform_extra_params(self, extra_params):
        transformed_extra_params = []
        for extra_param in extra_params:
            transformed_extra_params.append(torch.exp(extra_param))
        return transformed_extra_params  # returns std_dev

class LearnableModel_NonLinGauss_DropOut_global_cat(LearnableModel_catLL):
    def __init__(self, num_vars, num_layers, hid_dim, nonlin="leaky-relu",
                 intervention=False,
                 intervention_type="perfect",
                 intervention_knowledge="known",
                 num_regimes=1,
                 log_dropout = False):
        super(LearnableModel_NonLinGauss_DropOut_global_cat, self).__init__(num_vars, num_layers, hid_dim, 1, nonlin=nonlin,
                                                                 intervention=intervention,
                                                                 intervention_type=intervention_type,
                                                                 intervention_knowledge=intervention_knowledge,
                                                                 num_regimes=num_regimes)
        # extra parameters are log_std, p_drop, global_params (k,b,0,...,0 for all variables)
        extra_params = np.ones((self.num_vars, 2))
        np.random.shuffle(extra_params)
        # each element in the list represents a variable, the size of the element is the number of extra_params per var
        self.extra_params = nn.ParameterList()
        for i,extra_param in enumerate(extra_params):
            # self.extra_params.append(nn.Parameter(torch.tensor(np.log(extra_param).reshape(1)).type(torch.Tensor)))
            if i==0:
                self.extra_params.append(nn.Parameter(
                    torch.tensor(np.array([1,0.5,1])).type(torch.Tensor)))
            else:
                self.extra_params.append(nn.Parameter(
                    torch.tensor(np.array([1,0.5,1])).type(torch.Tensor)))
        self.log_dropout = log_dropout

    def get_distribution(self, dp, gp):
        """Modified distribution to include dropout rate linked to the mean of the gaussian

        Args:
            dp (list): list containing: 
                dp[0] (tensor): (bs) tensor - means of the prediction
                dp[1] (tensor): 0-dim tensor - std_devs of the prediction

        Returns:
            tuple:
                torch.distributions.MixtureSameFamily: Mixture distribution of dropout and prediction
                torch.Tensor: dropout probabilities
        """

        # Extract means, standard deviations, and sigmoid parameters
        bs = len(dp[0])
        means = dp[0]           # shape: bs 
        std_dev = dp[1]         # shape: 0-dim
        dropout_probs = torch.sigmoid(dp[2])  # shape: bs
        global_params = gp      # shape: num_vars
        

        return dropout_probs, Normal(means, torch.ones(bs) * std_dev)

    def transform_extra_params(self, extra_params):
        transformed_extra_params = []
        for extra_param in extra_params:
            transformed_extra_params.append(torch.exp(extra_param))
        return transformed_extra_params  # returns std_dev

class LearnableModel_NonLinGauss_Sigmoid_Dropout_cat(LearnableModel_catLL):
    def __init__(self, num_vars, num_layers, hid_dim, nonlin="leaky-relu",
                 intervention=False,
                 intervention_type="perfect",
                 intervention_knowledge="known",
                 num_regimes=1,
                 log_dropout = False,
                 sig_k = 1.0,
                 sig_b = 4.0,
                 lock_sigmoid = False
                 ):
        super(LearnableModel_NonLinGauss_Sigmoid_Dropout_cat, self).__init__(num_vars, num_layers, hid_dim, 1, nonlin=nonlin,
                                                                 intervention=intervention,
                                                                 intervention_type=intervention_type,
                                                                 intervention_knowledge=intervention_knowledge,
                                                                 num_regimes=num_regimes)
        # extra parameters are log_std, sigmoid_list (k,b,0,...,0 for all variables)
        extra_params = np.ones((self.num_vars, 2))
        np.random.shuffle(extra_params)
        # each element in the list represents a variable, the size of the element is the number of extra_params per var
        # => list of 2 elements: log_std, [k,b,
        self.extra_params = nn.ParameterList()
        for i,extra_param in enumerate(extra_params):
            # self.extra_params.append(nn.Parameter(torch.tensor(np.log(extra_param).reshape(1)).type(torch.Tensor)))
            if i==0:
                self.extra_params.append(nn.Parameter(
                    torch.tensor(np.array([1,4])).type(torch.Tensor)))
                # locking parameter: self.extra_params[-1][1].requires_grad = False
            elif i==1:
                self.extra_params.append(nn.Parameter(
                    torch.tensor(np.array([1,1])).type(torch.Tensor)))
                # locking parameter: self.extra_params[-1][1].requires_grad = False
            else:
                self.extra_params.append(nn.Parameter(
                    torch.tensor(np.array([1,1])).type(torch.Tensor)))
        self.log_dropout = log_dropout
        self.sig_k = sig_k
        self.sig_b = sig_b
        self.lock_sigmoid = lock_sigmoid

    def get_distribution(self, dp, gp):
        """Modified distribution to include dropout rate linked to the mean of the gaussian

        Args:
            dp (list): list containing: 
                dp[0] (tensor): (bs) tensor - means of the prediction
                dp[1] (tensor): 0-dim tensor - std_devs of the prediction

        Returns:
            tuple:
                torch.distributions.MixtureSameFamily: Mixture distribution of dropout and prediction
                torch.Tensor: dropout probabilities
        """

        # Extract means, standard deviations, and sigmoid parameters
        bs = len(dp[0])
        means = dp[0]           # shape: bs 
        std_dev = dp[1]         # shape: 0-dim
        global_params = gp      # shape: num_vars

        print("means shape: ", means.shape)
        print('means device: ', means.device)
        print('means: ', means[0:10])
        print("std_dev shape: ", std_dev.shape)
        print('std_dev device: ', std_dev.device)

        if self.lock_sigmoid:
            sig_k = torch.tensor(self.sig_k).to(means.device)
            sig_b = torch.tensor(self.sig_b).to(means.device)
            sig_k.requires_grad = False
            sig_b.requires_grad = False
        else:
            sig_k = global_params[0]
            sig_b = global_params[1]
        print("sig_k: ", sig_k)
        print("sig_b: ", sig_b)

        if self.log_dropout:
            #make means positive
            means = torch.relu(means)
            dropout_prob = 1 - torch.sigmoid(sig_k*(torch.log(means+1) - torch.ones(bs)*sig_b)) # returns NaN value?
        else:
            dropout_prob = 1 - torch.sigmoid(sig_k*(means - torch.ones(bs)*sig_b))  # shape: bs



        print("dropout_prob shape: ", dropout_prob.shape)
        print('dropout_prob device: ', dropout_prob.device)
        return dropout_prob, Normal(means, std_dev)

    def transform_extra_params(self, extra_params):
        transformed_extra_params = []
        for extra_param in extra_params:
            transformed_extra_params.append(torch.exp(extra_param))
        return transformed_extra_params  # returns std_dev