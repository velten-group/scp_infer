"""
Modified DCDI model that includes addition 'global' parameters -> sigmoid curve parameters
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


class LearnableModel_global_param(BaseModel):
    def __init__(self, num_vars, num_layers, hid_dim, num_params,
                 nonlin="leaky-relu", intervention=False,
                 intervention_type="perfect",
                 intervention_knowledge="known", num_regimes=1, size_factors = None):

        super(LearnableModel_global_param, self).__init__(num_vars, num_layers, hid_dim, num_params,
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
        WARNING: This is really a joint distribution only if the DAGness constraint on the mask is satisfied.
                 Otherwise the joint does not integrate to one.
        :param x: (batch_size, num_vars)
        :param weights: list of tensor that are coherent with self.weights
        :param biases: list of tensor that are coherent with self.biases
        :param mask: tensor, shape=(batch_size, num_vars)
        :param regime: np.ndarray, shape=(batch_size,)
        :return: (batch_size, num_vars) log-likelihoods
        """
        #print("x: ", x)
        density_params = self.forward_given_params(
            x, weights, biases, mask, regime)
        # print("density_params: ",density_params[0][0])

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
            else:
                global_params = None

            conditional = self.get_distribution(density_param, global_params, size_fact)
            x_d = x[:, i].detach() if detach else x[:, i]
            log_probs.append(conditional.log_prob(x_d).unsqueeze(1))

        return torch.cat(log_probs, 1)

    def get_distribution(self, dp, gp, sf):
        raise NotImplementedError

    def transform_extra_params(self, extra_params):
        raise NotImplementedError

class LearnableModel_NonLinGauss_Sigmoid_Dropout(LearnableModel_global_param):
    def __init__(self, num_vars, num_layers, hid_dim, nonlin="leaky-relu",
                 intervention=False,
                 intervention_type="perfect",
                 intervention_knowledge="known",
                 num_regimes=1,
                 log_dropout = False,
                 sig_k = 1.0,
                 sig_b = 4.0,
                 lock_sigmoid = False):
        super(LearnableModel_NonLinGauss_Sigmoid_Dropout, self).__init__(num_vars, num_layers, hid_dim, 1, nonlin=nonlin,
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
                    torch.tensor(np.array([1,0])).type(torch.Tensor)))
        self.log_dropout = log_dropout
        self.sig_k = sig_k
        self.sig_b = sig_b
        self.lock_sigmoid = lock_sigmoid

    def get_distribution(self, dp, gp, sf=None):
        # pred_likelihood =  torch.distributions.normal.Normal(dp[0], dp[2])  # mean, std_dev
        # dropout_probs = torch.sigmoid(dp[1]).unsqueeze(0)                   # dropout_prob
        # dropout_likelihood = torch.distributions.Normal(0.0, 0.1).log_prob(x)
        # total_likelihood = dropout_probs*dropout_likelihood + (1.0 - dropout_probs)*pred_likelihood
        # unsupported operand type(s) for *: 'Tensor' and 'Normal'
        """Modified distribution to include dropout rate linked to the mean of the gaussian

        Args:
            dp (list): list containing: 
                dp[0] (tensor): (bs) tensor - means of the prediction
                dp[1] (tensor): 0-dim tensor - std_devs of the prediction
            gp (tensor): (num_vars) tensor - global parameters for the sigmoid function
            sf (list/array): (bs) size factors each sample

        Returns:
            torch.distributions.MixtureSameFamily: Mixture distribution of dropout and prediction
        """

        # Extract means, standard deviations, and sigmoid parameters
        bs = len(dp[0])
        means = dp[0]           # shape: bs 
        std_dev = dp[1]         # shape: 0-dim
        global_params = gp      # shape: num_vars

        if self.lock_sigmoid:
            sig_k = torch.tensor(self.sig_k).to(means.device)
            sig_b = torch.tensor(self.sig_b).to(means.device)
            sig_k.requires_grad = False
            sig_b.requires_grad = False
        else:
            sig_k = global_params[0]
            sig_b = global_params[1]
        #print("sig_k: ", sig_k)
        #print("sig_b: ", sig_b)

        if self.log_dropout:
            #make means positive
            means = torch.relu(means)
            dropout_prob = 1 - torch.sigmoid(sig_k*(torch.log(means+1) - torch.ones(bs)*sig_b)) # returns NaN value?
        else:
            dropout_prob = 1 - torch.sigmoid(sig_k*(means - torch.ones(bs)*sig_b))

        # Create component distributions for each variable
        component_means = torch.stack(
            [torch.zeros(bs), means], dim=1)  # shape: bs x 2
        component_stds = torch.stack(
            [torch.ones(bs) * 0.1, torch.ones(bs) * std_dev], dim=1)  # shape: bs x 2
        components = Normal(component_means, component_stds)

        # Define the mixture distribution
        #print("means shape: ", means.shape)
        #print("b shape: ", (torch.ones(bs)*sig_b).shape)
        #print("sig_k shape: ", sig_k.shape)
        #print("dropout_prob shape: ", dropout_prob.shape)

        mixture_probs = torch.stack(
            [torch.ones(bs) * dropout_prob, 1.0 - torch.ones(bs) * dropout_prob], dim=1)  # shape: bs x 2
        
        #print("mixture_probs shape: ", mixture_probs.shape)
        mixture_probs = torch.stack(
            [dropout_prob, 1.0 - dropout_prob], dim=1)  # shape: bs x 2
        #print("mixture_probs shape-v2: ", mixture_probs.shape)
        mixture_distribution = MixtureSameFamily(
            Categorical(mixture_probs), components)

        return mixture_distribution

    def transform_extra_params(self, extra_params):
        transformed_extra_params = []
        for extra_param in extra_params:
            transformed_extra_params.append(torch.exp(extra_param))
        return transformed_extra_params  # returns std_dev


class LearnableModel_NonLin_NegBin_sc(LearnableModel_global_param):
    def __init__(self, num_vars, num_layers, hid_dim, nonlin="leaky-relu",
                 intervention=False,
                 intervention_type="perfect",
                 intervention_knowledge="known",
                 num_regimes=1,
                 rescale_means = False,
                 ):
        super(LearnableModel_NonLin_NegBin_sc, self).__init__(num_vars, num_layers, hid_dim, 2, nonlin=nonlin,
                                                           intervention=intervention,
                                                           intervention_type=intervention_type,
                                                           intervention_knowledge=intervention_knowledge,
                                                           num_regimes=num_regimes)
        # extra parameters are log_std, p_dropout
        # each element in the list represents a variable, the size of the element is the number of extra_params per var
        # => list of 2 elements: log_std, p_dropout for each variable

        # extra parameters are log_std
        # extra_params = np.ones((self.num_vars,))
        # np.random.shuffle(extra_params)
        # each element in the list represents a variable, the size of the element is the number of extra_params per var
        self.extra_params = nn.ParameterList()
        # for extra_param in extra_params:
        #    self.extra_params.append(nn.Parameter(torch.tensor(
        #        np.log(extra_param).reshape(1)).type(torch.Tensor)))
        self.rescale_means = rescale_means

    def get_distribution(self, dp, gp, sf=None):
        """Modified distribution to negative binomial

        COUNTS HAVE TO BE INTEGERS!!!

        Args:
            dp (list): list containing: 
                dp[0] (tensor): (bs) tensor - means of the prediction
                dp[1] (tensor): (bs) tensor - overdispersion parameters
            gp (tensor): (num_vars) tensor - global parameters - not used
            sf (list/array): (bs) size factors each sample

        Returns:
            torch.distributions.negative_binomial.NegativeBinomial: Negative binomial distribution
        """

        # Extract means, overdispersion parameters
        # softplus to ensure positive values
        means = F.softplus(dp[0])
        alpha = F.softplus(dp[1]) + 1e-5
        size_factors = sf
        #print("means: ", means[0])
        #print("alpha: ", alpha)
        #print("size_factors: ", size_factors[0])

        if self.rescale_means:
            #size_factors = torch.tensor(size_factors).to(means.device)
            #size_factors.requires_grad = False
            means = means * size_factors



        # Compute the function parameters from mean and overdispersion:
        r = 1.0 / alpha                         # limit on number of failures
        p = 1.0 - 1.0 / (1.0 + alpha * means)   # one minus the probability of success

        # compute the function parameters from mean and overdispersion:
        # r = 1/overdispersion
        # p = r/(r+mean)
        # r = 1.0 / overdispersion + 1e-6
        # p = 1.0 / (1.0 + overdispersion * means + 1e-6)

        return torch.distributions.negative_binomial.NegativeBinomial(total_count=r, probs=p)

    def transform_extra_params(self, extra_params):
        transformed_extra_params = []
        for extra_param in extra_params:
            transformed_extra_params.append(torch.exp(extra_param))
        return transformed_extra_params  # returns std_dev
