import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_normal

"""GloVe in PyTorch.

Reference:
[1] Jeffrey Pennington, Richard Socher, Christopher D. Manning
    GloVe: Global Vectors for Word Representation. arXiv:1512.03385
Paper Ref: https://nlp.stanford.edu/pubs/glove.pdf
Code  Ref: https://nlpython.com/implementing-glove-model-with-pytorch/  
      Ref: https://github.com/kefirski/pytorch_GloVe
"""

class GloveModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(GloveModel, self).__init__()
        self.wi = nn.Embedding(num_embeddings, embedding_dim)
        self.wj = nn.Embedding(num_embeddings, embedding_dim)
        self.bi = nn.Embedding(num_embeddings, 1)
        self.bj = nn.Embedding(num_embeddings, 1)

        self.wi.weight.data.uniform_(-1, 1)
        self.wj.weight.data.uniform_(-1, 1)
        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()

    def forward(self, i_indices, j_indices):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        b_i = self.bi(i_indices).squeeze()
        b_j = self.bj(j_indices).squeeze()

        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j

        return x

class wmse_loss(nn.Module):
    def __init__(self, x_ij, x_max, alpha):
        super(wmse_loss, self).__init__()
        self.x_ij = x_ij
        self.x_max = x_max
        self.alpha = alpha

    def weight_func(self, x, x_max, alpha):
        wx = (x/x_max)**alpha
        wx = torch.min(wx, torch.ones_like(wx))
        return wx.cuda()

    def forward(self, inputs, targets):
        weights_x = self.weight_func(self.x_ij, self.x_max, self.alpha)
        loss = weights_x * F.mse_loss(inputs, targets, reduction='none')
        return torch.mean(loss).cuda()
        


# class Glove(nn.Module):
#     def __init__(self, co_oc, emb_dimension, x_max=100, alpha=0.75):
#         """
#         :Although we write 'GloVe' with Capital Letter, for function clarity, I will write GloVe with Glove
#         :param co_oc: Co-occurrence ndarray with shape of [num_classes, num_classes]
#         :param embed_size: embedding size
#         :param x_max: An int representing cutoff of the weighting function
#         :param alpha: An float parameter of the weighting function
#         """

#         super(Glove, self).__init__()

#         self.emb_dimension = emb_dimension
#         self.x_max = x_max
#         self.alpha = alpha

#         ''' co_oc Matrix is shifted in order to prevent having log(0) '''
#         self.co_oc = co_oc + 1.0

#         [self.num_classes, _] = self.co_oc.shape

#         self.in_embed = nn.Embedding(self.num_classes)
#         self.in_embed.weight = xavier_normal(self.in_embed.weight)

#         self.in_bias = nn.Embedding(self.num_classes, 1)
#         self.in_bias.weight = xavier_normal(self.in_bias.weight)

#         self.out_embed = nn.Embedding(self.num_classes, self.embed_size)
#         self.out_embed.weight = xavier_normal(self.out_embed.weight)

#         self.out_bias = nn.Embedding(self.num_classes, 1)
#         self.out_bias.weight = xavier_normal(self.out_bias.weight)

#     def forward(self, input, output):
#         """
#         :Basically, Embedding Function doesn't not return embedding in forward function, but the loss itself, since training of embedding is done in the loss calculation process.

#         :param input: An array with shape of [batch_size] of int type
#         :param output: An array with shape of [batch_size] of int type
#         :return: loss estimation for Global Vectors word representations
#                  defined in nlp.stanford.edu/pubs/glove.pdf
#         """
#         batch_size = len(input)

#         co_occurences = np.array([self.co_oc[input[i], output[i]] for i in range(batch_size)])
#         weights = np.array([self._weight(var) for var in co_occurences])

#         input = Variable(torch.from_numpy(input))
#         output = Variable(torch.from_numpy(output))

#         input_embed = self.in_embed(input)
#         input_bias = self.in_bias(input)
#         output_embed = self.out_embed(output)
#         output_bias = self.out_bias(output)

#         return (torch.pow(
#             ((input_embed * output_embed).sum(1) + input_bias + output_bias).squeeze(1) - torch.log(co_occurences), 2
#         ) * weights).sum()


#     def _weight(self, x):
#         return 1 if x > self.x_max else (x / self.x_max) ** self.alpha

#     def embeddings(self):
#         return self.in_embed.weight.data.cpu().numpy() + self.out_embed.weight.data.cpu().numpy()