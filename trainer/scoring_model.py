import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import math
from scipy.special import comb
import pdb

training_config = {
	'lr' : 1e-4,
	'bsz' : 16,
	'nepochs' : 5
}

# Best config
# training_config = {
# 	'lr' : 1e-4,
# 	'bsz' : 16,
# 	'nepochs' : 5
# }


def create_tensor(shape, zero_out=1.0, requires_grad=True, is_cuda=True):
	inits = torch.zeros(*shape) #.uniform_(-1/shape[0], 1/shape[0]) * zero_out
	# Create the weights
	weights = inits.float().cuda() if is_cuda else inits.float()
	if requires_grad:
		weights.requires_grad = True
	return weights

def get_score_model(num_players, model_type):
	if model_type == 'logistic_regression':
		return LogisticReg(num_players=num_players)

class LogisticReg(nn.Module):
	def __init__(self, num_players=10):
		super(LogisticReg, self).__init__()
		self.num_players = num_players
		
		self.score_tensor = create_tensor((num_players, 1))
		self.intercept = create_tensor((1, 1),zero_out=0)

		self.optim = Adam([self.score_tensor, self.intercept], lr=training_config['lr'])
		self.loss_fn = nn.MSELoss() #reduction='none')
		self.reg_fn =  nn.L1Loss() 
		self.l1_weight = 10

	def update_with_info(self, run_info):
		mask_tensors, scores = run_info
		mask_tensors = torch.stack(mask_tensors).to(self.score_tensor.device)
		scores = torch.tensor(scores).to(self.score_tensor.device)
		num_masks = len(mask_tensors)
		for epoch_ in range(training_config['nepochs']):
			# generate a permutation
			perm = np.random.permutation(num_masks)
			running_loss_avg = 0.0
			n_batches = math.ceil(num_masks / training_config['bsz'])

# 			print('Sum : {}, Inter {}'.format(self.score_tensor.sum().item(), self.intercept.item()))
			for batch_id in range(n_batches):
				start_, end_ = int(training_config['bsz'] * batch_id), int(training_config['bsz'] * (batch_id + 1))
				xs = mask_tensors[perm[start_:end_]]
				ys = scores[perm[start_:end_]].view(-1, 1)
				preds = torch.matmul(xs, self.score_tensor) + self.intercept
				mse_loss = self.loss_fn(preds, ys)
				l1loss = self.reg_fn(self.score_tensor, torch.zeros_like(self.score_tensor))
				loss = mse_loss + (self.l1_weight * l1loss)
				print(mse_loss.item(), l1loss.item(), self.l1_weight)
				loss.backward()
# 				print(preds)
# 				print('Grad max : ', torch.max(self.score_tensor.grad), 'Intercept Norm : ',  torch.norm(self.intercept.grad))
				self.optim.step()
				self.optim.zero_grad()
				running_loss_avg += loss.item()
			running_loss_avg /= n_batches
			print('Epoch {} : Loss : {:.5f}'.format(epoch_, running_loss_avg))


	def get_scores(self):
		with torch.no_grad():
			predictions = self.score_tensor.detach()
			stats = predictions.mean().item(), predictions.std().item(), predictions.max().item(), predictions.min().item()
			print("Predictions Stats : Mean {:.7f}, Std {:.7f}, Max {:.7f}, Min {:.7f}".format(*stats))
			return predictions



