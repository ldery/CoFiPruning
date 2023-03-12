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
	'nepochs' : 5,
	'l1_weight': 0,
	'patience': 20,
	'patience_factor': 0.5
}

class Buffer(object):
	def __init__(self, max_size=500):
		self.max_size = max_size
		self.buffer = [[], []] 
	
	def size(self):
		return len(self.buffer[0])

	def add_to_buffer(self, x, y):
		self.buffer[0].extend(x)
		self.buffer[1].extend(y)
		if len(self.buffer[0]) > self.max_size:
			self.buffer[0] = self.buffer[0][-self.max_size:]
			self.buffer[1] = self.buffer[1][-self.max_size:]

	def sample(self, n_samples):
		len_ = self.size()
		result = [[], []]
		if len_ < n_samples:
			return result
		proba = np.arange(len_) + 1
		proba = proba / proba.sum()
		chosen = np.random.choice(len_, size=n_samples, replace=False, p=proba)
		for i_ in chosen:
			result[0].append(self.buffer[0][i])
			result[1].append(self.buffer[1][i])
		return result


def create_tensor(shape, zero_out=1.0, requires_grad=True, is_cuda=True):
	inits = torch.zeros(*shape)
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
		self.loss_fn = nn.MSELoss()
		self.reg_fn =  nn.L1Loss() 
		self.l1_weight = training_config['l1_weight']
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			self.optim, factor=training_config['patience_factor'], patience=training_config['patience']
		)
		self.buffer = Buffer()

	def update_with_info(self, run_info):
		mask_tensors, scores = run_info
		# get the extra samples from the buffer
		replay_masks, replay_scores = self.buffer.sample(len(scores))
		self.buffer.add_to_buffer(mask_tensors, scores)

		mask_tensors.extend(replay_masks)
		scores.extend(replay_scores)
		mask_tensors = torch.stack(mask_tensors).to(self.score_tensor.device)
		scores = torch.tensor(scores).to(self.score_tensor.device)
		num_masks = len(mask_tensors)
		for epoch_ in range(training_config['nepochs']):
			# generate a permutation
			perm = np.random.permutation(num_masks)
			running_loss_avg = 0.0
			n_batches = math.ceil(num_masks / training_config['bsz'])

			for batch_id in range(n_batches):
				start_, end_ = int(training_config['bsz'] * batch_id), int(training_config['bsz'] * (batch_id + 1))
				xs = mask_tensors[perm[start_:end_]]
				ys = scores[perm[start_:end_]].view(-1, 1)
				preds = torch.matmul(xs, self.score_tensor) + self.intercept
				mse_loss = self.loss_fn(preds, ys)
				l1loss = self.reg_fn(self.score_tensor, torch.zeros_like(self.score_tensor))
				loss = mse_loss + (self.l1_weight * l1loss)

				loss.backward()

				self.optim.step()
				self.optim.zero_grad()
				running_loss_avg += loss.item()
			running_loss_avg /= n_batches
			self.scheduler.step(running_loss_avg)
			print('Epoch {} : Loss : {:.5f}'.format(epoch_, running_loss_avg))


	def get_scores(self):
		with torch.no_grad():
			predictions = self.score_tensor.detach()
			stats = predictions.mean().item(), predictions.std().item(), predictions.max().item(), predictions.min().item()
			print("Predictions Stats : Mean {:.7f}, Std {:.7f}, Max {:.7f}, Min {:.7f}".format(*stats))
			return predictions



