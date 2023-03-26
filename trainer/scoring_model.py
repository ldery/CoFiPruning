import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import torch.nn.functional as F
import numpy as np
import math
from scipy.special import comb
import pdb
from tqdm import tqdm
from utils import *
from pprint import pprint
from copy import deepcopy
from torch.nn.init import kaiming_normal_


training_config = {
	'lr' : 1e-4,
	'var_lr': 1e2, #1.0,
	'bsz' : 16,
	'nepochs' : 20, #5
	'reg_weight': 1.0,
	'reg_reg_weight': 1e-2,
	'd_in': 3,
	'd_rep': 10,
	'dp_prob': 0.1,
	'patience': 5,
	'patience_factor': 0.5
}

EPS = 1e-10

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
		if n_samples > len_:
			return self.buffer
		proba = np.arange(len_) + 1
		proba = proba / proba.sum()
		chosen = np.random.choice(len_, size=n_samples, replace=False, p=proba)
		for i in chosen:
			result[0].append(self.buffer[0][i])
			result[1].append(self.buffer[1][i])
		return result

def create_tensor(shape, zero_out=1.0, requires_grad=True, is_cuda=True):
	inits = torch.zeros(*shape) #.uniform_(-1/shape[0], 1/shape[0]) * zero_out
	# Create the weights
	weights = inits.float().cuda() if is_cuda else inits.float()
	if requires_grad:
		weights.requires_grad = True
	return weights

def get_score_model(config, num_layers, num_players, model_type):
	if model_type == 'logistic_regression':
		return LogisticReg(num_players=num_players)
	elif model_type == 'non-linear':
		return ScoreModel(config, num_layers=num_layers, num_players=num_players)


class LinearModel(nn.Module):
	def __init__(self, num_players):
		super(LinearModel, self).__init__()
		self.score_tensor = nn.parameter.Parameter(create_tensor((num_players, 1)))
		self.variances = nn.parameter.Parameter(create_tensor((num_players, 1))) 
		self.num_players = num_players
		self.base_mask = None

	def reset_linear(self):
		del self.score_tensor
		self.score_tensor = nn.parameter.Parameter(create_tensor((self.num_players, 1)))
		self.variances = nn.parameter.Parameter(create_tensor((self.num_players, 1)))

	def regLoss(self):
		# This is the loss that takes the variances into account.
		chosen_vars = self.variances if self.base_mask is None else self.variances[self.base_mask > 0]
		chosen_scores = self.score_tensor if self.base_mask is None else self.score_tensor[self.base_mask > 0]
		v_and_w = ((chosen_scores**2) / (torch.exp(chosen_vars) + EPS)).sum()
		v_only = chosen_vars.sum()
		return v_only, v_and_w

	def forward(self, xs):
		return torch.matmul(xs, self.score_tensor)
	
	def get_scores(self, base_mask):
		with torch.no_grad():
			predictions = self.score_tensor.detach()
			stds = self.variances.detach().exp().sqrt()

			active_preds = predictions[base_mask > 0]
			active_stds = stds[base_mask > 0]

		stats = active_preds.mean().item(), active_preds.std().item(), active_preds.max().item(), active_preds.min().item()
		print("Predictions vals : Mean {:.7f}, Std {:.7f}, Max {:.7f}, Min {:.7f}".format(*stats))

		stats = active_stds.mean().item(), active_stds.std().item(), active_stds.max().item(), active_stds.min().item()
		print("Predictions Stds : Mean {:.7f}, Std {:.7f}, Max {:.7f}, Min {:.7f}".format(*stats))
		return predictions

	def set_base_mask(self, base_mask):
		self.base_mask = base_mask


class ScoreModel(nn.Module):
	def __init__(self, module_config, num_layers=12, num_players=None):
		super(ScoreModel, self).__init__()

		# Do some special init
		self.base_model = LinearModel(num_players)
		self.loss_fn = nn.MSELoss()

	def config_to_model_info(self, config, use_all=False):
		base_vec = []
		for k, v in config.items():
			entry = (v > 0).view(-1)
			if use_all:
				entry = torch.ones_like(v).view(-1) > 0
			base_vec.append(entry)
		return torch.concat(base_vec)

	def forward(self, xs):
		return self.base_model.forward(xs)
	
	def set_base_mask(self, base_mask):
		self.base_model.set_base_mask(self.config_to_model_info(base_mask))

	def get_scores(self, base_mask):
		return self.base_model.get_scores(self.config_to_model_info(base_mask))
	
	def run_epoch(self, xs, ys, is_train=True):
		# generate a permutation
		perm = np.random.permutation(len(ys))
		running_loss_avg = 0.0
		n_batches = math.ceil(len(ys) / training_config['bsz'])
		if not is_train:
			self.base_model.eval()
		max_error, min_error = -1, 1
		for batch_id in range(n_batches):
			start_, end_ = int(training_config['bsz'] * batch_id), int(training_config['bsz'] * (batch_id + 1))
			xs_masks = torch.stack([xs[i_] for i_ in perm[start_:end_]]).float().cuda()

			this_ys = ys[perm[start_:end_]].view(-1, 1)
			# do the forward pass heres
			preds = self.forward(xs_masks)

			loss = self.loss_fn(preds, this_ys)
			regLoss = self.base_model.regLoss()

			if batch_id == 0:
				for p_ in zip(
					preds[:3].squeeze().detach().cpu().numpy().tolist(), this_ys[:3].squeeze().detach().cpu().numpy().tolist()):
					print("Pred {:.3f} | GT {:.3f}".format(*p_))
				print("Loss {:.3f} | Reg V {:.6f} | Reg W/V {:.5f}".format(loss, *regLoss))

			# Do some logging here
			with torch.no_grad():
				errors = (preds - this_ys).abs()
				this_max_error = errors.max().item()
				this_min_error = errors.min().item()

			max_error = max(max_error, this_max_error)
			min_error = min(min_error, this_min_error)

			if is_train:
				# Clamp the losses to be within bounds of the training data likelihood.
				regLoss = torch.clamp(regLoss[0], -1.0, 1.0), torch.clamp(regLoss[1], -1.0, 1.0)
				loss = loss + (training_config['reg_weight'] * regLoss[1] + training_config['reg_reg_weight'] * regLoss[0])
				loss.backward()
# 				print(self.base_model.variances.grad.min().item(), self.base_model.variances.grad.mean().item(), self.base_model.variances.grad.max().item() )
# 				nn.utils.clip_grad_norm_(self.base_model.parameters(), training_config['gradient_clip'])
# 				print('After : ', self.base_model.variances.grad.mean().item(), self.base_model.variances.grad.max().item() )

				self.score_optim.step()
				self.score_optim.zero_grad()
				
				self.var_optim.step()
				self.var_optim.zero_grad()
				
			running_loss_avg += loss.item()
		running_loss_avg /= n_batches
		if not is_train:
			self.base_model.train()
		return running_loss_avg, max_error, min_error

	def update_with_info(self, run_info):
		# Setup the optimizer

		self.score_optim = Adam([self.base_model.score_tensor], lr=training_config['lr'])
		self.var_optim = SGD([self.base_model.variances], lr=training_config['var_lr'])

		xs, ys = run_info

		# Split into train and test
		perm = np.random.permutation(len(ys))
		trn_list, tst_list = perm[:int(0.8 * len(perm))], perm[int(0.8 * len(perm)):]

		tr_xs, tr_ys = [xs[i] for i in trn_list], [ys[i] for i in trn_list]
		tr_ys  = torch.tensor(tr_ys).float().cuda()
		
		ts_xs, ts_ys = [xs[i] for i in tst_list], [ys[i] for i in tst_list]
		ts_ys  = torch.tensor(ts_ys).float().cuda()

		best_loss, clone, since_best = 1e10, None, 0
		for epoch_ in range(training_config['nepochs']):

			print('Epoch - ', epoch_)
			run_out = self.run_epoch(tr_xs, tr_ys)
			print('Epoch {} [Trn] | Loss : {:.5f} | Max Error : {:.5f} | Min Error : {:.5f}'.format(epoch_, *run_out))
			run_out = self.run_epoch(ts_xs, ts_ys, is_train=False)
			print('Epoch {} [Tst] | Loss : {:.5f} | Max Error : {:.5f} | Min Error : {:.5f}'.format(epoch_, *run_out))

# 			active_stds = self.base_model.variances.detach().exp().sqrt()
# 			stats = active_stds.mean().item(), active_stds.std().item(), active_stds.max().item(), active_stds.min().item()
# 			print("Predictions Stds : Mean {:.7f}, Std {:.7f}, Max {:.7f}, Min {:.7f}".format(*stats))

			if run_out[0] < best_loss:
				print('New Best Model Achieved')
				best_loss = run_out[0]
				clone = deepcopy(self.base_model)
				since_best = 0

			since_best += 1
			if since_best > 3:
				break

		del self.base_model
		self.base_model = clone