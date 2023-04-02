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
	'var_lr' : 1e-2,
	'bsz' : 16,
	'y_var': 0.0025,
	'nepochs' : 20,
	'reg_weight': 1e-4, #1e-6,
	'prior_reg_weight': 1e-2,
	'patience': 5,
}

EPS = 1e-10


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
	def __init__(self, num_players, reg_scales=None):
		super(LinearModel, self).__init__()
		self.score_tensor = nn.parameter.Parameter(create_tensor((num_players + 1, 1)))
		self.variances = nn.parameter.Parameter(create_tensor((num_players + 1, 1))) 
		self.num_players = num_players
		self.base_mask = None
		if reg_scales is None:
			self.l1_weights = torch.zeros_like(self.score_tensor).fill_(training_config['reg_weight'])
		else:
			reg_scales = reg_scales.to(self.score_tensor.device)
			self.l1_weights = torch.ones_like(self.score_tensor) * reg_scales
		self.l1_weights.requires_grad = False
		# TODO [ldery] -- udpdate with relative weights

	def reset_linear(self):
		del self.score_tensor
		self.score_tensor = nn.parameter.Parameter(create_tensor((self.num_players + 1, 1)))
		self.variances = nn.parameter.Parameter(create_tensor((self.num_players + 1, 1)))

	def regLoss(self):
		l1 = self.score_tensor.abs()
		return (l1 * self.l1_weights).sum()

# 		# This is the loss that takes the variances into account.
# 		chosen_vars = self.variances if self.base_mask is None else self.variances[self.base_mask > 0]
# 		chosen_vars = chosen_vars.exp()
# 		chosen_scores = self.score_tensor if self.base_mask is None else self.score_tensor[self.base_mask > 0]
# 		reg_loss = (chosen_scores** 2 / chosen_vars).sum()
# 		return reg_loss

	def forward(self, xs):
		return torch.matmul(xs, self.score_tensor)

	def get_scores(self, base_mask):
		base_mask[-1] = 0 # Do not include the bias term
		with torch.no_grad():
			predictions = self.score_tensor.detach()
			stds = self.variances.detach().exp().sqrt()

			active_preds = predictions[base_mask > 0]
			active_stds = stds[base_mask > 0]

		stats = active_preds.mean().item(), active_preds.std().item(), active_preds.max().item(), active_preds.min().item()
		print("Predictions vals : Mean {:.7f}, Std {:.7f}, Max {:.7f}, Min {:.7f}".format(*stats))

		stats = active_stds.mean().item(), active_stds.std().item(), active_stds.max().item(), active_stds.min().item()
		print("Predictions Stds : Mean {:.7f}, Std {:.7f}, Max {:.7f}, Min {:.7f}".format(*stats))
		return predictions[:-1] # Do not include the last entry because this is the bias

	def set_base_mask(self, base_mask):
		self.base_mask = base_mask.to(self.score_tensor.device)
		assert self.base_mask[-1] == 1, 'The bias term should be on!'


class ScoreModel(nn.Module):
	def __init__(self, module_config, num_layers=12, num_players=None):
		super(ScoreModel, self).__init__()

		reg_scales = []
		reg_dict = {
						'head_z': 10, 'mlp_z' : 1, 
						'intermediate_z':100, 'hidden_z':10
		}
# 		reg_dict = {
# 						'head_z': 1, 'mlp_z' : 1, 
# 						'intermediate_z':1, 'hidden_z':1
# 		}
		for k, v in module_config.items():
			scale = training_config['reg_weight'] * reg_dict[k]
			reg_scales.append(torch.ones(v.numel()) * scale)
		reg_scales.append(torch.tensor([training_config['reg_weight']]))
		reg_scales = torch.concat(reg_scales)
		self.base_model = LinearModel(num_players, reg_scales=reg_scales)
		self.loss_fn = nn.MSELoss()

	def config_to_model_info(self, config, use_all=False):
		base_vec = []
		for k, v in config.items():
			entry = (v > 0).view(-1)
			if use_all:
				entry = torch.ones_like(v).view(-1) > 0
			base_vec.append(entry)
		base_vec.append(torch.tensor([1]))
		return torch.concat(base_vec)

	def forward(self, xs):
		return self.base_model.forward(xs)
	
	def set_base_mask(self, base_mask):
		self.base_model.set_base_mask(self.config_to_model_info(base_mask))

	def get_scores(self, base_mask):
		return self.base_model.get_scores(self.config_to_model_info(base_mask))

	def get_candidate(self, pool_size=5000):
		return None
# 		pool_size = 1 #if self.base_model.base_mask is None else pool_size
# 		base_set = torch.zeros(pool_size, self.base_model.num_players + 1, device=self.base_model.score_tensor.device) + 0.5
# 		random_sample = torch.bernoulli(base_set)
# 		return random_sample.cpu()
# 		if self.base_model.base_mask is None:
# 			return random_sample.cpu()
		
# 		random_sample *= self.base_model.base_mask
# 		# Do a selection based on UCB
# 		with torch.no_grad():
# 			stds = (self.base_model.variances.exp()).sqrt() / (self.base_model.base_mask.sum()).item()
# 			preds_ = random_sample.matmul(self.base_model.score_tensor)
# 			stds_ = random_sample.matmul(stds)
# 			total = preds_ + stds_
# 			print(total.mean().item(), total.max().item(), total.std().item())
# 			chosen = random_sample[torch.argmax(total)].cpu()
# 		return chosen

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
				print("Loss {:.5f} | Reg Loss {:.5f}".format(loss / training_config['y_var'] , training_config['reg_weight'] * regLoss))

			# Do some logging here
			with torch.no_grad():
				errors = (preds - this_ys).abs()
				this_max_error = errors.max().item()
				this_min_error = errors.min().item()

			max_error = max(max_error, this_max_error)
			min_error = min(min_error, this_min_error)

			if is_train:
				# Clamp the losses to be within bounds of the training data likelihood.
				loss = (loss / training_config['y_var'])  + (training_config['reg_weight'] * regLoss)
				loss.backward()

				self.score_optim.step()
				self.score_optim.zero_grad()

			running_loss_avg += loss.item()
		running_loss_avg /= n_batches
		if not is_train:
			self.base_model.train()
		return running_loss_avg, max_error, min_error
	
	def fit_variances(self, xs, ys):
		xs = torch.stack(xs).float().cuda()
		ys = ys.view(1, -1)
		var_optim = Adam([self.base_model.variances], lr=training_config['var_lr'])

		def marginal_likelihood(xs, ys):
			C_mat = torch.eye(ys.shape[-1], device=ys.device)
			C_mat += (xs * self.base_model.variances.exp().T).matmul(xs.T)
			l1 = torch.logdet(C_mat)
			l2 = ys.matmul(torch.inverse(C_mat)).matmul(ys.T) / training_config['y_var']
			return l1, l2

		for iter_ in range(training_config['nepochs']):
			loss_terms = marginal_likelihood(xs, ys)
			loss = loss_terms[0] + training_config['prior_reg_weight'] * loss_terms[1]
			print('[Epoch {}] | Loss : {:.5f} | T1 : {:.5f} | T2 : {:.5f} '.format(iter_, loss.item(), loss_terms[0].item(), training_config['prior_reg_weight'] * loss_terms[1].item()))
			
			loss.backward()
			grads = self.base_model.variances.grad
			var_optim.step()
			var_optim.zero_grad()

			if (iter_ % 5) == 0:
				with torch.no_grad():
					chosen = self.base_model.variances.exp()
					if self.base_model.base_mask is not None:
						chosen = self.base_model.variances[self.base_model.base_mask > 0] 
					mean_, max_, min_ = chosen.mean().item(), chosen.max().item(), chosen.min().item()
					print('\t', 'Mean : {:.5f} | Max : {:.5f} | Min : {:.5f}'.format(mean_, max_, min_))


	def update_with_info(self, run_info):
		xs, ys = run_info
		normalization = self.base_model.num_players if self.base_model.base_mask is None else self.base_model.base_mask.sum().item()
		normalization = np.sqrt(normalization)
		print('This is the normalization : ', normalization)
		# Split into train and test
		perm = np.random.permutation(len(ys))
		trn_list, tst_list = perm[:int(0.8 * len(perm))], perm[int(0.8 * len(perm)):]

		tr_xs, tr_ys = [xs[i] / normalization for i in trn_list], [ys[i] for i in trn_list]
		tr_ys  = torch.tensor(tr_ys).float().cuda()

		ts_xs, ts_ys = [xs[i] / normalization for i in tst_list], [ys[i] for i in tst_list]
		ts_ys  = torch.tensor(ts_ys).float().cuda()


# 		self.fit_variances(ts_xs, ts_ys)

		# Setup the optimizer.
		self.score_optim = Adam([self.base_model.score_tensor], lr=training_config['lr'])

		best_loss, clone, since_best = 1e10, None, 0
		for epoch_ in range(training_config['nepochs']):

			print('Epoch - ', epoch_)
			run_out = self.run_epoch(tr_xs, tr_ys)
			print('Epoch {} [Trn] | Loss : {:.5f} | Max Error : {:.5f} | Min Error : {:.5f}'.format(epoch_, *run_out))
			run_out = self.run_epoch(ts_xs, ts_ys, is_train=False)
			print('Epoch {} [Tst] | Loss : {:.5f} | Max Error : {:.5f} | Min Error : {:.5f}'.format(epoch_, *run_out))

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