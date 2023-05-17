import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
from collections import Counter

training_config = {
	'lr' :5e-4,
	'var_lr' : 1.0,
	'bsz' : 256,
	'y_var': 1e-4, #25,
	'nepochs' : 30,
	'prior_reg_weight': 2.0,
	'patience': 10,
}

EPS = 1e-10


def create_tensor(shape, zero_out=1.0, requires_grad=True, is_cuda=True):
	inits = torch.zeros(*shape) #.uniform_(-1/shape[0], 1/shape[0]) * zero_out
	# Create the weights
	weights = inits.float().cuda() if is_cuda else inits.float()
	if requires_grad:
		weights.requires_grad = True
	return weights

def get_score_model(config, num_layers, num_players, model_type, reg_weight, wandb):
	return ScoreModel(config, num_layers=num_layers, num_players=num_players, reg_weight=reg_weight, wandb=wandb)


class LinearModel(nn.Module):
	def __init__(self, num_players, reg_weight=None):
		super(LinearModel, self).__init__()
		self.score_tensor = nn.parameter.Parameter(create_tensor((num_players + 1, 1)))
		self.num_players = num_players
		self.base_mask = None
		self.reg_weight = reg_weight

	def set_prior_variance(self, prior_variance):
		self.prior_variance = prior_variance

	def reset_linear(self, bias_init):
		del self.score_tensor
		self.score_tensor = nn.parameter.Parameter(create_tensor((self.num_players + 1, 1)))
		with torch.no_grad():
			self.score_tensor[-1, 0] = bias_init

	def regLoss(self, mean_score_tensor):
		mean_score_tensor = 0.0 if mean_score_tensor is None else mean_score_tensor
		weighted_l2 = (self.score_tensor - mean_score_tensor)**2  * self.prior_variance
		return weighted_l2.sum()

	def forward(self, xs):
		return torch.matmul(xs, self.score_tensor)

	def get_scores(self, base_mask):
		assert base_mask[-1] == 1, 'The bias term should be on'
		with torch.no_grad():
			predictions = self.score_tensor.detach()
			active_preds = predictions[base_mask > 0][:-1, :]

		stats = active_preds.mean().item(), active_preds.std().item(), active_preds.max().item(), active_preds.min().item()
		print("Predictions vals : Mean {:.7f}, Std {:.7f}, Max {:.7f}, Min {:.7f}".format(*stats))

		return predictions[:-1] # Do not include the last entry because this is the bias

	def set_base_mask(self, base_mask):
		self.base_mask = base_mask.to(self.score_tensor.device)
		assert self.base_mask[-1] == 1, 'The bias term should be on!'


class ScoreModel(nn.Module):
	def __init__(self, module_config, num_layers=12, num_players=None, reg_weight=1e-7, wandb=None):
		super(ScoreModel, self).__init__()

		self.reg_weight = reg_weight
		self.base_model = LinearModel(num_players, reg_weight=reg_weight)
		self.loss_fn = nn.MSELoss()
		self.candidate_buffer = []
		self.curr_norm = 1.0
		self.wandb = wandb
		self.overall_iter = 0

	def set_prior_variance(self, prior_variance):
		self.prior_variance = prior_variance
		self.base_model.set_prior_variance(prior_variance)

	def reset_linear(self, bias_init):
		self.base_model.reset_linear(bias_init)

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

	def get_candidate(self, pool_size=10000):
		if len(self.candidate_buffer) == 0: 
			pool_size = 1 #if self.base_model.base_mask is None else pool_size
			base_set = torch.zeros(pool_size, self.base_model.num_players + 1, device=self.base_model.score_tensor.device) + 0.5
			random_sample = torch.bernoulli(base_set)
# 			if self.base_model.base_mask is None:
			return random_sample.cpu().squeeze()

# 			random_sample *= self.base_model.base_mask
# 			random_sample[:, -1] = 1
# 			normed_rand_sample = random_sample / self.curr_norm

# 			# Do a selection based on UCB
# 			with torch.no_grad():
# 				preds_ = normed_rand_sample.matmul(self.base_model.score_tensor)
# 				normed_rand_sample = normed_rand_sample

# 				stds_ = (normed_rand_sample.matmul(self.cov_mat) * normed_rand_sample).sum(axis=-1, keepdim=True)
# 				stds_ = stds_.sqrt()
# 				print('Pred range : ', preds_.max().item(), preds_.min().item())
# 				print('Stds range : ', stds_.max().item(), stds_.min().item())
# 				total = (preds_ + stds_).squeeze()
# 				print('Totals range : ', total.max().item(), total.min().item())
# 				stats = total.mean().item(), total.max().item(), total.min().item()
# 				if self.wandb is not None:
# 					self.wandb.log({
# 						'epoch': self.overall_iter, 'candidates/mean': stats[0],
# 						'candidates/max': stats[1], 'candidates/min': stats[2]
# 					})
# 				print('[Exp] Mean : {:.4f} | Max : {:.4f} | Min : {:.4f}'.format(*stats))
# 				chosen_idx = torch.argsort(total)[-10:]
# 				chosen = random_sample[chosen_idx].cpu()
# 				self.candidate_buffer.extend(chosen.unbind())

		return self.candidate_buffer.pop()

	def run_epoch(self, epoch_, xs, ys, mean=None, is_train=True):
		# generate a permutation
		perm = np.random.permutation(len(ys))
		running_loss_avg = 0.0
		n_batches = math.ceil(len(ys) / training_config['bsz'])
		if not is_train:
			self.base_model.eval()
		max_error, min_error = -1, 1
		all_preds, all_ys = [], []
		for batch_id in range(n_batches):
			start_, end_ = int(training_config['bsz'] * batch_id), int(training_config['bsz'] * (batch_id + 1))
			xs_masks = torch.stack([xs[i_] for i_ in perm[start_:end_]]).float().cuda()

			this_ys = ys[perm[start_:end_]].view(-1, 1)
			# do the forward pass heres
			preds = self.forward(xs_masks)

			loss = self.loss_fn(preds, this_ys)
			regLoss = self.base_model.regLoss(mean) * self.reg_weight

			all_ys.append(this_ys.cpu().numpy())
			all_preds.append(preds.detach().cpu().numpy())

			if batch_id == 0:
				for p_ in zip(
					preds[:3].squeeze().detach().cpu().numpy().tolist(), this_ys[:3].squeeze().detach().cpu().numpy().tolist()):
					print("Pred {:.3f} | GT {:.3f}".format(*p_))
				print("Loss {:.5f} | Reg Loss {:.5f}".format(loss, regLoss))

			# Do some logging here
			with torch.no_grad():
				errors = (preds - this_ys).abs()
				this_max_error = errors.max().item()
				this_min_error = errors.min().item()

			max_error = max(max_error, this_max_error)
			min_error = min(min_error, this_min_error)

			if is_train:
				# Clamp the losses to be within bounds of the training data likelihood.
				loss = loss + regLoss
				loss.backward()

				self.score_optim.step()
				self.score_optim.zero_grad()

			running_loss_avg += loss.item()
		running_loss_avg /= n_batches
		if not is_train:
			self.base_model.train()
		# Compute the kendalltau statistic
		kendall_tau = kendalltau(np.concatenate(all_preds), np.concatenate(all_ys)).correlation
		return running_loss_avg, max_error, min_error, kendall_tau


	def update_with_info(self, run_info):
		self.overall_iter += 1
		xs, ys = run_info
		normalization = self.base_model.num_players if self.base_model.base_mask is None else self.base_model.base_mask.sum().item()

		normalization = np.sqrt(normalization)
		self.set_prior_variance(1.0/normalization)
		self.curr_norm = normalization
		print('This is the normalization : ', normalization)

		# Split into train and test
		perm = np.random.permutation(len(ys))
		trn_list, tst_list = perm[:int(0.8 * len(perm))], perm[int(0.8 * len(perm)):]

		tr_xs, tr_ys = [xs[i] / normalization for i in trn_list], [ys[i] for i in trn_list]
		tr_ys  = torch.tensor(tr_ys).float().cuda()

		ts_xs, ts_ys = [xs[i] / normalization for i in tst_list], [ys[i] for i in tst_list]
		ts_ys  = torch.tensor(ts_ys).float().cuda()

		with torch.no_grad():
			mean_score_tensor = self.base_model.score_tensor.detach()
			mean_score_tensor.requires_grad = False

		bias_init = 0 #tr_ys.mean().item() * normalization
		self.base_model.reset_linear(bias_init)

		# Setup the optimizer.
		self.score_optim = Adam([self.base_model.score_tensor], lr=training_config['lr'])
		lr_scheduler = ReduceLROnPlateau(self.score_optim, mode='max', factor=0.5, patience=3, verbose=True, min_lr=1e-5)
		best_tau, clone, since_best = -1e5, None, 0
		for epoch_ in range(training_config['nepochs']):

			print('Epoch - ', epoch_)
			run_out = self.run_epoch(epoch_, tr_xs, tr_ys, mean=mean_score_tensor)
			print('Epoch {} [Trn] | Loss : {:.5f} | Max Error : {:.5f} | Min Error : {:.5f} | Kendall Tau : {:.5f}'.format(epoch_, *run_out))
			run_out = self.run_epoch(epoch_, ts_xs, ts_ys, is_train=False)
			print('Epoch {} [Tst] | Loss : {:.5f} | Max Error : {:.5f} | Min Error : {:.5f} | Kendall Tau : {:.5f}'.format(epoch_, *run_out))

			if run_out[-1] > best_tau:
				print('New Best Score Model Achieved')
				best_tau = run_out[-1]
				clone = deepcopy(self.base_model)
				since_best = 0

			lr_scheduler.step(run_out[-1]) # step based on the max error
			since_best += 1
			if since_best > training_config['patience']:
				break

		del self.base_model
		self.base_model = clone

		with torch.no_grad():
			tr_xs = torch.stack(tr_xs).cuda()

			variances = (tr_xs.T).matmul(tr_xs)
			print('Max : ', variances.max(), 'Min : ', variances.min().item())
			variances += (torch.eye(tr_xs.shape[-1]).cuda() * self.prior_variance)
			self.cov_mat = training_config['y_var'] * torch.linalg.inv(variances)

			ts_xs = torch.stack(ts_xs).float().cuda()
			ts_ys = ts_ys.cpu().numpy()
			with torch.no_grad():
				preds = self.base_model.forward(ts_xs).squeeze()
				preds = preds.detach().cpu().numpy()

			base_pred = np.ones_like(ts_ys) * tr_ys.mean().item()
			base_coef_det = kendalltau(ts_ys, base_pred)
			print('Our KendallTau = ', best_tau, ' | Naive Mean KendallTau = ', base_coef_det.correlation)
			our_max_err = max(np.abs(ts_ys - preds))
			base_maxerr = max(np.abs(ts_ys - base_pred))
			print('Our MaxError = {:.4f} | Naive Mean MaxErr = {:.4f}'.format(our_max_err, base_maxerr))
			our_loss = np.sqrt(((preds - ts_ys)**2).mean())
			naive_loss = np.sqrt(((base_pred - ts_ys)**2).mean())
			print('Our Loss = {:.5f} | Naive Mean Loss = {:.5f}'.format(our_loss, naive_loss))
			if self.wandb is not None:
				self.wandb.log({
					'epoch': self.overall_iter,
					'kt/ours': best_tau, 'kt/naive': base_coef_det,
					'loss/ours': our_loss, 'loss/naive': naive_loss,
					'maxerr/ours': our_max_err, 'maxerr/naive': base_maxerr,
				})

