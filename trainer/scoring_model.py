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
from collections import Counter

training_config = {
	'lr' :1e-4,
	'var_lr' : 1.0,
	'bsz' : 16,
	'y_var': 0.0001,#25,
	'nepochs' : 30,
	'prior_reg_weight': 2.0,
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

def get_score_model(config, num_layers, num_players, model_type, reg_weight, wandb):
	return ScoreModel(config, num_layers=num_layers, num_players=num_players, reg_weight=reg_weight, wandb=wandb)


class LinearModel(nn.Module):
	def __init__(self, num_players, reg_scales=None):
		super(LinearModel, self).__init__()
		self.score_tensor = nn.parameter.Parameter(create_tensor((num_players + 1, 1)))
		self.variances = nn.parameter.Parameter(create_tensor((num_players + 1, 1)) - np.log(num_players**2)) 
		self.num_players = num_players
		self.base_mask = None
		self.l1_weights = reg_scales.to(self.score_tensor.device).view(self.score_tensor.shape)
		self.l1_weights.requires_grad = False

	def reset_linear(self):
		del self.score_tensor
		del self.variances
		self.score_tensor = nn.parameter.Parameter(create_tensor((self.num_players + 1, 1)))
		init_scale = - np.log((self.base_mask.sum().item())**2)
		self.variances = nn.parameter.Parameter(create_tensor((self.num_players + 1, 1)) + init_scale)

	def regLoss(self):
		inv_variances = 1.0 / self.variances.exp()
		inv_variances *= self.l1_weights
		weighted_l2 = (self.score_tensor**2  * inv_variances)[:-1, :]
		return weighted_l2.sum()
		
# 		l1 = self.score_tensor.abs()
# 		if self.base_mask is not None:
# 			l1 = l1 * (self.base_mask.view(l1.shape))
# 		return (l1 * self.l1_weights).sum()


	def forward(self, xs):
		return torch.matmul(xs, self.score_tensor)

	def get_scores(self, base_mask):
		base_mask[-1] = 0 # Do not include the bias term
		with torch.no_grad():
			predictions = self.score_tensor.detach()
			stds = self.variances.detach().exp().sqrt()

			active_preds = predictions[base_mask > 0]
			base_mask[-1] = 0
			active_stds = stds[base_mask > 0]
			base_mask[-1] = 1

		stats = active_preds.mean().item(), active_preds.std().item(), active_preds.max().item(), active_preds.min().item()
		print("Predictions vals : Mean {:.7f}, Std {:.7f}, Max {:.7f}, Min {:.7f}".format(*stats))

		stats = active_stds.mean().item(), active_stds.std().item(), active_stds.max().item(), active_stds.min().item()
		print("Predictions Stds : Mean {:.7f}, Std {:.7f}, Max {:.7f}, Min {:.7f}".format(*stats))

		return predictions[:-1] # Do not include the last entry because this is the bias

	def set_base_mask(self, base_mask):
		self.base_mask = base_mask.to(self.score_tensor.device)
		assert self.base_mask[-1] == 1, 'The bias term should be on!'

class ScoreModel(nn.Module):
	def __init__(self, module_config, num_layers=12, num_players=None, reg_weight=1e-7, wandb=None):
		super(ScoreModel, self).__init__()

		reg_scales = []
		reg_dict = {
						'head_z': 1, 'mlp_z' : 1, 
						'intermediate_z':1, 'hidden_z':1
		}
		for k, v in module_config.items():
			scale = reg_weight * reg_dict[k]
			reg_scales.append(torch.ones(v.numel()) * scale)
		reg_scales.append(torch.tensor([reg_weight]))
		reg_scales = torch.concat(reg_scales)
		self.base_model = LinearModel(num_players, reg_scales=reg_scales)
		self.loss_fn = nn.MSELoss()
		self.candidate_buffer = []
		self.curr_norm = 1.0
		self.wandb = wandb
		self.overall_iter = 0
	
	def reset_linear(self):
		self.base_model.reset_linear()

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
			pool_size = 1 if self.base_model.base_mask is None else pool_size
			base_set = torch.zeros(pool_size, self.base_model.num_players + 1, device=self.base_model.score_tensor.device) + 0.5
			random_sample = torch.bernoulli(base_set)
			if self.base_model.base_mask is None:
				return random_sample.cpu().squeeze()

			random_sample *= self.base_model.base_mask
			random_sample[:, -1] = 1
			normed_rand_sample = random_sample / self.curr_norm

			# Do a selection based on UCB
			with torch.no_grad():
				stds = (self.base_model.variances.exp()).sqrt()
				preds_ = normed_rand_sample.matmul(self.base_model.score_tensor)

				normed_rand_sample[:, -1] = 0
				stds_ = normed_rand_sample.matmul(stds)
				total = (preds_ + stds_).squeeze()
				stats = total.mean().item(), total.max().item(), total.min().item()
				if self.wandb is not None:
					self.wandb.log({
						'epoch': self.overall_iter, 'candidates/mean': stats[0],
						'candidates/max': stats[1], 'candidates/min': stats[2]
					})
				print('[Exp] Mean : {:.4f} | Max : {:.4f} | Min : {:.4f}'.format(*stats))
				chosen_idx = torch.argsort(total)[-10:]
				chosen = random_sample[chosen_idx].cpu()
				self.candidate_buffer.extend(chosen.unbind())

		return self.candidate_buffer.pop()

	def run_epoch(self, epoch_, xs, ys, is_train=True):
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
		return running_loss_avg, max_error, min_error


	def fit_variances(self, tr, ts):

		def print_state(desc):
			with torch.no_grad():
				chosen = (self.base_model.variances.exp().sqrt())[:-1]
				if self.base_model.base_mask is not None:
					chosen = chosen[self.base_model.base_mask[:-1] > 0] 
				mean_, max_, min_ = chosen.mean().item(), chosen.max().item(), chosen.min().item()
				print('\t', desc, '[Vars] Mean : {:.7f} | Max : {:.7f} | Min : {:.7f}'.format(mean_, max_, min_))

		print_state('Before Iter')

		# We are removing the last dimension to make the optimization more stable
		tr_xs, tr_ys = tr
		tr_xs = torch.stack(tr_xs)[:, :-1].float().cuda()
		tr_ys = tr_ys.view(1, -1)

		ts_xs, ts_ys = ts
		ts_xs = torch.stack(ts_xs)[:, :-1].float().cuda()
		ts_ys = ts_ys.view(1, -1)
		
		var_optim = Adam([self.base_model.variances], lr=training_config['var_lr'])

		def marginal_likelihood(xs, ys):
			vars_ = (self.base_model.variances.exp().T)[:, :-1] / training_config['y_var']
			C_mat = torch.eye(ys.shape[-1], device=ys.device)
			C_mat += (xs * vars_).matmul(xs.T)
			l1 = ys.matmul(torch.inverse(C_mat)).matmul(ys.T)
			l2 = torch.logdet(C_mat)
			return l1, l2

		best_loss, clone, since_best = 1e10, None, 0
		for iter_ in range(training_config['nepochs']):
			loss_terms = marginal_likelihood(tr_xs, tr_ys)
			reg_term = training_config['prior_reg_weight'] * loss_terms[1]
			loss = loss_terms[0] + reg_term #torch.clamp(reg_term, min=0.0)
			print('[Epoch {}] | Loss : {:.7f} | T1 : {:.7f} | T2 : {:.7f} '.format(
				iter_, loss.item(), loss_terms[0].item(), reg_term.item())
			)

			loss.backward()
			var_optim.step()
			var_optim.zero_grad()

			with torch.no_grad():
				loss_terms = marginal_likelihood(ts_xs, ts_ys)
				reg_term = training_config['prior_reg_weight'] * loss_terms[1]
				loss = loss_terms[0] + reg_term #torch.clamp(reg_term, min=0.0)
			# check the validation performance and cache
			if loss.item() < best_loss:
				print('New Best Variance Model Achieved - old = {:.7f} | new = {:.7f}'.format(best_loss, loss.item()))
				best_loss = loss.item()
				clone = deepcopy(self.base_model)
				since_best = 0

			since_best += 1
			if since_best > 5:
				break

		del self.base_model
		self.base_model = clone
		print_state('After Iter')


	def update_with_info(self, run_info):
		self.overall_iter += 1
		xs, ys = run_info
		normalization = self.base_model.num_players if self.base_model.base_mask is None else self.base_model.base_mask.sum().item()
		normalization = np.sqrt(normalization)
		self.curr_norm = normalization
		print('This is the normalization : ', normalization)
		# Split into train and test
		perm = np.random.permutation(len(ys))
		trn_list, tst_list = perm[:int(0.8 * len(perm))], perm[int(0.8 * len(perm)):]

		tr_xs, tr_ys = [xs[i] / normalization for i in trn_list], [ys[i] for i in trn_list]
		tr_ys  = torch.tensor(tr_ys).float().cuda()

		ts_xs, ts_ys = [xs[i] / normalization for i in tst_list], [ys[i] for i in tst_list]
		ts_ys  = torch.tensor(ts_ys).float().cuda()

		self.fit_variances((tr_xs, tr_ys), (ts_xs, ts_ys))

		# Setup the optimizer.
		self.score_optim = Adam([self.base_model.score_tensor], lr=training_config['lr'])
		lr_scheduler = ReduceLROnPlateau(self.score_optim, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-5)
		best_loss, clone, since_best = 1e10, None, 0
		for epoch_ in range(training_config['nepochs']):

			print('Epoch - ', epoch_)
			run_out = self.run_epoch(epoch_, tr_xs, tr_ys)
			print('Epoch {} [Trn] | Loss : {:.5f} | Max Error : {:.5f} | Min Error : {:.5f}'.format(epoch_, *run_out))
			run_out = self.run_epoch(epoch_, ts_xs, ts_ys, is_train=False)
			print('Epoch {} [Tst] | Loss : {:.5f} | Max Error : {:.5f} | Min Error : {:.5f}'.format(epoch_, *run_out))

			if run_out[0] < best_loss:
				print('New Best Score Model Achieved')
				best_loss = run_out[0]
				clone = deepcopy(self.base_model)
				since_best = 0

			lr_scheduler.step(run_out[0]) # step based on the max error
			since_best += 1
			if since_best > 5:
				break

		del self.base_model
		self.base_model = clone

		with torch.no_grad():
			ts_xs = torch.stack(ts_xs).float().cuda()
			ts_ys = ts_ys.cpu().numpy()
			with torch.no_grad():
				preds = self.base_model.forward(ts_xs).squeeze()
				preds = preds.detach().cpu().numpy()

			our_coef_det = r2_score(ts_ys, preds)
			base_pred = np.ones_like(ts_ys) * tr_ys.mean().item()
			base_coef_det = r2_score(ts_ys, base_pred)
			print('Our R^2 = {:.4f} | Naive Mean R^2 = {:.4f}'.format(our_coef_det, base_coef_det))
			our_max_err = max(np.abs(ts_ys - preds))
			base_maxerr = max(np.abs(ts_ys - base_pred))
			print('Our MaxError = {:.4f} | Naive Mean MaxErr = {:.4f}'.format(our_max_err, base_maxerr))
			our_loss = np.sqrt(((preds - ts_ys)**2).mean())
			naive_loss = np.sqrt(((base_pred - ts_ys)**2).mean())
			print('Our Loss = {:.5f} | Naive Mean Loss = {:.5f}'.format(our_loss, naive_loss))
			if self.wandb is not None:
				self.wandb.log({
					'epoch': self.overall_iter,
					'r^2/ours': our_coef_det, 'r^2/naive': base_coef_det,
					'loss/ours': our_loss, 'loss/naive': naive_loss,
					'maxerr/ours': our_max_err, 'maxerr/naive': base_maxerr,
				})

