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
	'bsz' : 16,
	'nepochs' : 10, #5
	'l1_weight': 0,
# 	'd_in': 10,
# 	'd_rep': 100,
	'd_in': 3,
	'd_rep': 10,
	'dp_prob': 0.1,
	'patience': 5,
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


# Weight initialization
def weight_init_fn(init_bias):
	def fn(layer):
		if isinstance(layer, nn.Linear):
			layer.weight.data.normal_(std=1e-3)
	# 		kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
			if layer.bias is not None:
				layer.bias.data.fill_(init_bias)
	return fn

class MixedLinearModel(nn.Module):
	def __init__(self, module_config, num_layers, num_players):
		super(MixedLinearModel, self).__init__()
		# Initialize the embeddings
		input_embed_dim = training_config['d_in']
		self.layer_id_embed = nn.Embedding(num_layers + 2, input_embed_dim, padding_idx=0) # Adding plus 1 because we are considering the model embed as a layer.
		self.module_type_embed = nn.Embedding(len(module_config) + 1, input_embed_dim, padding_idx=0) # + 1 for the padding
		num_modules = sum([v.numel() for k, v in module_config.items()])
		self.module_embed = nn.Embedding(num_modules + 1, input_embed_dim, padding_idx=0) # + 1 for the padding

		self.score_tensor = nn.parameter.Parameter(create_tensor((num_players, 1)))
		self.num_players = num_players
		# Member featurizer
		hidden_sz = training_config['d_rep']
		self.featurizer = nn.Sequential(
			nn.ReLU(),
			nn.Linear(input_embed_dim, hidden_sz),
			nn.Dropout(training_config['dp_prob']),
			nn.ReLU(),
		)
		self.featurizer.apply(weight_init_fn(0.0))
		self.joint_predictor = nn.Sequential(
			nn.Linear(hidden_sz, 1),
		)
		self.joint_predictor.apply(weight_init_fn(0.5))
	
	def reset_linear(self):
		del self.score_tensor
		self.score_tensor = nn.parameter.Parameter(create_tensor((self.num_players, 1)))

	def forward(self, xs):
		individual_preds = torch.matmul(xs[-1], self.score_tensor)
		input_embeds =  self.layer_id_embed(xs[0]) + self.module_type_embed(xs[1]) + self.module_embed(xs[2])
		features = self.featurizer(input_embeds)
		pad_mask = (xs[0] > 0).unsqueeze(-1).float()
		features = (features * pad_mask).sum(axis=1) / pad_mask.sum(axis=1)
		joint_preds = self.joint_predictor(features)
# 		# TODO [ldery] -- remove this
		joint_preds = torch.zeros_like(joint_preds)
		return individual_preds , joint_preds
	
	def get_scores(self):
		with torch.no_grad():
			predictions = self.score_tensor.detach()
		stats = predictions.mean().item(), predictions.std().item(), predictions.max().item(), predictions.min().item()
		print("Predictions Stats : Mean {:.7f}, Std {:.7f}, Max {:.7f}, Min {:.7f}".format(*stats))
		return predictions


class ScoreModel(nn.Module):
	def __init__(self, module_config, num_layers=12, num_players=None):
		super(ScoreModel, self).__init__()

		# Do some special init
		self.base_model = MixedLinearModel(module_config, num_layers, num_players)
		self.setup_vectors(module_config, num_layers)

		self.loss_fn = nn.MSELoss()

	def setup_vectors(self, base_mask, num_layers):
		# Do a 1 time computation to set things up
		type_index = 0
		layers, m_types, modules = [], [], []
		for t_id, (k, v) in enumerate(base_mask.items()):

			m_types.append(torch.ones_like(v).view(-1) + t_id)

			l_ids = torch.ones_like(v)
			
			if v.shape[0] == num_layers:
				l_ids = l_ids.view(num_layers, -1)
				l_ids += torch.arange(num_layers).view(-1, 1)

			layers.append(l_ids.view(-1))
			type_index += v.numel()

		self.layers = torch.concat(layers)
		self.layers.requires_grad = False
		self.m_types = torch.concat(m_types)
		self.m_types.requires_grad = False
		self.mods = torch.arange(type_index) + 1
		self.mods.requires_grad = False


	def config_to_model_info(self, config, use_all=False):
		base_vec = []
		for k, v in config.items():
			entry = (v > 0).view(-1)
			if use_all:
				entry = torch.ones_like(v).view(-1) > 0
			base_vec.append(entry)
		base_vec = torch.concat(base_vec)
		layers = torch.masked_select(self.layers, base_vec).long()
		m_types = torch.masked_select(self.m_types, base_vec).long()
		modules = torch.masked_select(self.mods, base_vec).long()
		assert layers.shape == m_types.shape == modules.shape, 'Irregular shape given'
		return layers, m_types, modules, base_vec

	def forward(self, xs):
		return self.base_model.forward(xs)

	def get_scores(self, base_mask):
		return self.base_model.get_scores()

	def pad_sequence(self, xs):
		lens = [len(x[0]) for x in xs]
		max_seq_len = max(lens)
		# instantiate a tensor
		bsz = len(xs)
		new_xs = [torch.zeros((bsz, max_seq_len)).long() for _ in range(len(xs[0]))]
		for i in range(bsz):
			for k in range(len(xs[0])):
				new_xs[k][i, :lens[i]] = xs[i][k]
		return [x.cuda() for x in new_xs]
	
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
			this_xs = [xs[i_][:-1] for i_ in perm[start_:end_]]
			xs_masks = torch.stack([xs[i_][-1] for i_ in perm[start_:end_]]).float().cuda()

			# We have to do smart padding here
			this_xs = self.pad_sequence(this_xs)
			this_ys = ys[perm[start_:end_]].view(-1, 1)
			# do the forward pass here
			this_xs.append(xs_masks)
			individ_preds, joint_preds = self.forward(this_xs)
			preds = individ_preds + joint_preds

			if batch_id == 0:
				for p_ in zip(
					individ_preds[:3].squeeze().detach().cpu().numpy().tolist(), joint_preds[:3].squeeze().detach().cpu().numpy().tolist(),
					preds[:3].squeeze().detach().cpu().numpy().tolist(), this_ys[:3].squeeze().detach().cpu().numpy().tolist()):
					print("Ind {:.3f} | Joint {:.6f} | Pred {:.3f} | True {:.3f}".format(*p_))

			# Do some logging here
			with torch.no_grad():
				errors = (preds - this_ys).abs()
				this_max_error = errors.max().item()
				this_min_error = errors.min().item()
			max_error = max(max_error, this_max_error)
			min_error = min(min_error, this_min_error)

			loss = self.loss_fn(preds, this_ys)
			if is_train:
				loss.backward()
				self.optim.step()
				self.optim.zero_grad()
			running_loss_avg += loss.item()
		running_loss_avg /= n_batches
		if not is_train:
			self.base_model.train()
		return running_loss_avg, max_error, min_error

	def update_with_info(self, run_info):
		# Setup the optimizer
		# reset base model.
# 		self.base_model.reset_linear()
		self.optim = Adam(self.base_model.parameters(), lr=training_config['lr'])

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


class LogisticReg(nn.Module):
	def __init__(self, num_players=10):
		super(LogisticReg, self).__init__()
		self.num_players = num_players
		
		self.score_tensor = create_tensor((num_players, 1))
		self.intercept = create_tensor((1, 1),zero_out=0)

		self.optim = Adam([self.score_tensor, self.intercept], lr=training_config['lr'])
		self.loss_fn = nn.MSELoss() #reduction='none')
		self.reg_fn =  nn.L1Loss() 
		self.l1_weight = 0.0

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




# # Get info about max info
# 		n_active_members, n_layer_members = {}, {}
# 		n_active = 0
# 		for k, v in base_mask.items():
# 			n_layer_members[k] = (v > 0).squeeze().sum(axis=-1)
# 			n_active_members[k] = n_layer_members[k].sum().item()
# 			n_active += n_active_members[k]

# 		layer_ids, module_types, modules = [], [], []
# 		for mask in run_info:
# 			this_layers = torch.zeros((n_active,))
# 			this_mtypes = torch.zeros((n_active,))
# 			this_modules = torch.zeros((n_active,))
# 			start_idx = 0
# 			# Could do a lot less work here.
# 			for t_id, (k, v) in enumerate(mask.items()):
# 				# setup the active module ids
# 				active_mods = torch.arange(v.numel())[v.view(-1) > 0] + 1 # Need to check.
# 				this_modules[start_idx : start_idx + len(active_mods)] = active_mods + # start_idx -- fix ldery

# 				# setup the type_ids
# 				this_mtypes[start_idx : (start_idx + len(active_mods))] = t_id + 1

# 				# setup the layer_ids. This is a bit more complicated
# 				layer_active = v.squeeze().sum(axis=-1)
# 				l_start = 0
# 				for l_id, n_l in enumerate(layer_active):
# 					this_layers[(start_idx + l_start): (start_idx + l_start + n_l)] = l_id + 1
# 					l_start += n_layer_members[k][l_id]

# 				start_idx += n_active_members[k]

# 			layer_ids.append(this_layers)
# 			module_types.append(this_mtypes)
# 			modules.append(this_modules)
