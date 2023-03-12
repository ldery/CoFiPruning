import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import math
from scipy.special import comb
import pdb
from tqdm import tqdm
from utils import *

training_config = {
	'lr' : 1e-3,
	'bsz' : 16,
	'nepochs' : 5,
	'd_in': 5,
	'd_rep': 10,
	'dp_prob': 0.1,
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

def get_score_model(config, num_layers, num_players, model_type):
	if model_type == 'logistic_regression':
		return LogisticReg(num_players=num_players)
	elif model_type == 'non-linear':
		return NonLinearModel(config, num_layers=num_layers)

class NonLinearModel(nn.Module):
	def __init__(self, module_config, num_layers=12):
		super(NonLinearModel, self).__init__()

		# Initialize the embeddings
		input_embed_dim = training_config['d_in']
		self.layer_id_embed = nn.Embedding(num_layers + 2, input_embed_dim, padding_idx=0) # Adding plus 1 because we are considering the model embed as a layer.
		self.module_type_embed = nn.Embedding(len(module_config) + 1, input_embed_dim, padding_idx=0) # + 1 for the padding
		num_modules = sum([v.numel() for k, v in module_config.items()])
		self.module_embed = nn.Embedding(num_modules + 1, input_embed_dim, padding_idx=0) # + 1 for the padding

		self.setup_vectors(module_config, num_layers)

		# Member featurizer
		hidden_sz = training_config['d_rep']
		self.featurizer = nn.Sequential(
			nn.Dropout(training_config['dp_prob']),
			nn.ReLU(),
			nn.Linear(input_embed_dim, 5 * hidden_sz),
			nn.ReLU(),
			nn.Linear(5 * hidden_sz, hidden_sz),
			nn.Dropout(training_config['dp_prob']),
		)
		
		self.predictor = nn.Sequential(
			nn.Linear(hidden_sz, int(hidden_sz / 2)),
			nn.ReLU(),
			nn.Linear(int(hidden_sz / 2), 1)
		)
		
		# Init the predictor last layer to 0.5 as a good init
		for k, v in self.predictor.named_parameters():
			if k == '2.bias':
				with torch.no_grad():
					v.fill_(0.5)

		parameters = [
			{'params': self.layer_id_embed.parameters()},
			{'params': self.module_type_embed.parameters()},
			{'params': self.module_embed.parameters()},
			{'params': self.featurizer.parameters()},
			{'params': self.predictor.parameters()},
		]
		self.optim = Adam(parameters, lr=training_config['lr'])
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
		return layers, m_types, modules


	def forward(self, xs):
		input_embeds =  self.layer_id_embed(xs[0]) + self.module_type_embed(xs[1]) + self.module_embed(xs[2])
		featurized = self.featurizer(input_embeds) # B x S x d
		# Sum across the element axes.
		featurized = featurized.mean(axis=1) # B x d
		predictions = self.predictor(featurized)
		return predictions

	def get_scores(self, base_mask):
		self.eval()
		layers, m_types, modules = self.config_to_model_info(base_mask, use_all=True)
		layers, m_types, modules = layers.cuda(), m_types.cuda(), modules.cuda()
		with torch.no_grad():
			bsz = 2**13
			predictions = []
			n_batches = math.ceil(layers.shape[0] / bsz)
			for b_id in tqdm(range(n_batches)):
				start_, end_ = int(bsz * b_id), int(bsz * (b_id + 1))
				this_layers = layers[start_: end_].view(-1, 1)
				this_mtypes = m_types[start_: end_].view(-1, 1)
				this_modules = modules[start_: end_].view(-1, 1)
				# get the predictions
				this_predictions = self.forward([this_layers.cuda(), this_mtypes.cuda(), this_modules.cuda()])
				predictions.append(this_predictions)

		self.train()
		predictions = torch.concat(predictions)
		stats = predictions.mean().item(), predictions.std().item(), predictions.max().item(), predictions.min().item()
		print("Predictions Stats : Mean {:.7f}, Std {:.7f}, Max {:.7f}, Min {:.7f}".format(*stats))
		return predictions

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

	def update_with_info(self, run_info):
		xs, ys = run_info
		ys = torch.tensor(ys).cuda()
		for epoch_ in range(training_config['nepochs']):
			# generate a permutation
			perm = np.random.permutation(len(ys))
			running_loss_avg = 0.0
			n_batches = math.ceil(len(ys) / training_config['bsz'])
			for batch_id in range(n_batches):
				start_, end_ = int(training_config['bsz'] * batch_id), int(training_config['bsz'] * (batch_id + 1))
				this_xs = [xs[i_] for i_ in perm[start_:end_]]
				# We have to do smart padding here
				this_xs = self.pad_sequence(this_xs)

				this_ys = ys[perm[start_:end_]].view(-1, 1)
				# do the forward pass here
				preds = self.forward(this_xs)

				loss = self.loss_fn(preds, this_ys)
				loss.backward()
# 				print(preds)
# 				print('Grad max : ', torch.max(self.score_tensor.grad), 'Intercept Norm : ',  torch.norm(self.intercept.grad))
				self.optim.step()
				self.optim.zero_grad()
				running_loss_avg += loss.item()
			running_loss_avg /= n_batches
			print('Epoch {} : Loss : {:.5f}'.format(epoch_, running_loss_avg))


			
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
