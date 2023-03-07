from collections import defaultdict
from functools import partial
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from copy import deepcopy
import pickle as pkl
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
from transformers import Trainer
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import Adam, SGD
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.trainer_pt_utils import nested_concat, nested_numpify
from transformers.trainer_utils import (PREFIX_CHECKPOINT_DIR, EvalPrediction,
									EvaluationStrategy, PredictionOutput,
									TrainOutput)
from transformers.utils import logging
from transformers.training_args import TrainingArguments

from args import AdditionalArguments
from utils.cofi_utils import *
from utils.utils import *
from torch.nn import CrossEntropyLoss
import random
from .kernelshap import calculate_result

import wandb
import pdb

logger = logging.get_logger(__name__)

glue_tasks = {"cola": "matthews_correlation",
		  "mnli": "mnli/acc",
		  "mrpc": "accuracy",
		  "sst2": "accuracy",
		  "stsb": "corr",
		  "qqp": "accuracy",
		  "qnli": "accuracy",
		  "rte": "accuracy",
		  "sst2_aug": "accuracy",
		  "rte_aug": "accuracy",
		  "mrpc_aug": "accuracy",
		  "qnli_aug": "accuracy",
		  "stsb_aug": "corr",}

# TransRate Code
def coding_rate(Z, eps=1e-4):
	n, d = Z.shape
	eye = torch.eye(d).to(Z.device)
	(_, rate)= torch.linalg.slogdet((eye +  (1.0/(n * eps)) * (Z.T @ Z)))
	return 0.5 * rate

def transRate(Z, y, eps=1e-4):
	Z = Z - torch.mean(Z, axis=0, keepdims=True)
	RZ = coding_rate(Z)
	RZY = 0
	K = int(y.max() + 1)
	for i in range(K):
		RZY += coding_rate( Z[(y == i ).flatten()])
	return (RZ - RZY/K).item()

class My_Trainer(Trainer):
	def __init__(
			self,
			model: PreTrainedModel = None,
			args: TrainingArguments = None,
			additional_args: AdditionalArguments = None,
			data_collator: Optional[DataCollator] = None,
			train_dataset: Optional[Dataset] = None,
			eval_dataset: Optional[Dataset] = None,
			tokenizer: Optional[PreTrainedTokenizerBase] = None,
			model_init: Callable[[], PreTrainedModel] = None,
			compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
			l0_module=None,
			teacher_model=None,
			**kwargs,
	):
		Trainer.__init__(self, model, args, data_collator, train_dataset,
						 eval_dataset, tokenizer, model_init, compute_metrics, **kwargs)

		self.additional_args = additional_args
		
		self.l0_module = l0_module
		self.prepruning_finetune_steps = 100
		self.start_prune = False

		self.l0_optimizer = None

		self.best_zs = None
		self.nlayers = self.l0_module.num_hidden_layers
		self.nheads_per_layer =  self.l0_module.num_attention_heads
		self.num_labels = model.num_labels
		self.best_random_mask = None

		self.lagrangian_optimizer = None
		self.kshap_info = None

		self.start_saving_best = True if self.additional_args.pruning_type is None else False

		self.teacher_model = teacher_model
		if self.teacher_model is not None:
			self.teacher_model = self.teacher_model.to(self.args.device)
			self.teacher_model.eval()

		log_level = logging.CRITICAL #args.get_process_log_level()
		logging.set_verbosity(log_level)
		logger.setLevel(log_level)
		
		# Setup for our method
		self.arch_comp_keys = ['head_z', 'mlp_z', 'intermediate_z', 'hidden_z']
		self.separate_keys = ['intermediate_z']

		base_zs = self.l0_module.forward(training=False)
		self.base_zs = {}
		for k, v in base_zs.items():
			v_ = v.detach()
			fill_val = 0.5 if k in self.arch_comp_keys else 1.0
			v_.zero_().fill_(fill_val)
			v_.requires_grad = False
			self.base_zs[k] = v_

		self.acceptable_sparsity_delta = self.additional_args.sparsity_epsilon
		self.train_batcher = iter(self.get_train_dataloader())
		self.val_batcher = iter(self.get_eval_dataloader())
		self.fitness_strategy = self.additional_args.fitness_strategy
		self.masks_per_round = self.additional_args.masks_per_round

	def gen_random_mask(self, paired=False):
		mask = {}
		anti_mask = {} if paired else None
		for k, v in self.base_zs.items():
			if k in self.arch_comp_keys:
				mask[k] = torch.bernoulli(v)
				if anti_mask is not None:
					anti_mask[k] = (1.0 - mask[k]) * (v > 0.0) # only generate anti-mask for active nodes
			else:
				mask[k] = torch.ones_like(v)
				if anti_mask is not None:
					anti_mask[k] = torch.ones_like(v)
		return (mask, anti_mask) if paired else (mask, )

	def get_next_batch(self, is_train=True):
		this_batcher = self.train_batcher if is_train else self.val_batcher
		try:
			batch_ = next(this_batcher)
		except:
			if is_train:
				self.train_batcher = iter(self.get_train_dataloader())
				batch_ = next(self.train_batcher)
			else:
				self.val_batcher = iter(self.get_eval_dataloader())
				batch_ = next(self.val_batcher)
		return batch_

	def run_through_model(self, model_mask, inputs):
		self.fill_inputs_with_zs(model_mask, inputs)
		inputs = self._prepare_inputs(inputs)
		embeds = self.model(**inputs)['pooler_output']
		return embeds, inputs.get("labels")
	
	def get_mask_perf(self, cur_mask, fitness_strategy=None):
		fitness_strategy = fitness_strategy if fitness_strategy else self.fitness_strategy
		if fitness_strategy == 'linear_fit':
			with torch.no_grad():
				# Get a batch of data
				tr_inputs = self.get_next_batch(is_train=True)
				val_inputs = self.get_next_batch(is_train=False)
				tr_outputs = self.run_through_model(cur_mask, tr_inputs)
				val_outputs = self.run_through_model(cur_mask, val_inputs)
			return self.linear_fit_and_evaluate(tr_outputs, val_outputs)
		elif fitness_strategy == 'embedding_cosine':
			assert self.teacher_model is not None, 'To use this strategy the teacher must be available'
			with torch.no_grad():
				tr_inputs = self.get_next_batch(is_train=True)
				original_embeds = self.teacher_model(** self._prepare_inputs(tr_inputs))['pooler_output']
				masked_embeds = self.run_through_model(cur_mask, tr_inputs)[0]
				cos_scores = nn.CosineSimilarity(dim=-1, eps=1e-6)(masked_embeds, original_embeds) + 1.0
			return cos_scores.mean().item()
		elif fitness_strategy == 'transRate':
			with torch.no_grad():
				# Get a batch of data
				tr_inputs = self.get_next_batch(is_train=True)
				val_inputs = self.get_next_batch(is_train=False)

				tr_outputs = self.run_through_model(cur_mask, tr_inputs)
				val_outputs = self.run_through_model(cur_mask, val_inputs)

				Z, y = torch.concat([tr_outputs[0], val_outputs[0]]), torch.concat([tr_outputs[1], val_outputs[1]])
				transrate = transRate(Z, y)
				return transrate
		else: 
			raise ValueError('Incorrect fitness strategy specified : ', fitness_strategy)

	def normalize_scores(self, scores, type_=None):
		mean_, std_ = np.nanmean(scores), np.nanstd(scores)
		normalized_scores = (scores - mean_) / (std_ + 1e-8) # epsilon
		normalized_scores = np.nanmean(normalized_scores, axis=0)
		if type_ in ['head_z', 'intermediate_z']:
			normalized_scores = normalized_scores[:, :, 0] - normalized_scores[:, :, 1]
		else:
			normalized_scores = normalized_scores[:, 0] - normalized_scores[:, 1]
		return normalized_scores

	def reset_base_zs_local(self, kshap_info):
		max_min_delta = ( self.num_labels - 1.0) / self.num_labels
		shapley_values = calculate_result(kshap_info[0], kshap_info[1], max_min_delta)

		occupancies = {}
		pointer = 0

		for k in self.arch_comp_keys:
			active_nodes = self.base_zs[k] > 0
			num_active = active_nodes.sum()
			
			shapley_subset = shapley_values[pointer : pointer + num_active]
			threshold = np.quantile(shapley_subset, self.additional_args.quantile_cutoff)
			shapley_bool = shapley_subset > threshold
			
			self.base_zs[k][active_nodes] = torch.FloatTensor(shapley_bool / 2.0)
			occupancies[k] = (self.base_zs[k] * 2.0).mean().item() # Because we have 0.5 as the bernoulli probability
			pointer += num_active
		return occupancies

	def reset_base_zs_global(self, kshap_info, module_perfs):
		max_min_delta = ( self.num_labels - 1.0) / self.num_labels
		shapley_values = calculate_result(kshap_info[0], kshap_info[1], max_min_delta)

		threshold = np.quantile(shapley_values, self.additional_args.quantile_cutoff)
		shapley_bool = shapley_values > threshold

		A = kshap_info[0][shapley_bool, :][:, shapley_bool]
		b = kshap_info[1][shapley_bool]
		self.kshap_info = (A, b)

		occupancies = {}
		pointer = 0
		for k in self.arch_comp_keys:
			if k not in self.separate_keys:
				active_nodes = self.base_zs[k] > 0
				num_active = active_nodes.sum()
				self.base_zs[k][active_nodes] = torch.FloatTensor(shapley_bool[pointer : pointer + num_active] / 2.0)
				pointer += num_active
			else:
				# Doing a local selection here
				scores = np.array(module_perfs[k])
				scores[scores < 0] = np.NaN
				scores = np.nanmean(scores, axis=0)
				scores = scores[:, :, 0] - scores[:, :, 1]
				chosen = (scores[self.base_zs[k].squeeze() > 0]).flatten()
				chosen = chosen[~np.isnan(chosen)]
				threshold = np.quantile(chosen, self.additional_args.quantile_cutoff)
				mask = torch.tensor(scores > threshold).float()
				mask = mask.unsqueeze(1).unsqueeze(1)

				self.base_zs[k] *= mask
			occupancies[k] = (self.base_zs[k] * 2.0).mean().item() # Because we have 0.5 as the bernoulli probability
		return occupancies

	def check_and_retrieve_past_runs(self):
		scores_save_path = os.path.join(self.args.output_dir, 'module_perfs.pkl') 
		if os.path.exists(scores_save_path):
			print('Resetting Initial Based Mask')
			scores_dict = pkl.load(open(scores_save_path, 'rb'))
			self.best_random_mask = torch.load(os.path.join(self.args.output_dir, "best_random_mask.pt"))
			self.base_zs = torch.load(os.path.join(self.args.output_dir, "zs.pt"))
			occupancies = {}
			for k, v in self.base_zs.items():
				occupancies[k] = v.mean().item()
				self.base_zs[k] *= 0.5
			return occupancies
		return {k: 1.0 for k in self.arch_comp_keys}

	def mask_to_bool_vec(self, mask):
		this_vec = []
		for k in self.arch_comp_keys:
			if k in self.separate_keys:
				continue
			result = torch.masked_select(mask[k], self.base_zs[k] > 0)
			this_vec.append(result)
		return torch.concat(this_vec)

	def construct_module_scores(self, paired=False, module_perfs=None):
		module_perfs = defaultdict(list) if module_perfs is None else module_perfs
		best_perf_so_far, best_random_mask = -1.0, None

		num_iters = int(self.masks_per_round / 2) if paired else self.masks_per_round
		# For kernel shap estimates
		null_perf = 1.0 / self.num_labels
		boolean_mats, bool_vecs = [None, None], [None, None]
		for idx_ in tqdm(range(num_iters)):
			pair = self.gen_random_mask(paired=paired)
			for p_id, cur_mask in enumerate(pair):
				if self.additional_args.get_random_architecture:
					this_perf = random.uniform(0, 1)
				else:
					this_perf = self.get_mask_perf(cur_mask)

				# convert mask to boolean vector
				this_bool_vec = self.mask_to_bool_vec(cur_mask)

				vec_entry = this_bool_vec * (this_perf - null_perf) / num_iters
				bool_vecs[p_id] = vec_entry + bool_vecs[p_id] if (idx_ != 0) else vec_entry

				this_bool_mat = this_bool_vec.outer(this_bool_vec) / num_iters
				boolean_mats[p_id] = this_bool_mat + boolean_mats[p_id] if (idx_ != 0) else this_bool_mat

				# Do some storage for the special key
				for k in self.separate_keys:
					scores = self.set_scores(cur_mask[k].squeeze().unsqueeze(-1), this_perf)
					module_perfs[k].append(scores)

				if this_perf > best_perf_so_far:
					best_perf_so_far = this_perf
					best_random_mask = cur_mask

		kshap_info = boolean_mats[0].cpu().numpy(), bool_vecs[0].cpu().numpy()
		if paired:
			kshap_info = (kshap_info[0] + boolean_mats[1].cpu().numpy()) / 2.0, (kshap_info[1] + bool_vecs[0].cpu().numpy()) / 2.0,
		return best_perf_so_far, best_random_mask, kshap_info, module_perfs

	def train(self):
		# TODO[ldery] - go from hard-coded value
		mask_embeddings_map = []
		self.model.eval()
		prev_occ_dict = self.check_and_retrieve_past_runs()
		print('We achieved the following loaded occupancies')
		print(' | '.join(['{}-occupancy : {:.3f}'.format(k, v) for k, v in prev_occ_dict.items()]))

		initial_occupancy = calculate_parameters(self.model)
		round_ = 0
		best_perf, module_perfs = -1, None
		while round_ < self.additional_args.max_prune_rounds:
			round_ += 1
			print('Starting Round : ', round_)
			best_perf, best_random_mask, kshap_info, module_perfs = self.construct_module_scores(module_perfs=module_perfs)
			if self.kshap_info is not None:
				A = (self.kshap_info[0]*(round_ - 1) + kshap_info[0]) / round_
				b = (self.kshap_info[1]*(round_ - 1) + kshap_info[1]) / round_
				self.kshap_info = A, b
			else:
				self.kshap_info = kshap_info

			if self.additional_args.do_local_thresholding:
				this_occ_dict = self.reset_base_zs_local(self.kshap_info)
			else:
				this_occ_dict = self.reset_base_zs_global(self.kshap_info, module_perfs)

			print('\t', ' | '.join(['{}-occupancy : {:.3f}'.format(k, v) for k, v in this_occ_dict.items()]))
			assert all([occ <= prev_occ_dict[k] for k, occ in this_occ_dict.items()]), 'Occupancy should not increase over time!'

			# Get the best mask at the moment
			cur_best_mask = deepcopy(self.base_zs)
			for k in self.arch_comp_keys:
				cur_best_mask[k] *= 2.0

			our_perf = self.get_mask_perf(cur_best_mask)
			print('\t', '[{}] Best Random Perf : {:.3f}. Our Perf : {:.3f}'.format(self.fitness_strategy, best_perf, our_perf))
			if self.additional_args.fitness_strategy != 'linear_fit':
				random_perf = self.get_mask_perf(best_random_mask, fitness_strategy='linear_fit') if best_random_mask else None
				our_perf = self.get_mask_perf(cur_best_mask, fitness_strategy='linear_fit')
				print('\t', '[Linear Fit]| Best Random Perf : {:.3f}. Our Perf : {:.3f}'.format(random_perf, our_perf))

			# Caching for saving in case this is the last round
			self.best_random_mask = best_random_mask
			# TODO [ldery] -- need to find a way to persist the information from previous iterations

			self.best_zs = cur_best_mask

			cur_sparsity = self.calculate_model_sparsity(cur_best_mask, initial_occupancy)
			print('\t', 'Current Sparsity Level : {:.3f}'.format(cur_sparsity))
			if abs(cur_sparsity - self.additional_args.target_sparsity) < self.acceptable_sparsity_delta:
				break

			if cur_sparsity > self.additional_args.target_sparsity:
				print('\t', 'We overshot. Post Hoc Fix with Module Perfs Recommended')
				break
			prev_occ_dict = this_occ_dict


	def calculate_model_sparsity(self, mask, initial_occupancy):
		model_cpy = deepcopy(self.model)
		prune_model_with_z(mask, model_cpy)
		param_sparsity = 1.0 - calculate_parameters(model_cpy) / initial_occupancy
		del model_cpy
		return param_sparsity


	def set_scores(self, mask, perf):
		on_scores = (mask * perf) + (mask - 1)
		off_scores = (1 - mask) * perf - mask
		return np.array(torch.cat([on_scores, off_scores], axis=-1))


	def linear_fit_and_evaluate(self, train_, test_):
		xs, ys = train_
		test_x, test_y = test_

		# TODO[ldery] - test out simple linear model with 5e-3
		# Run N-GD Steps to learn linear classifier
		linear_model = torch.nn.Sequential(
			nn.BatchNorm1d(xs.shape[-1]), 
			nn.Linear(xs.shape[-1], self.num_labels)
		)
# 		linear_model = nn.Linear(xs.shape[-1], self.num_labels)
		linear_model.to(xs.device)
		optim = Adam(linear_model.parameters(), lr=2e-2) # TODO [ldery] - ablate a reasonable learning rate.
		max_eval_acc = -1.0
		num_iters = 20 # TODO[ldery] - ablate the number of steps
		for j_ in range(num_iters): 
			optim.zero_grad()
			logits_ = linear_model(xs).view(-1, self.num_labels)
			loss_ = CrossEntropyLoss()(logits_, ys.view(-1))
			loss_.backward()
			optim.step()
			if (j_ % 5 == 0) or (j_ == (num_iters - 1)):
				# Now do an eval step
				linear_model.eval()
				predictions = linear_model(test_x).argmax(axis=-1)
				eval_accuracy = (predictions.eq(test_y) * 1.0).mean().item()
				max_eval_acc = max(max_eval_acc, eval_accuracy)
				linear_model.train()
# 			print(j_, loss_.item(), eval_accuracy, max_eval_acc)
# 		print(max_eval_acc)
# 		print('---'*30)
		return max_eval_acc


	def save_model(self, output_dir: Optional[str] = None):
		output_dir = output_dir if output_dir is not None else self.args.output_dir
		torch.save(self.l0_module, os.path.join(output_dir, "l0_module.pt"))

		if self.best_zs is not None:
			print('Saving the best zs')
			zs = self.best_zs
		elif self.l0_module is not None:
			print('Ordinarily, we shouldnt be here -- we should have saved the best zs')
			zs = self.l0_module.forward(training=False)
		torch.save(zs, os.path.join(output_dir, "zs.pt"))
		if self.best_random_mask is not None:
			torch.save(self.best_random_mask, os.path.join(output_dir, "best_random_mask.pt"))
		self.model.save_pretrained(output_dir)


	def shortens_inputs(self, inputs):
		max_length = inputs["attention_mask"].sum(-1).max().item()
		inputs["input_ids"] = inputs["input_ids"][:, :max_length]
		inputs["attention_mask"] = inputs["attention_mask"][:, :max_length]
		if "token_type_ids" in inputs:
			inputs["token_type_ids"] = inputs["token_type_ids"][:, :max_length]

	def fill_inputs_with_zs(self, zs, inputs):
		for key in zs:
			inputs[key] = zs[key]
