from collections import defaultdict
from functools import partial
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from copy import deepcopy

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

		self.lagrangian_optimizer = None

		self.start_saving_best = True if self.additional_args.pruning_type is None else False

		self.teacher_model = teacher_model
		if self.teacher_model is not None:
			self.teacher_model = self.teacher_model.to(self.args.device)

		log_level = args.get_process_log_level()
		logging.set_verbosity(log_level)
		logger.setLevel(log_level)
		
		# Setup for our method
		base_zs = self.l0_module.forward(training=False)
		self.base_zs = {}
		for k, v in base_zs.items():
			v_ = v.detach()
			v_.zero_().fill_(0.5)
			v_.requires_grad = False
			self.base_zs[k] = v_
		
		self.train_batcher = iter(self.get_train_dataloader())
		self.val_batcher = iter(self.get_eval_dataloader())

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


	def train(self):
		# TODO[ldery] - go from hard-coded value
		n_total_masks = 1 #10000
		mask_embeddings_map = []
		arch_comp_keys = ['head_z', 'mlp_z']

		self.model.eval()
		not_random_ = True
		if not_random_:
			for mask_ in tqdm(range(n_total_masks)):
				with torch.no_grad():
					# Get a batch of data
					tr_inputs = self.get_next_batch(is_train=True)
					val_inputs = self.get_next_batch(is_train=False)
					cur_mask = self.gen_random_mask(arch_comp_keys)

					self.fill_inputs_with_zs(cur_mask, val_inputs)
					val_inputs = self._prepare_inputs(val_inputs)
					val_embeds = self.model(**val_inputs)['pooler_output']

					pair_ = (cur_mask, (self.run_through_model(cur_mask, tr_inputs), self.run_through_model(cur_mask, val_inputs)))
					mask_embeddings_map.append(pair_)

			# mask_embeddings_map => {mask : # [(tr_x, tr_y), (val_x, val_y)]}
			self.best_zs = self.get_best_mask(mask_embeddings_map, arch_comp_keys)
		else:
			self.best_zs = self.gen_random_mask(arch_comp_keys) #self.head_choices_to_masks(best_heads)
			best_heads = [(i_, np.random.choice(12)) for i_ in range(12)]
			self.best_zs['head_z'] = torch.zeros_like(self.best_zs['head_z'])
			self.best_zs['mlp_z'] = torch.zeros_like(self.best_zs['mlp_z'])
			for p_ in best_heads:
				self.best_zs['head_z'][p_[0], :, p_[1], :, :] = 1.0

			for l in np.random.choice(12, 3):
				self.best_zs['mlp_z'][l] = 1.0
			print(self.best_zs)

	def get_best_mask(self, mask_embeddings_map, arch_comp_keys):
		best_mask = None
		for key in arch_comp_keys:
			if key == 'head_z':
				scores = self.get_attnhead_scores(mask_embeddings_map)
				mean_, std_ = np.mean(scores, axis=1, keepdims=True), np.std(scores, axis=1, keepdims=True)
				normalized_scores = (scores - mean_) / (std_ + 1e-8) # epsilon
				normalized_scores = normalized_scores[:, :, 0] - normalized_scores[:, :, 1]
				best_heads = np.argmax(normalized_scores, axis=-1)
			elif key == 'mlp_z':
				scores = self.get_mlp_scores(mask_embeddings_map)
				mean_, std_ = np.mean(scores, axis=1, keepdims=True), np.std(scores, axis=1, keepdims=True)
				normalized_scores = (scores - mean_) / (std_ + 1e-8) # epsilon
				normalized_scores = normalized_scores[:, 0] - normalized_scores[:, 1]
				best_heads = np.argsort(normalized_scores)[:3] # 3 is hardcoded for now
			pdb.set_trace()
			best_mask = self.arch_choices_to_masks(best_heads, mask_to_update=best_mask, key=key)
		print([best_mask[k] for k in arch_comp_keys])
		return best_mask

	def get_mlp_scores(self, mask_embeddings_map):
		def mlp_is_on(mlp_id, mask_dict):
			return (mask_dict['mlp_z'][mlp_id]).item() == 1

		delta_perfs = defaultdict(float)
		mlp_scores = []
		for l_ in range(self.nlayers):
			deltas = []
			on_embeds, off_embeds = self.separate_embeddings(mask_embeddings_map, partial(mlp_is_on, l_))
			delta_perf = self.linear_fit_and_evaluate(on_embeds, off_embeds)
			mlp_scores.append(delta_perf)
		return np.array(mlp_scores)

	def get_attnhead_scores(self, mask_embeddings_map):
		def head_is_on(head_id, mask_dict):
			return (mask_dict['head_z'][head_id[0], :, head_id[1], :, :]).item() == 1

		delta_perfs = defaultdict(float)
		head_scores = []
		for l_ in range(self.nlayers):
			deltas = []
			for h_ in range(self.nheads_per_layer):
				# separate masks into those with head turned on and those with it turned off
				on_embeds, off_embeds = self.separate_embeddings(mask_embeddings_map, partial(head_is_on, (l_, h_)))
				delta_perf = self.linear_fit_and_evaluate(on_embeds, off_embeds)
				deltas.append(delta_perf)
			head_scores.append(deltas)
		return np.array(head_scores)

	def separate_embeddings(self, mask_embeddings_map, check_activated):
		on_embs = [[[], []], [[], []]] # [train(x, y), val(x, y)]
		off_embs = deepcopy(on_embs)
		for (mask_, embeds_) in mask_embeddings_map:
			(tr_x, tr_y), (val_x, val_y) = embeds_
			to_use = on_embs if check_activated(mask_) else off_embs
			to_use[0][0].append(tr_x)
			to_use[0][1].append(tr_y)
			to_use[1][0].append(val_x)
			to_use[1][1].append(val_y)
		return on_embs, off_embs

	def arch_choices_to_masks(self, choices, mask_to_update=None, key='head_z'):
		def update_fn(choices, key, tensor_):
			tensor_.zero_()
			if key == 'head_z':
				for l, h_id in enumerate(choices):
					tensor_[l, :, h_id, :, :] = 1.0
			elif key == 'mlp_z':
				tensor_[choices] = 1.0
			return tensor_

		if mask_to_update is None:
			mask_to_update = {k: torch.ones_like(v) for k, v in self.base_zs.items()}
		mask_to_update[key] = update_fn(choices, key, mask_to_update[key])
		return mask_to_update

	def gen_random_mask(self, arch_comp_keys):
		mask = {}
		for k, v in self.base_zs.items():
			if k in arch_comp_keys:
				assert (v == 0.5).all(), 'There should be equal odds of unit being turned on or off'
				mask[k] = torch.bernoulli(v)
			else:
				mask[k] = torch.ones_like(v)
		return mask

	def linear_fit_and_evaluate(self, on_embeds, off_embeds):
		tr_on, tr_off = on_embeds[0], off_embeds[0]
		eval_on, eval_off = on_embeds[1], off_embeds[1]
		def fit_and_evaluate(train_, test_):
			xs, ys = train_
			xs, ys = torch.concat(xs), torch.concat(ys)

			# Run N-GD Steps to learn linear classifier
			linear_model = torch.nn.Sequential(nn.BatchNorm1d(xs.shape[-1]), nn.Linear(xs.shape[-1], self.num_labels))
			linear_model.to(xs.device)
			optim = Adam(linear_model.parameters(), lr=1e-2) # TODO [ldery] - ablate a reasonable learning rate.
			print('iterating ... ')
			for j_ in range(20): # TODO[ldery] - ablate the number of steps
				optim.zero_grad()
				logits_ = linear_model(xs).view(-1, self.num_labels)
				loss_ = CrossEntropyLoss()(logits_, ys.view(-1))
				loss_.backward()
				optim.step()
# 				if j_ % 9 == 0:
				print(j_, loss_.item())
			# Do the evaluation
			linear_model.eval()
			with torch.no_grad():
				xs, ys = test_
				xs, ys = torch.concat(xs), torch.concat(ys)
				preds = linear_model(xs)
			return self.compute_metrics(EvalPrediction(predictions=preds.cpu(), label_ids=ys.cpu()))['accuracy'] # We are assuming we are dealing with classification only for now

		on_result = fit_and_evaluate(tr_on, eval_on) if len(tr_on[0]) > 0 else -1
		off_result = fit_and_evaluate(tr_off, eval_off) if len(tr_off[0]) > 0 else -1
		print(on_result, off_result)
		return on_result, off_result


	def save_model(self, output_dir: Optional[str] = None):
		output_dir = output_dir if output_dir is not None else self.args.output_dir
		torch.save(self.l0_module, os.path.join(output_dir, "l0_module.pt"))

		if self.best_zs is not None:
			zs = self.best_zs
		elif self.l0_module is not None:
			zs = self.l0_module.forward(training=False)
		torch.save(zs, os.path.join(output_dir, "zs.pt"))

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
