from collections import defaultdict
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


# 	def generation_step(self, mask_, model_, inputs):
		

	def train(self):
# 		# TODO[ldery] - go from hard-coded value
# 		n_total_masks = 2000
# 		mask_embeddings_map = []

# 		train_dataloader = iter(self.get_train_dataloader())
# 		val_dataloader = iter(self.get_eval_dataloader())
# 		for mask_ in tqdm(range(n_total_masks)):
# 			with torch.no_grad():
# 				# Fix this nasty thing later
# 				try:
# 					tr_inputs = next(train_dataloader)
# 				except:
# 					train_dataloader = iter(self.get_train_dataloader())
# 					tr_inputs = next(train_dataloader)
				
# 				try:
# 					val_inputs = next(val_dataloader)
# 				except:
# 					val_dataloader = iter(self.get_train_dataloader())
# 					val_inputs = next(val_dataloader)

# 				cur_mask = self.gen_random_mask()

# 				self.fill_inputs_with_zs(cur_mask, tr_inputs)
# 				tr_inputs = self._prepare_inputs(tr_inputs)
# 				tr_embeds = self.model(**tr_inputs)['pooler_output']

# 				self.fill_inputs_with_zs(cur_mask, val_inputs)
# 				val_inputs = self._prepare_inputs(val_inputs)
# 				val_embeds = self.model(**val_inputs)['pooler_output']

# 				pair_ = (cur_mask, ((tr_embeds, tr_inputs.get("labels")), (val_embeds, val_inputs.get("labels"))))
# 				mask_embeddings_map.append(pair_)
		
# # 		pdb.set_trace()
# 		# mask_embeddings_map => {mask : # [(tr_x, tr_y), (val_x, val_y)]}
# 		self.best_zs = self.get_best_head_mask(mask_embeddings_map)
		best_heads = [(i_, np.random.choice(12)) for i_ in range(12)]
# 		print(best_heads)
		self.best_zs = self.head_choices_to_masks(best_heads)
		print(self.best_zs)
		pdb.set_trace()
		print('this is a test')
		

	# Start simple with just the heads !!
	def get_best_head_mask(self, mask_embeddings_map):
		# TODO [ldery]
		# Only look @ the heads first. Will generalize later
		delta_perfs = defaultdict(float)
		best_heads = []
		for l_ in range(12): # Hard coded - fix 
			deltas = []
			for h_ in range(12): # Hard coded - fix 
				# separate masks into those with head turned on and those with it turned off
				on_embeds, off_embeds = self.separate_embeddings_for_head(mask_embeddings_map, (l_, h_))
				delta_perf = self.linear_fit_and_evaluate(on_embeds, off_embeds)
				delta_perfs[(l_, h_)] = delta_perf
				deltas.append(delta_perf)
			best_heads.append((l_, np.argmax(deltas)))
		# create a new mask from the keys
		print(best_heads)
		pdb.set_trace()
		return self.head_choices_to_masks(best_heads)

	def separate_embeddings_for_head(self, mask_embeddings_map, head_id):
		on_embs = [[[], []], [[], []]] # [train(x, y), val(x, y)]
		off_embs = deepcopy(on_embs)
		for (mask_, embeds_) in mask_embeddings_map:
			(tr_x, tr_y), (val_x, val_y) = embeds_
			is_off = (mask_['head_z'][head_id[0], :, head_id[1], :, :]).item() == 0
			to_use = off_embs if is_off else on_embs
			to_use[0][0].append(tr_x)
			to_use[0][1].append(tr_y)
			to_use[1][0].append(val_x)
			to_use[1][1].append(val_y)
		return on_embs, off_embs

	def head_choices_to_masks(self, best_head_ids): # TODO[ldery] - clean up and standardized naming
		this_mask = {k: torch.ones_like(v) for k, v in self.base_zs.items()}
# 		this_mask['head_z'] = torch.zeros_like(this_mask['head_z'])
# 		for l, h_id in enumerate(best_head_ids):
# 			this_mask['head_z'][l, :, h_id, :, :] = 1.0
		return this_mask

	def gen_random_mask(self, key='head_z'):
		mask = {}
		# TODO [ldery] - generalize for non-heads
		for k, v in self.base_zs.items():
			if k == key:
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
			print(xs.shape, ys.shape)
			# Run N-GD Steps to learn linear classifier
			linear_model = torch.nn.Sequential(nn.BatchNorm1d(xs.shape[-1]), nn.Linear(xs.shape[-1], self.num_labels))
			linear_model.to(xs.device)
			optim = Adam(linear_model.parameters(), lr=1e-3) # TODO [ldery] - ablate a reasonable learning rate.
# 			print('iterating ... ')
			for j_ in range(10): # TODO[ldery] - ablate the number of steps
				optim.zero_grad()
				logits_ = linear_model(xs).view(-1, self.num_labels)
				loss_ = CrossEntropyLoss()(logits_, ys.view(-1))
				loss_.backward()
				optim.step()
				if j_ % 10 == 0:
					print(j_, loss_.item())
			# Do the evaluation
			linear_model.eval()
			with torch.no_grad():
				xs, ys = test_
				xs, ys = torch.concat(xs), torch.concat(ys)
				preds = linear_model(xs)
			return self.compute_metrics(EvalPrediction(predictions=preds.cpu(), label_ids=ys.cpu()))['accuracy'] # We are assuming we are dealing with classification only for now
		on_result = fit_and_evaluate(tr_on, eval_on)
		off_result = fit_and_evaluate(tr_off, eval_off)
		return on_result - off_result


	def save_model(self, output_dir: Optional[str] = None):
		output_dir = output_dir if output_dir is not None else self.args.output_dir
		torch.save(self.l0_module, os.path.join(output_dir, "l0_module.pt"))

		if self.best_zs is not None:
			zs = self.best_zs
		elif self.l0_module is not None:
			zs = self.l0_module.forward(training=False)
		torch.save(zs, os.path.join(output_dir, "zs.pt"))

		self.model.save_pretrained(output_dir)

	def calculate_layer_distillation_loss(self, teacher_outputs, student_outputs, zs):
		mse_loss = torch.nn.MSELoss(reduction="mean")
		if self.additional_args.do_layer_distill: #! only do layer distill
			mlp_z = None
			head_layer_z = None
			# logger.info(f"zs={zs}")
			if "mlp_z" in zs:
				mlp_z = zs["mlp_z"].detach().cpu()
			if "head_layer_z" in zs:
				head_layer_z = zs["head_layer_z"].detach().cpu()

			teacher_layer_output = teacher_outputs[2][1:] #! hidden states, with a length of 12. Every has a shape of [32, 65, 768]
			student_layer_output = student_outputs[2][1:] 

			# distilliting existing layers
			if self.additional_args.layer_distill_version == 2:
				for layer_num, (t_layer_o, s_layer_o) in enumerate(zip(teacher_layer_output, student_layer_output)):
					s_layer_o = self.model.layer_transformation(s_layer_o)
					l = mse_loss(t_layer_o, s_layer_o)
					if mlp_z is None or mlp_z[layer_num] > 0:
						layer_loss += l

			# distilling layers with a minimal distance
			elif self.additional_args.layer_distill_version > 2:
				l = []
				if self.additional_args.layer_distill_version > 4:
					specified_teacher_layers = [i for i in range(12)]
					if self.additional_args.layer_distill_version ==5:
						specified_teacher_layers = sorted(random.sample(specified_teacher_layers, 4))
					elif self.additional_args.layer_distill_version ==6:
						result_layers_T= []
						skip_window = len(specified_teacher_layers)//4
						for i in range(0, len(specified_teacher_layers), skip_window):
							result_layers_T.append(random.sample(specified_teacher_layers[i:i+skip_window], 1)[0])
						specified_teacher_layers = result_layers_T
					specified_teacher_layers[0] = max(2, specified_teacher_layers[0])
				else:
					specified_teacher_layers = [2, 5, 8, 11]
				# logger.info(f"sampled teacher layers: {specified_teacher_layers}")
				transformed_s_layer_o = [self.model.layer_transformation(
					s_layer_o) for s_layer_o in student_layer_output]
				specified_teacher_layer_reps = [
					teacher_layer_output[i] for i in specified_teacher_layers] #! teacher: 4x[32,113,768]

				device = transformed_s_layer_o[0].device
				for t_layer_o in specified_teacher_layer_reps:
					for i, s_layer_o in enumerate(transformed_s_layer_o): #! student: 12x[32,113,768]
						l.append(mse_loss(t_layer_o, s_layer_o))
				layerwiseloss = torch.stack(l).reshape(
					len(specified_teacher_layer_reps), len(student_layer_output)) #! [4,12]

				existing_layers = None
				if head_layer_z is not None:
					existing_layers = head_layer_z != 0
					existing_layers = existing_layers.to(layerwiseloss.device)

				layer_loss = 0
				#! no ordering restriction specified
				if self.additional_args.layer_distill_version == 3:
					alignment = torch.argmin(layerwiseloss, dim=1)
				#! added the ordering restriction -> to choose the min loss in 4 student layers
				elif self.additional_args.layer_distill_version in (3, 4, 5, 6):
					last_aligned_layer = 12
					alignment = []
					for search_index in range(len(specified_teacher_layers)-1, -1, -1):
						indexes = layerwiseloss[search_index].sort()[1]
						if existing_layers is not None:
							align = indexes[(
								indexes < last_aligned_layer) & existing_layers]
						else:
							align = indexes[indexes < last_aligned_layer]
						if len(align) > 0:
							align = align[0]
						else:
							align = last_aligned_layer
						alignment.append(align)
						last_aligned_layer = align
					alignment.reverse()
					alignment = torch.tensor(alignment).to(device)
				else:
					logger.info(
						f"{self.additional_args.layer_distill_version} version is not specified.")
					sys.exit()

				layerwise = torch.arange(len(specified_teacher_layers)).to(device)
				layer_loss += layerwiseloss[layerwise, alignment].sum() #! layerwise: teacher (specified layers) / alignment: student (min loss layers) / layerwiseloss: [4,12]
				if self.global_step % 100 == 0:
					logger.info(f"v{self.additional_args.layer_distill_version} Global step: {self.global_step}, Alignment: " + str(alignment))
			return layer_loss
		else:
			return None

	def calculate_distillation_loss(self, teacher_outputs, student_outputs, zs):
		layer_loss = self.calculate_layer_distillation_loss(teacher_outputs, student_outputs, zs)
		distill_loss = layer_loss

		ce_distill_loss = F.kl_div(
			input=F.log_softmax(
				student_outputs[1] / self.additional_args.distill_temp, dim=-1), #! logits: [32,3]
			target=F.softmax(
				teacher_outputs[1] / self.additional_args.distill_temp, dim=-1), #! distill_temp: 2.0
			reduction="batchmean") * (self.additional_args.distill_temp ** 2)

		loss = self.additional_args.distill_ce_loss_alpha * ce_distill_loss
		if distill_loss is not None:
			loss += self.additional_args.distill_loss_alpha * distill_loss

		return distill_loss, ce_distill_loss, loss

	def shortens_inputs(self, inputs):
		max_length = inputs["attention_mask"].sum(-1).max().item()
		inputs["input_ids"] = inputs["input_ids"][:, :max_length]
		inputs["attention_mask"] = inputs["attention_mask"][:, :max_length]
		if "token_type_ids" in inputs:
			inputs["token_type_ids"] = inputs["token_type_ids"][:, :max_length]


	def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> List[torch.Tensor]:
		model.train()
		if self.l0_module is not None:
			self.l0_module.train()
		inputs = self._prepare_inputs(inputs)

		distill_loss = None
		distill_ce_loss = None
		if self.teacher_model is not None:
			with torch.no_grad():
				# only retain inputs of certain keys
				teacher_inputs_keys = ["input_ids", "attention_mask", "token_type_ids", "position_ids", "labels",
									   "output_attentions", "output_hidden_states", "return_dict"]
				teacher_inputs = {key: inputs[key]
								  for key in teacher_inputs_keys if key in inputs}
				self.shortens_inputs(teacher_inputs)
				teacher_outputs = self.teacher_model(**teacher_inputs)
			self.shortens_inputs(inputs)
			student_outputs = model(**inputs) #! get the two outputs

			zs = {key: inputs[key] for key in inputs if "_z" in key} #! extract the zs
			distill_loss, distill_ce_loss, loss = self.calculate_distillation_loss(
				teacher_outputs, student_outputs, zs)
		else:
			loss = self.compute_loss(model, inputs)

		lagrangian_loss = None
		if self.start_prune:
			lagrangian_loss, _, _ = \
				self.l0_module.lagrangian_regularization(
					self.global_step - self.prepruning_finetune_steps)
			loss += lagrangian_loss

		if self.args.gradient_accumulation_steps > 1:
			loss = loss / self.args.gradient_accumulation_steps

		loss.backward()

		# wandb.log({"loss": loss.detach(),
		#         "lagrangian_loss": lagrangian_loss.detach() if lagrangian_loss is not None else None,
		#         "distill_layer_loss": distill_loss.detach() if distill_loss is not None else None,
		#         "distill_ce_loss": distill_ce_loss.detach() if distill_ce_loss is not None else None})

		return {"loss": loss.detach(),
				"lagrangian_loss": lagrangian_loss.detach() if lagrangian_loss is not None else None,
				"distill_layer_loss": distill_loss.detach() if distill_loss is not None else None,
				"distill_ce_loss": distill_ce_loss.detach() if distill_ce_loss is not None else None}

	def fill_inputs_with_zs(self, zs, inputs):
		for key in zs:
			inputs[key] = zs[key]
