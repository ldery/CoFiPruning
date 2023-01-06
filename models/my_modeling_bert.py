import logging
import math
from typing import Optional, Tuple, Union
import os
import torch
from .modeling_bert import *

from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.file_utils import hf_bucket_url, cached_path
from utils.cofi_utils import *
logger = logging.getLogger(__name__)


class FastPrunerFeaturizerModel(BertForSequenceClassification):
	def __init__(self, config):
		super().__init__(config)
		self.bert = CoFiBertModel(config)

	@classmethod
	def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
		if os.path.exists(pretrained_model_name_or_path):
			weights = torch.load(os.path.join(pretrained_model_name_or_path, "pytorch_model.bin"), map_location=torch.device("cpu"))
		else:
			archive_file = hf_bucket_url(pretrained_model_name_or_path, filename="pytorch_model.bin") 
			resolved_archive_file = cached_path(archive_file)
			weights = torch.load(resolved_archive_file, map_location="cpu")


		# Convert old format to new format if needed from a PyTorch state_dict
		old_keys = []
		new_keys = []
		for key in weights.keys():
			new_key = None
			if "gamma" in key:
				new_key = key.replace("gamma", "weight")
			if "beta" in key:
				new_key = key.replace("beta", "bias")
			if new_key:
				old_keys.append(key)
				new_keys.append(new_key)
		for old_key, new_key in zip(old_keys, new_keys):
			weights[new_key] = weights.pop(old_key)

		if "config" not in kwargs:
			config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
			config.do_layer_distill = False
		else:
			config = kwargs["config"]

		model = cls(config)

		load_pruned_model(model, weights)
		return model

	def forward(
			self,
			input_ids=None,
			attention_mask=None,
			token_type_ids=None,
			position_ids=None,
			inputs_embeds=None,
			labels=None,
			output_attentions=None,
			output_hidden_states=None,
			return_dict=None,
			head_z=None,
			head_layer_z=None,
			intermediate_z=None,
			mlp_z=None,
			hidden_z=None,
	):

		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			inputs_embeds=inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			head_z=head_z,
			head_layer_z=head_layer_z,
			intermediate_z=intermediate_z,
			mlp_z=mlp_z,
			hidden_z=hidden_z
		) #! [32, 68, 768]

		return outputs

