import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import timm
import math
import tokenizers
import torch
import torch.nn as nn
import transformers
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
import random
from vla_modules import DisentangleAdapter, ActionPredictor, AttentionPooling

from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    STOP_INDEX,
    NormalizationType,
)

from .configuration_prismatic import OpenVLAConfig, PrismaticConfig

# Set up logger
logger = logging.getLogger(__name__)

from .modeling_prismatic import OpenVLAForActionPrediction, PrismaticCausalLMOutputWithPast


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, indices):
        """
        indices - [B, L]
        """
        B, L = indices.size()
        indices = indices.view(-1)
        pe = self.pe[indices]
        pe = pe.view(B, L, -1)
        return pe
    

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model: int, nheads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nheads, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, q, k, v, attention_mask):
        if attention_mask is not None:
            raise NotImplementedError("CrossAttentionLayer does not support attention_mask yet.")
        out = q + self.attn(q, k, v, key_padding_mask=attention_mask)[0]
        out = self.ff(self.norm1(out)) + out
        return out
    
class CrossAttention(nn.Module):
    def __init__(self, d_model: int, nheads: int = 8, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, nheads) for _ in range(num_layers)
        ])

    def forward(self, q, k, v, attention_mask):
        for layer in self.layers:
            q = layer(q, k, v, attention_mask)
        return q


class MemoryVLAForActionPrediction(OpenVLAForActionPrediction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def patch_compressor(self):
        self.per_token_compressor = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
        )
        self.cog_token_compressor = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
        )

    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        proprio=None,
        proprio_projector=None,
        noisy_actions=None,
        noisy_action_projector=None,
        diffusion_timestep_embeddings=None,
        use_film: bool = False,
        other_pixel_values: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training

        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None

        # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===
        if input_ids.shape[1] == 1:
            assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
            assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
            assert labels is None, "Unexpected key `labels` provided during cached generation!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Unimodal Forward ===
        elif pixel_values is None:
            assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Multimodal Forward ===
        elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
            assert past_key_values is None, "Unexpected key `past_key_values` provided during multimodal forward!"

            # Get input embeddings (from language model embeddings)
            input_embeddings = self.get_input_embeddings()(input_ids)  # (B, seq_len, D)

            # Extract action masks
            all_actions_mask = self._process_action_masks(labels)

            # Extract the language portion of the input embeddings (i.e. remove the action tokens portion)
            language_embeddings = input_embeddings[~all_actions_mask].reshape(
                input_embeddings.shape[0], -1, input_embeddings.shape[2]
            )  # (B, lang_seq_len, llm_dim)
            if other_pixel_values is not None and getattr(self.config, "invswap_ratio", 1.0) < 1.0:
                assert not use_film
                # selected_future_index = torch.randint(0, other_pixel_values.size(1), (1,)).item()
                other_image_features = self._process_vision_features(other_pixel_values, language_embeddings, use_film, use_disentangle=True)
                if isinstance(other_image_features, tuple):
                    other_static, other_dynamic = other_image_features
            
            # Get visual features
            projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film, use_disentangle=getattr(self.config, "static_ratio", 0.0) > 0.0) # (B, num_images * num_patches, dim)
            projected_patch_embeddings = projected_patch_embeddings.view(pixel_values.size(0) * self.vision_backbone.num_images_in_input, -1, projected_patch_embeddings.size(-1)) # (B, num_images, num_patches, dim)
            assert proprio is None

            assert not isinstance(projected_patch_embeddings, tuple)
            # projected_patch_embeddings = torch.cat([other_image_features, projected_patch_embeddings], dim=1) if other_pixel_values is not None else projected_patch_embeddings # for memory
            # Add proprioceptive state if provided
            projected_patch_embeddings = self._process_proprio_features(
                projected_patch_embeddings, proprio, proprio_projector
            )

            # [Diffusion] Add diffusion timestep embedding if provided
            if diffusion_timestep_embeddings is not None:
                # For simplicity, just append diffusion timestep embedding to the end of projected vision patch tokens
                projected_patch_embeddings = torch.cat(
                    (projected_patch_embeddings, diffusion_timestep_embeddings), dim=1
                )

            # Process action embeddings
            if noisy_actions is not None:
                # Get mask corresponding to all action tokens
                all_actions_mask = self._process_action_masks(labels)

                # Reshape noisy actions into individual action tokens
                # noisy_actions: (B, chunk_len, action_dim) -> (B, chunk_len * action_dim, 1)
                B = noisy_actions.shape[0]
                noisy_actions = noisy_actions.reshape(B, -1).unsqueeze(-1)

                # Project noisy action tokens into language model embedding space
                noisy_action_features = noisy_action_projector(noisy_actions)  # (B, chunk_len * action_dim, llm_dim)

                # Replace embeddings of the action tokens with noisy action embeddings
                input_embeddings = self._replace_input_embeddings(
                    input_embeddings, all_actions_mask, noisy_action_features
                )
            else:
                # Replace the embeddings of the action tokens with zeros
                # (Later on, the positional embeddings will be added to them)
                all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
                input_embeddings = input_embeddings * ~all_actions_mask

            # Build multimodal embeddings & attention mask
            input_embeddings = input_embeddings.repeat_interleave(self.vision_backbone.num_images_in_input, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.vision_backbone.num_images_in_input, dim=0) if attention_mask is not None else None
            multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )
            all_actions_mask = all_actions_mask.repeat_interleave(self.vision_backbone.num_images_in_input, dim=0)
            action_token_mask = torch.cat([all_actions_mask[:, :1], torch.zeros(projected_patch_embeddings.size(0), projected_patch_embeddings.size(1), 1, device=all_actions_mask.device, dtype=all_actions_mask.dtype), all_actions_mask[:, 1:]], dim=1)
            # Dispatch to language model
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=None,
                action_token_mask=action_token_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # get the last token of each sample. use attention mask
            last_hidden_states = language_model_output.hidden_states[-1]  # (B * (num_images), seq_len, llm_dim)
            # gather all action tokens
            action_token_indices = torch.nonzero(all_actions_mask.squeeze(-1), as_tuple=False)  # (num_action_tokens, 2)
            cog_tokens = last_hidden_states[action_token_indices[:,0], action_token_indices[:,1], :].view(last_hidden_states.size(0), -1, last_hidden_states.size(-1)) # (B * N, 56, D)
            cog_tokens = self.cog_token_compressor(cog_tokens)  # (B * N, 56, 128)
            per_tokens = self.per_token_compressor(projected_patch_embeddings)  # (B * N, num_patches, 128)


            
            multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )

            # Build labels for multimodal sequence if needed
            multimodal_labels = self._build_multimodal_labels(labels, projected_patch_embeddings)
            action_token_mask = torch.cat([all_actions_mask[:, :1], torch.zeros(projected_patch_embeddings.size(0), projected_patch_embeddings.size(1), 1, device=all_actions_mask.device, dtype=all_actions_mask.dtype), all_actions_mask[:, 1:]], dim=1)

            # Dispatch to language model
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=multimodal_labels,
                action_token_mask=action_token_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Otherwise =>> Assume Invalid! ===
        elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
            raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output

        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
            static_features=(static, other_static) if "static" in locals() and "other_static" in locals() else None,
        )
