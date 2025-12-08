import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
import torch.nn.functional as F
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

    def forward(self, x):
        """
        indices - [B, L, ...]
        """
        B, L = x.size()[:2]
        indices = torch.arange(L, device=x.device)
        pe = self.pe[indices]
        pe = pe.unsqueeze(1)
        return x + pe
    
from .tinglin_transformer import Attention, CrossAttention, MLP


class GateFusion(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        
    def forward(self, Hx, x):
        """
        Hx, x: (B, L, D)
        """
        g = self.gate(torch.cat([Hx, x], dim=-1))  # (B, L, D)
        return g * Hx + (1 - g) * x
    

class MemoryVLAForActionPrediction(OpenVLAForActionPrediction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_compressor()

    def patch_compressor(self):
        per_compressed_dim = 128
        cog_compressed_dim = 1024
        self.nolora_per_token_compressor = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, per_compressed_dim),
        ).to(self.language_model.dtype)
        self.nolora_cog_token_compressor = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, cog_compressed_dim),
        ).to(self.language_model.dtype)
        self.nolora_per_positional_encoding = PositionalEncoding(d_model=per_compressed_dim, max_len=5000).to(self.language_model.dtype)
        self.nolora_cog_positional_encoding = PositionalEncoding(d_model=cog_compressed_dim, max_len=5000).to(self.language_model.dtype)

        self.nolora_per_gate_fusion = GateFusion(d_model=per_compressed_dim).to(self.language_model.dtype)
        self.nolora_cog_gate_fusion = GateFusion(d_model=cog_compressed_dim).to(self.language_model.dtype)
        
        self.nolora_post_cog_attention = Attention(d_model=cog_compressed_dim, n_heads=8).to(self.language_model.dtype)
        self.nolora_cog_per_attention = CrossAttention(d_model=cog_compressed_dim, d_model_q=cog_compressed_dim, d_model_kv=per_compressed_dim, n_heads=8).to(self.language_model.dtype)
        self.nolora_ffn = MLP(in_features=cog_compressed_dim, hidden_features=4096, out_features=4096).to(self.language_model.dtype)


    def _run_diffusion_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        noise,
        action_head,
        projected_patch_embeddings,
        labels,
        attention_mask,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        noisy_action_projector,
        cache=None
    ):
        raise NotImplementedError
        """Run diffusion-based action prediction"""
        # Clone embedding for reuse in each timestep
        orig_projected_patch_embeddings = projected_patch_embeddings.clone()
        curr_noisy_actions = noise

        # Reverse diffusion: Iteratively denoise to generate action prediction
        for t in action_head.noise_scheduler.timesteps:
            # Get diffusion model's noise prediction (conditioned on VLA latent embedding, current noisy action
            # embedding, and diffusion timestep embedding)
            timesteps = torch.Tensor([t]).to(labels.device)
            diffusion_timestep_embeddings = (
                action_head.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
            )  # (B, llm_dim)
            diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

            # [Diffusion] Replace the embeddings of the action tokens with noisy actions
            # (Later on, the positional embeddings will be added to them)

            # For simplicity, append diffusion timestep embedding to the end of projected vision tokens
            projected_patch_embeddings = torch.cat(
                (orig_projected_patch_embeddings, diffusion_timestep_embeddings), dim=1
            )

            # Reshape and project noisy actions into language embedding space
            B = curr_noisy_actions.shape[0]
            orig_curr_noisy_actions_shape = curr_noisy_actions.shape
            curr_noisy_actions = curr_noisy_actions.reshape(B, -1).unsqueeze(-1)
            noisy_action_features = noisy_action_projector(curr_noisy_actions)
            curr_noisy_actions = curr_noisy_actions.reshape(orig_curr_noisy_actions_shape)

            # Replace action token embeddings with noisy action embeddings
            input_embeddings = self._replace_input_embeddings(
                input_embeddings.clone(), all_actions_mask, noisy_action_features
            )

            # Build multimodal embeddings and attention mask
            multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )

            # Forward pass through language model
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=cache,
                inputs_embeds=multimodal_embeddings,
                labels=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # Extract hidden states for action portion of response
            last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
            actions_hidden_states = last_hidden_states[
                :,
                NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                :,
            ]  # (B, act_chunk_len, D)

            # Predict noise and update noisy actions: x_t -> x_{t-1}
            noise_pred = action_head.predict_noise(actions_hidden_states)
            curr_noisy_actions = action_head.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

        curr_noisy_actions = curr_noisy_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        # Return final actions
        return curr_noisy_actions.float().cpu().detach().numpy(), actions_hidden_states

    def _regression_or_discrete_prediction(
        self,
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        NUM_PATCHES,
        NUM_PROMPT_TOKENS,
        action_head=None,
        cache=None
    ):
        """Run L1 regression-based continuous action prediction or discrete action tokens prediction."""
        # Zero out action token embeddings
        B = input_embeddings.size(0)
        all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
        input_embeddings = input_embeddings * ~all_actions_mask

        input_embeddings = input_embeddings.repeat_interleave(self.vision_backbone.num_images_in_input, dim=0)
        attention_mask = attention_mask.repeat_interleave(self.vision_backbone.num_images_in_input, dim=0) if attention_mask is not None else None
        # Build multimodal embeddings and attention mask
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )
        all_actions_mask = all_actions_mask.repeat_interleave(self.vision_backbone.num_images_in_input, dim=0)
        action_token_mask = torch.cat([all_actions_mask[:, :1], torch.zeros(projected_patch_embeddings.size(0), projected_patch_embeddings.size(1), 1, device=all_actions_mask.device, dtype=all_actions_mask.dtype), all_actions_mask[:, 1:]], dim=1)
        # Forward pass through language model
        if cache is not None:
            cache_length = cache[0][0].shape[2]
            multimodal_embeddings = multimodal_embeddings[:, cache_length:]
            action_token_mask = action_token_mask[:, cache_length:]
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=cache,
            inputs_embeds=multimodal_embeddings,
            action_token_mask=action_token_mask,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        # get the last token of each sample. use attention mask
        last_hidden_states = language_model_output.hidden_states[-1]  # (B * (num_images), seq_len, llm_dim)
        # gather all action tokens
        action_token_indices = torch.nonzero(all_actions_mask.squeeze(-1), as_tuple=False)  # (num_action_tokens, 2)
        cog_tokens = last_hidden_states[action_token_indices[:,0], action_token_indices[:,1], :].view(last_hidden_states.size(0), -1, last_hidden_states.size(-1)) # (B * N, 56, D)
        cog_tokens = self.nolora_cog_token_compressor(cog_tokens).view(B, self.vision_backbone.num_images_in_input, 56, -1)  # (B, N, 56, 128)
        per_tokens = self.nolora_per_token_compressor(projected_patch_embeddings).view(B, self.vision_backbone.num_images_in_input, 256, -1)  # (B, N, num_patches, 128)
        
        curr_cog_tokens, past_cog_tokens = cog_tokens[:, -1:], cog_tokens[:, :-1].contiguous()  # (B, 1, 56, 128), (B, N-1, 56, 128)
        curr_per_tokens, past_per_tokens = per_tokens[:, -1:], per_tokens[:, :-1].contiguous()  # (B, 1, num_patches, 128), (B, N-1, num_patches, 128)
        
        H_cog = F.scaled_dot_product_attention(
            query=curr_cog_tokens.view(curr_cog_tokens.size(0), -1, curr_cog_tokens.size(-1)),
            key=self.nolora_cog_positional_encoding(past_cog_tokens).view(past_cog_tokens.size(0), -1, past_cog_tokens.size(-1)),
            value=past_cog_tokens.view(past_cog_tokens.size(0), -1, past_cog_tokens.size(-1)),
        ) # (B, 1 * 56, 128)
        H_per = F.scaled_dot_product_attention(
            query=curr_per_tokens.view(curr_per_tokens.size(0), -1, curr_per_tokens.size(-1)),
            key=self.nolora_per_positional_encoding(past_per_tokens).view(past_per_tokens.size(0), -1, past_per_tokens.size(-1)),
            value=past_per_tokens.view(past_per_tokens.size(0), -1, past_per_tokens.size(-1)),
        ) # (B, 1 * num_patches, 128)
        cog_tokens = self.nolora_cog_gate_fusion(H_cog.view_as(curr_cog_tokens), curr_cog_tokens).squeeze(1) # (B, 56, 128)
        per_tokens = self.nolora_per_gate_fusion(H_per.view_as(curr_per_tokens), curr_per_tokens).squeeze(1) # (B, num_patches, 128)
        
        cog_tokens = self.nolora_post_cog_attention(cog_tokens) + cog_tokens
        cog_tokens = self.nolora_cog_per_attention(cog_tokens, per_tokens, per_tokens) + cog_tokens
        cog_tokens = self.nolora_ffn(cog_tokens)
        
        def fix_hidden_states(hidden_states):
            hidden_states = hidden_states.view(B, self.vision_backbone.num_images_in_input, -1, hidden_states.size(-1))
            hidden_states = hidden_states[:, -1].contiguous()
            return hidden_states
        
        language_model_output.logits = language_model_output.logits.view(B, self.vision_backbone.num_images_in_input, -1, language_model_output.logits.size(-1))[:, -1].contiguous()
        multimodal_labels = self._build_multimodal_labels(labels, projected_patch_embeddings.view(B, self.vision_backbone.num_images_in_input, -1, projected_patch_embeddings.size(-1))[:, -1])
        language_model_output.hidden_states = tuple(fix_hidden_states(h) for h in language_model_output.hidden_states)
        language_model_output.past_key_values = None  # Not supported in this modified forward
        
        if cache is not None:
            NUM_PROMPT_TOKENS = NUM_PROMPT_TOKENS - cache_length

        if action_head is not None:
            # Extract hidden states for action tokens
            last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
            actions_hidden_states = last_hidden_states[
                :,
                NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                :,
            ]  # (B, act_chunk_len, D)

            # L1 regression prediction
            normalized_actions = action_head.predict_action(actions_hidden_states)
            normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
            normalized_actions = normalized_actions.float().cpu().detach().numpy()
            reponse_ids = None
        else:
            NUM_PROMPT_TOKENS = NUM_PROMPT_TOKENS + NUM_PATCHES
            batch_size = language_model_output.logits.shape[0]
            device = language_model_output.logits.device
        
            start_indices = torch.tensor([NUM_PROMPT_TOKENS], device=device).unsqueeze(1)  # [batch_size, 1]
            position_offsets = torch.arange(ACTION_DIM * NUM_ACTIONS_CHUNK, device=device).unsqueeze(0)  # [1, seq_length]
            seq_indices = start_indices + position_offsets  # [batch_size, ACTION_DIM*NUM_ACTIONS_CHUNK]
            #test end
            #test add
            #print("language_model_output",language_model_output.logits.shape[-1])
            #print("self.vocab_size",self.vocab_size) 32000
            #topk_values, topk_indices = torch.topk(language_model_output.logits, k=256, dim=-1)
            #print(topk_indices)
            #assert language_model_output.logits.shape[-1] == self.vocab_size
            #test add
            do_sample = False
            if do_sample == False:
                #org
                # reponse_ids = language_model_output.logits[
                #         :,
                #         NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                #     ].argmax(dim=2)
                #reponse_ids = language_model_output.logits[actions_masks].argmax(dim=2)
                #org end
                
                #padding
                # reponse_ids = language_model_output.logits[
                #     torch.arange(batch_size, device=device).unsqueeze(-1),  
                #     seq_indices, 
                #     :
                # ].argmax(dim=2)  
                #padding end
                
                #padding + only get last 256 token
                reponse_ids_logits = language_model_output.logits[
                    torch.arange(batch_size, device=device).unsqueeze(-1),  
                    seq_indices, 
                    :
                ]
                start_index = self.vocab_size - 256 
                response_last256 = reponse_ids_logits[..., -256-64:-64]  # Shape: [batch_size, seq_len, 256]
                last256_argmax = response_last256.argmax(dim=-1)  # Shape: [batch_size, seq_len]
                reponse_ids = last256_argmax + start_index  # Shape: [batch_size, seq_len]
                #padding + only get last 256 token end
                
                predicted_action_token_ids = reponse_ids.cpu().numpy()
                    
            else:
                assert temperature>0
                #org 
                # action_logits  = language_model_output.logits[
                #         :,
                #         NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
                #     ]
                #action_logits = language_model_output.logits[actions_masks]
                #org end
                
                action_logits = language_model_output.logits[
                    torch.arange(batch_size, device=device).unsqueeze(-1),  
                    seq_indices, 
                    :
                ]  
                # padding 
                # scaled_logits = action_logits / temperature
                # probs = torch.softmax(scaled_logits, dim=-1)
                # probs_flat = probs.reshape(-1, probs.shape[-1])  # (B*act_chunk_len, vocab_size)
                # sampled_indices_flat = torch.multinomial(probs_flat, num_samples=1)  # (B*act_chunk_len, 1)
                # reponse_ids = sampled_indices_flat.view(action_logits.shape[0], -1)
                # padding end 
                
                #padding + only get last 256 token
                action_logits_last256 = action_logits[..., -256-64:-64]
                scaled_logits = action_logits_last256 / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                assert probs.shape[-1] == 256
                probs_flat = probs.reshape(-1, probs.shape[-1])
                sampled_indices_flat = torch.multinomial(probs_flat, num_samples=1)
                original_ids_flat = sampled_indices_flat + (self.vocab_size - 256)
                reponse_ids = original_ids_flat.view(action_logits.shape[0], -1)
                #padding + only get last 256 token end
                
                predicted_action_token_ids = reponse_ids.cpu().numpy()
        
            discretized_actions = self.vocab_size - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
            normalized_actions = self.bin_centers[discretized_actions]
            #normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
            normalized_actions = normalized_actions.reshape(-1, ACTION_DIM)

        return normalized_actions, reponse_ids, language_model_output.past_key_values

    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        proprio=None,
        proprio_projector=None,
        action_head=None,
        noisy_action_projector=None,
        use_film: bool = False,
        cache = None,
        **kwargs: str,
    ) -> np.ndarray:
        """Predict actions from input sequence, with options for different prediction methods.

        Args:
            input_ids: Input token ids
            unnorm_key: Key for unnormalization statistics
            proprio: Proprioceptive features
            proprio_projector: Projector for proprioceptive features
            action_head: Optional head for L1 regression or diffusion-based prediction
            noisy_action_projector: Projector for noisy actions in diffusion-based prediction
            use_film: Whether to use FiLM conditioning
            **kwargs: Additional arguments including pixel_values and attention_mask

        Returns:
            Tuple of (unnormalized_actions, action_hidden_states)
        """
        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        pixel_values = kwargs["pixel_values"]
        attention_mask = kwargs["attention_mask"]
        # Create fake labels tensor (needed for action mask)
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        # Get number of tokens in prompt (excluding the start token)
        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1  # Subtract action tokens and stop token

        # Prepare inputs by adding necessary tokens
        input_ids, attention_mask = self._prepare_input_for_action_prediction(input_ids, attention_mask)

        # Update labels tensor for action mask computation later
        labels = self._prepare_labels_for_action_prediction(labels, input_ids)

        # Get input embeddings and action masks
        input_embeddings = self.get_input_embeddings()(input_ids)
        all_actions_mask = self._process_action_masks(labels)

        # Extract language embeddings
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )

        # Process vision features
        projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film, use_disentangle=False)
        projected_patch_embeddings = projected_patch_embeddings.view(pixel_values.size(0) * self.vision_backbone.num_images_in_input, -1, projected_patch_embeddings.size(-1)) # (B, num_images, num_patches, dim)

        # Add proprioceptive features if provided
        use_proprio = proprio_projector is not None and proprio is not None
        if use_proprio:
            proprio = torch.Tensor(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
            projected_patch_embeddings = self._process_proprio_features(
                projected_patch_embeddings, proprio, proprio_projector
            )

        # Use diffusion if provided, otherwise use regression or discrete prediction
        use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")

        # Calculate number of patches (including proprio token and/or diffusion timestep embedding if present)
        NUM_PATCHES = self.vision_backbone.get_num_patches() 
        if use_proprio:
            NUM_PATCHES += 1
        if use_diffusion:
            NUM_PATCHES += 1

        if use_diffusion:
            # Sample random noise with shape equal to output action, used as the starting state for reverse diffusion
            noise = torch.randn(
                size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM), device=input_embeddings.device, dtype=input_embeddings.dtype
            )

            # Run diffusion-based prediction
            normalized_actions, actions_hidden_states = self._run_diffusion_prediction(
                input_embeddings,
                all_actions_mask,
                noise,
                action_head,
                projected_patch_embeddings,
                labels,
                attention_mask,
                NUM_PATCHES,
                NUM_PROMPT_TOKENS,
                noisy_action_projector,
                cache=cache,
            )
        else:
            # Run regression or discrete token-based prediction
            normalized_actions, actions_hidden_states, past_key_values = self._regression_or_discrete_prediction(
                input_embeddings,
                all_actions_mask,
                projected_patch_embeddings,
                attention_mask,
                labels,
                NUM_PATCHES,
                NUM_PROMPT_TOKENS,
                action_head,
                cache=cache,
            )

        # Unnormalize predicted actions
        actions = self._unnormalize_actions(normalized_actions, unnorm_key)

        return actions, actions_hidden_states, None

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
            
            # Get visual features
            projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film, use_disentangle=False)
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
            cog_tokens = self.nolora_cog_token_compressor(cog_tokens).view(input_ids.size(0), self.vision_backbone.num_images_in_input, 56, -1)  # (B, N, 56, 128)
            per_tokens = self.nolora_per_token_compressor(projected_patch_embeddings).view(input_ids.size(0), self.vision_backbone.num_images_in_input, 256, -1)  # (B, N, num_patches, 128)
            
            curr_cog_tokens, past_cog_tokens = cog_tokens[:, -1:], cog_tokens[:, :-1].contiguous()  # (B, 1, 56, 128), (B, N-1, 56, 128)
            curr_per_tokens, past_per_tokens = per_tokens[:, -1:], per_tokens[:, :-1].contiguous()  # (B, 1, num_patches, 128), (B, N-1, num_patches, 128)
            
            H_cog = F.scaled_dot_product_attention(
                query=curr_cog_tokens.view(curr_cog_tokens.size(0), -1, curr_cog_tokens.size(-1)),
                key=self.nolora_cog_positional_encoding(past_cog_tokens).view(past_cog_tokens.size(0), -1, past_cog_tokens.size(-1)),
                value=past_cog_tokens.view(past_cog_tokens.size(0), -1, past_cog_tokens.size(-1)),
            ) # (B, 1 * 56, 128)
            H_per = F.scaled_dot_product_attention(
                query=curr_per_tokens.view(curr_per_tokens.size(0), -1, curr_per_tokens.size(-1)),
                key=self.nolora_per_positional_encoding(past_per_tokens).view(past_per_tokens.size(0), -1, past_per_tokens.size(-1)),
                value=past_per_tokens.view(past_per_tokens.size(0), -1, past_per_tokens.size(-1)),
            ) # (B, 1 * num_patches, 128)
            cog_tokens = self.nolora_cog_gate_fusion(H_cog.view_as(curr_cog_tokens), curr_cog_tokens).squeeze(1) # (B, 56, 128)
            per_tokens = self.nolora_per_gate_fusion(H_per.view_as(curr_per_tokens), curr_per_tokens).squeeze(1) # (B, num_patches, 128)
            
            cog_tokens = self.nolora_post_cog_attention(cog_tokens) + cog_tokens
            cog_tokens = self.nolora_cog_per_attention(cog_tokens, per_tokens, per_tokens) + cog_tokens
            cog_tokens = self.nolora_ffn(cog_tokens)
            
            def fix_hidden_states(hidden_states):
                hidden_states = hidden_states.view(input_ids.size(0), self.vision_backbone.num_images_in_input, -1, hidden_states.size(-1))
                hidden_states = hidden_states[:, -1].contiguous()
                return hidden_states
            
            language_model_output.logits = language_model_output.logits.view(input_ids.size(0), self.vision_backbone.num_images_in_input, -1, language_model_output.logits.size(-1))[:, -1].contiguous()
            multimodal_labels = self._build_multimodal_labels(labels, projected_patch_embeddings.view(input_ids.size(0), self.vision_backbone.num_images_in_input, -1, projected_patch_embeddings.size(-1))[:, -1])
            
            loss = self.loss_function(logits=language_model_output.logits, labels=multimodal_labels, vocab_size=self.language_model.config.vocab_size)
            language_model_output.loss = loss
            language_model_output.hidden_states = tuple(fix_hidden_states(h) for h in language_model_output.hidden_states)
            language_model_output.past_key_values = None
            
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
