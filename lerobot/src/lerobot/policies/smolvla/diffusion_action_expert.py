"""
Diffusion Action Expert for SmolVLA

Implements a DDPM-style trajectory diffusion model (in the spirit of Motion Planning Diffusion)
that replaces the Flow Matching head. It operates on full action chunks conditioned on
VLM features and supports gradient-based cost guidance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    """Configuration for the diffusion-based action expert."""

    # Dimensions
    action_dim: int = 7
    hidden_dim: int = 384
    num_heads: int = 8
    num_layers: int = 8

    # Diffusion parameters
    num_diffusion_steps: int = 25
    beta_schedule: str = "cosine"
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # Action chunk parameters
    chunk_size: int = 50

    # Inference
    num_inference_steps: int = 10


class SinusoidalPositionEmbeddings(nn.Module):
    """Standard sinusoidal position embeddings for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffusionTransformerBlock(nn.Module):
    """
    Transformer block with alternating Cross-Attention and Self-Attention
    over action tokens, conditioned on VLM features.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        is_cross_attention: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.is_cross_attention = is_cross_attention

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,  # [B, L, H]
        time_emb: torch.Tensor,  # [B, H]
        context: Optional[torch.Tensor] = None,  # [B, C, H]
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        x = x + time_emb.unsqueeze(1)

        # Attention
        residual = x
        x = self.norm1(x)

        if self.is_cross_attention and context is not None:
            x, _ = self.attn(query=x, key=context, value=context)
        else:
            x, _ = self.attn(query=x, key=x, value=x, attn_mask=causal_mask)

        x = residual + x

        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x


class DiffusionActionExpert(nn.Module):
    """
    Diffusion-based action expert used in place of Flow Matching.

    - Learns to predict Gaussian noise on action chunks.
    - Supports MPD-style gradient guidance during sampling.
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config

        # Projections
        self.action_proj = nn.Linear(config.action_dim, config.hidden_dim)
        self.context_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(config.num_layers):
            is_cross = i % 2 == 0
            self.blocks.append(
                DiffusionTransformerBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    is_cross_attention=is_cross,
                )
            )

        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.output_proj = nn.Linear(config.hidden_dim, config.action_dim)

        self._setup_diffusion_schedule()

    # ------------------------------------------------------------------ #
    # Diffusion utilities
    # ------------------------------------------------------------------ #
    def _setup_diffusion_schedule(self) -> None:
        """Initialise beta / alpha schedules."""
        T = self.config.num_diffusion_steps

        if self.config.beta_schedule == "linear":
            betas = torch.linspace(self.config.beta_start, self.config.beta_end, T)
        elif self.config.beta_schedule == "cosine":
            steps = torch.arange(T + 1)
            alpha_bar = torch.cos(((steps / T) + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {self.config.beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    @staticmethod
    def _create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    # ------------------------------------------------------------------ #
    # Core network
    # ------------------------------------------------------------------ #
    def forward(
        self,
        noisy_actions: torch.Tensor,  # [B, T, D]
        timesteps: torch.Tensor,  # [B]
        vlm_features: torch.Tensor,  # [B, S, H]
    ) -> torch.Tensor:
        """Predict noise for given noisy actions and timestep."""
        batch_size, chunk_size, _ = noisy_actions.shape
        device = noisy_actions.device

        x = self.action_proj(noisy_actions)
        context = self.context_proj(vlm_features)

        time_emb = self.time_embed(timesteps)
        causal_mask = self._create_causal_mask(chunk_size, device)

        for block in self.blocks:
            if block.is_cross_attention:
                x = block(x, time_emb, context=context)
            else:
                x = block(x, time_emb, causal_mask=causal_mask)

        x = self.output_norm(x)
        predicted_noise = self.output_proj(x)
        return predicted_noise

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[
            t
        ][:, None, None]

        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
        return x_t, noise

    def compute_loss(
        self,
        actions: torch.Tensor,  # [B, T, D]
        vlm_features: torch.Tensor,
        actions_is_pad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard DDPM MSE loss between predicted and true noise."""
        batch_size = actions.shape[0]
        device = actions.device

        t = torch.randint(
            0, self.config.num_diffusion_steps, (batch_size,), device=device
        ).long()

        noise = torch.randn_like(actions)
        noisy_actions, _ = self.q_sample(actions, t, noise)

        predicted_noise = self.forward(noisy_actions, t, vlm_features)

        loss = F.mse_loss(predicted_noise, noise, reduction="none")

        if actions_is_pad is not None:
            mask = ~actions_is_pad.unsqueeze(-1)
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()

        return loss

    @torch.no_grad()
    def sample(
        self,
        vlm_features: torch.Tensor,
        batch_size: int = 1,
        guidance_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        guidance_scale: float = 1.0,
        use_ddim: bool = False,
    ) -> torch.Tensor:
        """
        Reverse diffusion sampling with optional MPD-style gradient guidance.

        By default uses the standard DDPM sampler. When `use_ddim=True`, a
        deterministic DDIM sampler with fewer steps can be used.

        NOTE: gradient-based guidance requires calling this method outside
        of `torch.inference_mode()`. In the standard LeRobot inference path
        we pass `guidance_fn=None` to avoid autograd inside inference_mode.
        """
        device = vlm_features.device
        chunk_size = self.config.chunk_size
        action_dim = self.config.action_dim

        # Choose timesteps (supporting optional DDIM-like sparse schedule)
        if use_ddim and self.config.num_inference_steps < self.config.num_diffusion_steps:
            step_ratio = max(
                1, self.config.num_diffusion_steps // self.config.num_inference_steps
            )
            timesteps = list(range(0, self.config.num_diffusion_steps, step_ratio))[::-1]
        else:
            timesteps = list(range(self.config.num_diffusion_steps))[::-1]

        # Start from pure noise
        x = torch.randn(batch_size, chunk_size, action_dim, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            predicted_noise = self.forward(x, t_batch, vlm_features)

            # Optional MPD-style guidance.
            if guidance_fn is not None and t > 0:
                # Guidance is only valid when autograd is enabled.
                # This is expected to be called outside inference_mode.
                alpha_cumprod = self.alphas_cumprod[t]
                with torch.enable_grad():
                    x_0_pred = (
                        x
                        - torch.sqrt(1 - alpha_cumprod) * predicted_noise
                    ) / torch.sqrt(alpha_cumprod)
                    x_0_pred = x_0_pred.detach().clone().requires_grad_(True)
                    cost = guidance_fn(x_0_pred)
                    grad = torch.autograd.grad(cost.sum(), x_0_pred)[0]
                predicted_noise = predicted_noise + guidance_scale * grad

            if use_ddim and i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                x = self._ddim_step(x, predicted_noise, t, t_prev)
            else:
                x = self._ddpm_step(x, predicted_noise, t)

        return x

    def _ddpm_step(self, x: torch.Tensor, predicted_noise: torch.Tensor, t: int) -> torch.Tensor:
        """Single DDPM update step x_t -> x_{t-1}."""
        alpha = self.alphas[t]
        beta = self.betas[t]

        mean = self.sqrt_recip_alphas[t] * (
            x - beta / self.sqrt_one_minus_alphas_cumprod[t] * predicted_noise
        )

        if t > 0:
            noise = torch.randn_like(x)
            variance = self.posterior_variance[t]
            x = mean + torch.sqrt(variance) * noise
        else:
            x = mean
        return x

    def _ddim_step(
        self,
        x: torch.Tensor,
        predicted_noise: torch.Tensor,
        t: int,
        t_prev: int,
    ) -> torch.Tensor:
        """Single deterministic DDIM-like update step."""
        alpha_cumprod_t = self.alphas_cumprod[t]
        if t_prev >= 0:
            alpha_cumprod_t_prev = self.alphas_cumprod[t_prev]
        else:
            alpha_cumprod_t_prev = torch.tensor(1.0, device=x.device, dtype=x.dtype)

        # Predict x_0 and clamp for numerical stability
        x_0_pred = (
            x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise
        ) / torch.sqrt(alpha_cumprod_t)
        x_0_pred = torch.clamp(x_0_pred, -10.0, 10.0)

        # Direction towards x_{t-1}
        dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev) * predicted_noise
        x_prev = torch.sqrt(alpha_cumprod_t_prev) * x_0_pred + dir_xt
        return x_prev


# --------------------------------------------------------------------------- #
# Cost functions for MPD-style guidance
# --------------------------------------------------------------------------- #


class SmoothnessCost(nn.Module):
    """Encourage smooth trajectories via velocity/acceleration/jerk penalties."""

    def __init__(
        self,
        velocity_weight: float = 0.1,
        accel_weight: float = 1.0,
        jerk_weight: float = 10.0,
    ):
        super().__init__()
        self.velocity_weight = velocity_weight
        self.accel_weight = accel_weight
        self.jerk_weight = jerk_weight

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        vel = actions[:, 1:] - actions[:, :-1]
        acc = vel[:, 1:] - vel[:, :-1]
        jerk = acc[:, 1:] - acc[:, :-1]

        cost = (
            self.velocity_weight * (vel**2).mean()
            + self.accel_weight * (acc**2).mean()
            + self.jerk_weight * (jerk**2).mean()
        )
        return cost


class JointLimitsCost(nn.Module):
    """Quadratic penalty for violating joint limits."""

    def __init__(self, joint_min: torch.Tensor, joint_max: torch.Tensor, margin: float = 0.1):
        super().__init__()
        self.register_buffer("joint_min", joint_min)
        self.register_buffer("joint_max", joint_max)
        self.margin = margin

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        lower_violation = F.relu(self.joint_min + self.margin - actions)
        upper_violation = F.relu(actions - self.joint_max + self.margin)
        return (lower_violation**2 + upper_violation**2).mean()


class CombinedGuidance:
    """Combine multiple cost functions into a single scalar cost."""

    def __init__(self, costs: dict[str, tuple[nn.Module, float]]):
        self.costs = costs

    def __call__(self, actions: torch.Tensor) -> torch.Tensor:
        total = 0.0
        for _, (cost_fn, weight) in self.costs.items():
            total = total + weight * cost_fn(actions)
        return total


