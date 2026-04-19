"""Shared building blocks for the digital-twin / SAC pipeline.

This module is the single source of truth for:
  * The `Config` dataclass (all hyperparameters and paths).
  * The `GliomaTwinEnv` simulation environment.
  * The SAC agent (Actor, Critic, ReplayBuffer, SACAgent).
  * Plotting style + checkpoint helpers.

Both `train_sac.py` and `evaluate_policies.py` import from here so that
evaluation can be performed against a saved checkpoint without re-training.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Plotting style (single source of truth)
# ---------------------------------------------------------------------------

SEED_COLORS = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161']

PLOT_RC = {
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.6,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'axes.axisbelow': True,
}


def apply_plot_style() -> None:
    plt.rcParams.update(PLOT_RC)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _default_data_path() -> str:
    # Allow overriding via env var so the repo is not tied to any one machine.
    return os.environ.get(
        "BRAIN_TWIN_DATA",
        "data/RADIOMICS_PCA_DATA.csv",
    )


@dataclass
class Config:
    """All training / evaluation hyperparameters in one place."""

    MAX_MONTHS: int = 24
    DATA_PATH: str = field(default_factory=_default_data_path)
    EPISODES: int = 2000
    BATCH_SIZE: int = 64
    BUFFER_SIZE: int = 100_000
    HIDDEN_DIM: int = 128
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.0
    LR_ACTOR: float = 1e-4
    LR_CRITIC: float = 1e-4
    LR_ALPHA: float = 1e-4
    GAMMA: float = 0.99
    TAU: float = 0.01
    GRAD_CLIP: float = 0.5
    INIT_ALPHA: float = 0.2
    MAX_ALPHA: float = 1.0
    TARGET_ENTROPY_SCALE: float = 0.98
    EVAL_FREQUENCY: int = 100
    EVAL_EPISODES: int = 50
    SEEDS: Tuple[int, ...] = (42, 123, 456)
    SAVE_DIR: Path = field(default_factory=lambda: Path("./outputs"))
    LOG_FREQUENCY: int = 10

    # Indices into the PCA-augmented state vector that are treated as the
    # tumour-volume and toxicity-proxy dimensions. They default to the values
    # used in the original prototype but can be overridden if the PCA basis
    # changes.
    VOLUME_INDEX: int = 4
    TOXICITY_PROXY_INDEX: int = 7


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class GliomaTwinEnv(gym.Env):
    """A radiomics-conditioned digital twin of glioma progression."""

    GROWTH_RANGE = (1.02, 1.04)
    CHEMO_EFFICACY = (0.90, 0.95)
    RADIO_EFFICACY = (0.85, 0.92)
    TOXICITY_TOLERANCE = (0.9, 1.1)
    VOLUME_PENALTY = 0.1
    TOXICITY_PENALTY = 0.5
    TERMINAL_PENALTY = -10.0

    def __init__(
        self,
        data_df: pd.DataFrame,
        scaler: StandardScaler,
        max_months: int = Config.MAX_MONTHS,
        volume_index: int = Config.VOLUME_INDEX,
        toxicity_proxy_index: int = Config.TOXICITY_PROXY_INDEX,
    ):
        super().__init__()
        self.df = data_df
        self.scaler = scaler
        self.max_months = max_months
        self.pc_cols = [c for c in self.df.columns if c.startswith('PC_')]
        self.action_space = gym.spaces.Discrete(4)
        self.state_dim = len(self.pc_cols) + 1
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # Named indices: avoids silent breakage if the number of PCA
        # components changes. Bounds-checked once here rather than on every
        # step.
        if not 0 <= volume_index < len(self.pc_cols):
            raise ValueError(
                f"volume_index={volume_index} out of range for "
                f"{len(self.pc_cols)} PCA components"
            )
        if not 0 <= toxicity_proxy_index < len(self.pc_cols):
            raise ValueError(
                f"toxicity_proxy_index={toxicity_proxy_index} out of range "
                f"for {len(self.pc_cols)} PCA components"
            )
        self.i_volume = volume_index
        self.i_toxicity = toxicity_proxy_index

    def reset(self, patient_idx: Optional[int] = None) -> np.ndarray:
        if patient_idx is not None:
            if patient_idx >= len(self.df):
                raise ValueError(
                    f"patient_idx {patient_idx} out of bounds for df of "
                    f"size {len(self.df)}"
                )
            row = self.df.iloc[patient_idx]
        else:
            row = self.df.sample(1).iloc[0]

        pca_features = row[self.pc_cols].values.astype(np.float32)
        self.initial_pca = self.scaler.transform(pca_features.reshape(1, -1)).flatten()

        self.growth = np.random.uniform(*self.GROWTH_RANGE)
        self.chemo = np.random.uniform(*self.CHEMO_EFFICACY)
        self.radio = np.random.uniform(*self.RADIO_EFFICACY)
        self.tolerance = np.random.uniform(*self.TOXICITY_TOLERANCE)

        self.toxicity = 0.0
        self.month = 0
        self.state = np.append(self.initial_pca.copy(), [0.0])
        return self.state.astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.month += 1
        self.state[self.i_volume] *= self.growth

        if action == 1:
            self.state[self.i_volume] *= self.chemo
            self.toxicity += 0.2
        elif action == 2:
            self.state[self.i_volume] *= self.radio
            self.toxicity += 0.3
        elif action == 3:
            self.state[self.i_volume] *= min(self.chemo, self.radio)
            self.toxicity += 0.5

        volume_delta = self.state[self.i_volume] - self.initial_pca[self.i_volume]
        self.state[self.i_toxicity] += 0.05 * volume_delta
        self.state[-1] = float(self.month) / self.max_months

        reward = 1.0 - self.VOLUME_PENALTY * max(0, self.state[self.i_volume] - 1.0)
        reward -= self.TOXICITY_PENALTY * (self.toxicity / self.tolerance)

        done = False
        if self.state[self.i_toxicity] > 3.0 or self.toxicity > 3.0:
            reward = self.TERMINAL_PENALTY
            done = True
        elif self.month >= self.max_months:
            done = True

        return self.state.astype(np.float32), reward, done, {
            'month': self.month,
            'toxicity': self.toxicity,
            'volume': self.state[self.i_volume],
        }


# ---------------------------------------------------------------------------
# SAC: Actor, Critic, Replay Buffer, Agent
# ---------------------------------------------------------------------------


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = Config.HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor):
        logits = self.net(state)
        return torch.distributions.Categorical(logits=logits)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = Config.HIDDEN_DIM):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        action_onehot = nn.functional.one_hot(action, self.action_dim).float()
        x = torch.cat([state, action_onehot], dim=1)
        return self.net(x)


class ReplayBuffer:
    def __init__(self, size: int = Config.BUFFER_SIZE):
        self.buffer = deque(maxlen=size)

    def add(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size: int, device: torch.device):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        s, a, r, s2, d = zip(*[self.buffer[i] for i in indices])
        return (
            torch.FloatTensor(np.array(s)).to(device),
            torch.LongTensor(a).to(device),
            torch.FloatTensor(r).to(device),
            torch.FloatTensor(np.array(s2)).to(device),
            torch.FloatTensor(d).to(device),
        )

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim, self.cfg.HIDDEN_DIM).to(device)
        self.critic1 = Critic(state_dim, action_dim, self.cfg.HIDDEN_DIM).to(device)
        self.critic2 = Critic(state_dim, action_dim, self.cfg.HIDDEN_DIM).to(device)
        self.critic1_target = Critic(state_dim, action_dim, self.cfg.HIDDEN_DIM).to(device)
        self.critic2_target = Critic(state_dim, action_dim, self.cfg.HIDDEN_DIM).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.cfg.LR_ACTOR)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=self.cfg.LR_CRITIC)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=self.cfg.LR_CRITIC)

        self.target_entropy = -np.log(1.0 / action_dim) * self.cfg.TARGET_ENTROPY_SCALE
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.cfg.LR_ALPHA)

        self.buffer = ReplayBuffer(self.cfg.BUFFER_SIZE)

    @property
    def alpha(self) -> float:
        return min(self.log_alpha.exp().item(), self.cfg.MAX_ALPHA)

    def select_action(self, state, deterministic: bool = False) -> int:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist = self.actor(state_t)
            action = dist.probs.argmax().item() if deterministic else dist.sample().item()
        return action

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.cfg.BATCH_SIZE:
            return {}

        s, a, r, s2, d = self.buffer.sample(self.cfg.BATCH_SIZE, device)

        with torch.no_grad():
            next_dist = self.actor(s2)
            next_a = next_dist.sample()
            next_logp = next_dist.log_prob(next_a)
            next_q1 = self.critic1_target(s2, next_a)
            next_q2 = self.critic2_target(s2, next_a)
            next_q = torch.min(next_q1, next_q2)
            target_q = r.unsqueeze(1) + self.cfg.GAMMA * (1 - d.unsqueeze(1)) * (
                next_q - self.alpha * next_logp.unsqueeze(1)
            )

        q1 = self.critic1(s, a)
        critic1_loss = nn.MSELoss()(q1, target_q)
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.cfg.GRAD_CLIP)
        self.critic1_opt.step()

        q2 = self.critic2(s, a)
        critic2_loss = nn.MSELoss()(q2, target_q)
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.cfg.GRAD_CLIP)
        self.critic2_opt.step()

        dist = self.actor(s)
        a_sample = dist.sample()
        logp = dist.log_prob(a_sample)
        q1_pi = self.critic1(s, a_sample)
        q2_pi = self.critic2(s, a_sample)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * logp.unsqueeze(1) - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.GRAD_CLIP)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            self.log_alpha.clamp_(max=np.log(self.cfg.MAX_ALPHA))

        for p, p_targ in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            p_targ.data.copy_(self.cfg.TAU * p.data + (1 - self.cfg.TAU) * p_targ.data)
        for p, p_targ in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            p_targ.data.copy_(self.cfg.TAU * p.data + (1 - self.cfg.TAU) * p_targ.data)

        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha,
            'alpha_loss': alpha_loss.item(),
        }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_agent(agent: SACAgent, path: Path) -> None:
    """Persist all SACAgent tensors to a single .pt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'state_dim': agent.state_dim,
            'action_dim': agent.action_dim,
            'actor': agent.actor.state_dict(),
            'critic1': agent.critic1.state_dict(),
            'critic2': agent.critic2.state_dict(),
            'critic1_target': agent.critic1_target.state_dict(),
            'critic2_target': agent.critic2_target.state_dict(),
            'log_alpha': agent.log_alpha.detach().cpu(),
        },
        path,
    )


def load_agent(path: Path, cfg: Optional[Config] = None) -> SACAgent:
    """Recreate an SACAgent from a checkpoint produced by `save_agent`."""
    ckpt = torch.load(Path(path), map_location=device)
    agent = SACAgent(ckpt['state_dim'], ckpt['action_dim'], cfg=cfg)
    agent.actor.load_state_dict(ckpt['actor'])
    agent.critic1.load_state_dict(ckpt['critic1'])
    agent.critic2.load_state_dict(ckpt['critic2'])
    agent.critic1_target.load_state_dict(ckpt['critic1_target'])
    agent.critic2_target.load_state_dict(ckpt['critic2_target'])
    with torch.no_grad():
        agent.log_alpha.copy_(ckpt['log_alpha'].to(device))
    return agent
