"""Multi-seed SAC training entry point.

Usage:
    python train_sac.py --data data/RADIOMICS_PCA_DATA.csv --out outputs/

Each seed produces an `outputs/seed_<n>/` directory containing:
    agent.pt        - SAC weights (loadable with `src.core.load_agent`)
    scaler.pkl      - fitted StandardScaler used to build the env
    metrics.json    - per-episode rewards, lengths, validation metrics
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.core import (
    Config,
    GliomaTwinEnv,
    SACAgent,
    SEED_COLORS,
    apply_plot_style,
    device,
    save_agent,
)


apply_plot_style()


def train_sac(
    cfg: Config,
    data_path: Path,
    seed: int = 42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    df = pd.read_csv(data_path)
    pc_cols = [col for col in df.columns if col.startswith('PC_')]

    train_df = df.sample(frac=0.8, random_state=seed)
    val_df = df.drop(train_df.index)

    scaler = StandardScaler()
    scaler.fit(train_df[pc_cols])

    env_train = GliomaTwinEnv(train_df, scaler, max_months=cfg.MAX_MONTHS)
    env_val = GliomaTwinEnv(val_df, scaler, max_months=cfg.MAX_MONTHS)

    agent = SACAgent(env_train.state_dim, env_train.action_space.n, cfg=cfg)

    train_rewards: List[float] = []
    train_lengths: List[int] = []
    val_metrics: List[Dict] = []
    losses: List[Dict] = []

    print(f"\n{'='*60}")
    print(f"Training SAC Agent (Seed: {seed})")
    print(f"{'='*60}")

    pbar = tqdm(range(cfg.EPISODES), desc=f"Seed {seed}")

    for episode in pbar:
        state = env_train.reset()
        episode_reward = 0.0

        for t in range(cfg.MAX_MONTHS):
            action = agent.select_action(state, deterministic=False)
            next_state, reward, done, _ = env_train.step(action)

            agent.buffer.add(state, action, reward, next_state, done)

            if len(agent.buffer) >= cfg.BATCH_SIZE:
                loss_dict = agent.update()
                if loss_dict:
                    losses.append(loss_dict)

            state = next_state
            episode_reward += reward

            if done:
                break

        train_rewards.append(episode_reward)
        train_lengths.append(t + 1)

        if episode % cfg.LOG_FREQUENCY == 0:
            recent_survival = np.mean(train_lengths[-cfg.LOG_FREQUENCY:])
            pbar.set_postfix({
                'survival': f'{recent_survival:.1f}',
                'alpha': f'{agent.alpha:.3f}',
            })

        if (episode + 1) % cfg.EVAL_FREQUENCY == 0:
            val_survivals = []
            for _ in range(cfg.EVAL_EPISODES):
                state = env_val.reset()
                for t in range(env_val.max_months):
                    action = agent.select_action(state, deterministic=True)
                    state, reward, done, _ = env_val.step(action)
                    if done:
                        break
                val_survivals.append(t + 1)

            val_metrics.append({
                'episode': episode + 1,
                'mean_survival': float(np.mean(val_survivals)),
                'std_survival': float(np.std(val_survivals)),
            })

    print("\nTraining completed")
    print(
        f"   Mean Survival: {np.mean(train_lengths):.2f} \u00b1 "
        f"{np.std(train_lengths):.2f} months"
    )

    return agent, {
        'train_rewards': train_rewards,
        'train_lengths': train_lengths,
        'val_metrics': val_metrics,
        'losses': losses,
        'seed': seed,
    }, env_val, scaler


def smooth_curve(data: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    if len(data) < 10:
        return data
    return gaussian_filter1d(data, sigma=sigma)


def plot_publication_figure(all_results: List[Dict], save_dir: Path) -> None:
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    fig.suptitle(
        'Soft Actor-Critic Training Dynamics for Glioma Treatment Optimization',
        fontweight='bold', fontsize=15, y=0.98,
    )

    # A: Training Survival Curves
    ax1 = fig.add_subplot(gs[0, 0])
    for idx, result in enumerate(all_results):
        seed = result['seed']
        survival = np.array(result['train_lengths'])
        smoothed = smooth_curve(survival, sigma=15.0)
        episodes = np.arange(len(smoothed))
        ax1.plot(episodes, smoothed, label=f'Seed {seed}',
                 color=SEED_COLORS[idx % len(SEED_COLORS)], alpha=0.85, linewidth=2.2)
    ax1.set_xlabel('Training Episode', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Survival Duration (months)', fontweight='bold', fontsize=12)
    ax1.set_title('A. Training Survival Curves', fontweight='bold', loc='left', fontsize=13, pad=10)
    ax1.legend(frameon=True, fancybox=True, shadow=True, loc='best', framealpha=0.95)
    ax1.set_xlim(0, len(all_results[0]['train_lengths']))
    ax1.set_ylim(bottom=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # B: Training Rewards
    ax2 = fig.add_subplot(gs[0, 1])
    for idx, result in enumerate(all_results):
        seed = result['seed']
        rewards = np.array(result['train_rewards'])
        smoothed = smooth_curve(rewards, sigma=15.0)
        episodes = np.arange(len(smoothed))
        ax2.plot(episodes, smoothed, label=f'Seed {seed}',
                 color=SEED_COLORS[idx % len(SEED_COLORS)], alpha=0.85, linewidth=2.2)
    ax2.set_xlabel('Training Episode', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Episode Reward', fontweight='bold', fontsize=12)
    ax2.set_title('B. Training Rewards', fontweight='bold', loc='left', fontsize=13, pad=10)
    ax2.legend(frameon=True, fancybox=True, shadow=True, loc='best', framealpha=0.95)
    ax2.set_xlim(0, len(all_results[0]['train_rewards']))
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.0, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # C: Validation Performance
    ax3 = fig.add_subplot(gs[1, 0])
    for idx, result in enumerate(all_results):
        seed = result['seed']
        val_metrics = result['val_metrics']
        if val_metrics:
            episodes = [m['episode'] for m in val_metrics]
            means = [m['mean_survival'] for m in val_metrics]
            stds = [m['std_survival'] for m in val_metrics]
            color = SEED_COLORS[idx % len(SEED_COLORS)]
            ax3.plot(episodes, means, label=f'Seed {seed}', color=color,
                     linewidth=2.5, marker='o', markersize=5, alpha=0.9)
            ax3.fill_between(episodes,
                             np.array(means) - np.array(stds),
                             np.array(means) + np.array(stds),
                             color=color, alpha=0.15)
    ax3.set_xlabel('Training Episode', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Validation Survival (months)', fontweight='bold', fontsize=12)
    ax3.set_title('C. Validation Performance', fontweight='bold', loc='left', fontsize=13, pad=10)
    ax3.legend(frameon=True, fancybox=True, shadow=True, loc='best', framealpha=0.95)
    ax3.set_ylim(bottom=0)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # D: Entropy Temperature
    ax4 = fig.add_subplot(gs[1, 1])
    for idx, result in enumerate(all_results):
        seed = result['seed']
        losses = result['losses']
        if losses:
            alphas = [loss['alpha'] for loss in losses]
            steps = np.arange(len(alphas))
            smoothed_alpha = (
                smooth_curve(np.array(alphas), sigma=50.0) if len(alphas) > 100 else alphas
            )
            ax4.plot(steps, smoothed_alpha, label=f'Seed {seed}',
                     color=SEED_COLORS[idx % len(SEED_COLORS)], linewidth=2.2, alpha=0.85)
    ax4.set_xlabel('Update Step', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Temperature (\u03b1)', fontweight='bold', fontsize=12)
    ax4.set_title('D. Entropy Temperature Tuning', fontweight='bold', loc='left', fontsize=13, pad=10)
    ax4.legend(frameon=True, fancybox=True, shadow=True, loc='best', framealpha=0.95)
    ax4.set_ylim(bottom=0)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / 'sac_training_results.png', dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'sac_training_results.pdf', dpi=300, bbox_inches='tight')
    print(f"\nFigures saved to {save_dir}")
    plt.close()


def parse_args() -> argparse.Namespace:
    cfg = Config()
    p = argparse.ArgumentParser(description="Train multi-seed SAC on the glioma digital twin.")
    p.add_argument('--data', type=Path, default=Path(cfg.DATA_PATH),
                   help='Path to the PCA-reduced radiomics CSV.')
    p.add_argument('--out', type=Path, default=cfg.SAVE_DIR,
                   help='Directory to write per-seed checkpoints and metrics into.')
    p.add_argument('--episodes', type=int, default=cfg.EPISODES,
                   help='Training episodes per seed.')
    p.add_argument('--seeds', type=int, nargs='+', default=list(cfg.SEEDS),
                   help='Random seeds to train.')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config()
    cfg.DATA_PATH = str(args.data)
    cfg.SAVE_DIR = args.out
    cfg.EPISODES = args.episodes
    cfg.SEEDS = tuple(args.seeds)

    cfg.SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting SAC training")
    print(f"Configuration: {cfg.EPISODES} episodes, {len(cfg.SEEDS)} seeds, device={device}")

    all_results = []

    for seed in cfg.SEEDS:
        agent, results, env_val, scaler = train_sac(cfg, args.data, seed=seed)
        all_results.append(results)

        seed_dir = cfg.SAVE_DIR / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        save_agent(agent, seed_dir / "agent.pt")
        with open(seed_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open(seed_dir / "metrics.json", "w") as f:
            json.dump({
                'seed': results['seed'],
                'train_rewards': results['train_rewards'],
                'train_lengths': results['train_lengths'],
                'val_metrics': results['val_metrics'],
            }, f)

    plot_publication_figure(all_results, cfg.SAVE_DIR)

    print("\nTraining complete")
    print("\nFinal Performance Summary:")
    print("=" * 60)
    for result in all_results:
        seed = result['seed']
        final_survival = np.mean(result['train_lengths'][-100:])
        final_std = np.std(result['train_lengths'][-100:])
        print(f"Seed {seed}: {final_survival:.2f} \u00b1 {final_std:.2f} months")

    overall_mean = np.mean([np.mean(r['train_lengths'][-100:]) for r in all_results])
    overall_std = np.std([np.mean(r['train_lengths'][-100:]) for r in all_results])
    print(f"\nOverall: {overall_mean:.2f} \u00b1 {overall_std:.2f} months")
    print("=" * 60)


if __name__ == "__main__":
    main()
