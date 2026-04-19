"""Compare SAC against random and heuristic baselines, plus reward ablation.

Usage:
    python evaluate_policies.py \
        --checkpoint outputs/seed_42/agent.pt \
        --scaler outputs/seed_42/scaler.pkl \
        --data data/RADIOMICS_PCA_DATA.csv \
        --out outputs/eval/

Produces, in `--out`:
    policy_comparison.png / .pdf
    ablation_study.png / .pdf
    summary.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from scipy import stats

from src.core import (
    Config,
    GliomaTwinEnv,
    SACAgent,
    apply_plot_style,
    device,
    load_agent,
)


apply_plot_style()


# ---------------------------------------------------------------------------
# Policy definitions
# ---------------------------------------------------------------------------


def random_policy(state, rng: np.random.Generator) -> int:
    """Random baseline: uniformly samples actions."""
    return int(rng.integers(0, 4))


def heuristic_policy(state, volume_index: int = 4) -> int:
    """Rule-based heuristic: graduated response keyed off the volume dim."""
    volume = state[volume_index]
    if volume < 1.0:
        return 0
    elif volume < 1.5:
        return 1
    elif volume < 2.0:
        return 2
    return 3


def sac_policy(state, actor) -> int:
    """SAC learned policy (stochastic sample)."""
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist = actor(state_t)
        return int(dist.sample().item())


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_policy(env: GliomaTwinEnv, policy_fn: Callable, n_episodes: int = 1000,
                    verbose: bool = True) -> Dict:
    episode_lengths = []
    episode_rewards = []
    action_log = []
    toxicity_log = []
    volume_log = []

    if verbose:
        print(f"  Evaluating over {n_episodes} episodes...")

    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        episode_actions = []
        episode_toxicity = []
        episode_volumes = []
        t = 0

        for t in range(env.max_months):
            action = policy_fn(state)
            next_state, reward, done, info = env.step(action)
            episode_actions.append(action)
            episode_toxicity.append(info.get('toxicity', 0))
            episode_volumes.append(info.get('volume', 0))
            state = next_state
            total_reward += reward
            if done:
                break

        episode_lengths.append(t + 1)
        episode_rewards.append(total_reward)
        action_log.extend(episode_actions)
        toxicity_log.append(np.mean(episode_toxicity))
        volume_log.append(episode_volumes[-1] if episode_volumes else 0)

    return {
        "mean_survival": float(np.mean(episode_lengths)),
        "std_survival": float(np.std(episode_lengths)),
        "median_survival": float(np.median(episode_lengths)),
        "min_survival": float(np.min(episode_lengths)),
        "max_survival": float(np.max(episode_lengths)),
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "survival_data": episode_lengths,
        "reward_data": episode_rewards,
        "actions": action_log,
        "mean_toxicity": float(np.mean(toxicity_log)),
        "mean_final_volume": float(np.mean(volume_log)),
    }


# ---------------------------------------------------------------------------
# Ablation environment (survival-only reward)
# ---------------------------------------------------------------------------


class AblationEnv(GliomaTwinEnv):
    """Ablated reward: survival only, without the toxicity/volume penalties."""

    def step(self, action: int):
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

        reward = 1.0
        done = False

        if self.state[self.i_toxicity] > 3.0 or self.toxicity > 3.0:
            reward = -10.0
            done = True
        elif self.month >= self.max_months:
            done = True

        return self.state.astype(np.float32), reward, done, {
            'month': self.month,
            'toxicity': self.toxicity,
            'volume': self.state[self.i_volume],
        }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


POLICY_COLORS = {
    "Random": "#95A5A6",
    "Heuristic": "#E67E22",
    "SAC": "#27AE60",
}
ABLATION_COLORS = ["#27AE60", "#E74C3C"]


def plot_policy_comparison(results: Dict[str, Dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Policy Evaluation for Glioma Treatment Optimization',
                 fontweight='bold', fontsize=15, y=1.02)

    policies = list(results.keys())

    # A: Mean survival
    ax1 = axes[0]
    x_pos = np.arange(len(policies))
    means = [results[p]["mean_survival"] for p in policies]
    stds = [results[p]["std_survival"] for p in policies]
    bars = ax1.bar(x_pos, means, yerr=stds, capsize=6,
                   color=[POLICY_COLORS[p] for p in policies],
                   edgecolor='black', linewidth=1.2, alpha=0.85,
                   error_kw={'linewidth': 2, 'elinewidth': 2, 'capthick': 2})
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + std + 0.5,
                 f'{mean:.1f}\u00b1{std:.1f}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')
    ax1.set_ylabel('Survival Time (months)', fontweight='bold', fontsize=12)
    ax1.set_title('A. Mean Survival Performance', fontweight='bold', loc='left',
                  fontsize=13, pad=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(policies, fontweight='medium')
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, max(means) + max(stds) + 3)

    # B: Action distribution
    ax2 = axes[1]
    action_labels = ["Observe", "TMZ", "RT", "Combined"]
    x_actions = np.arange(len(action_labels))
    width = 0.25
    for i, name in enumerate(policies):
        counts = Counter(results[name]["actions"])
        freqs = [counts.get(j, 0) / len(results[name]["actions"]) for j in range(4)]
        offset = (i - 1) * width
        ax2.bar(x_actions + offset, freqs, width, label=name,
                color=POLICY_COLORS[name], alpha=0.85, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Action Probability', fontweight='bold', fontsize=12)
    ax2.set_title('B. Treatment Strategy Distribution', fontweight='bold',
                  loc='left', fontsize=13, pad=10)
    ax2.set_xticks(x_actions)
    ax2.set_xticklabels(action_labels, fontweight='medium')
    ax2.legend(frameon=True, fancybox=True, shadow=True, edgecolor='black',
               loc='upper right', ncol=1)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # C: Survival distribution
    ax3 = axes[2]
    survival_data = [results[p]["survival_data"] for p in policies]
    positions = list(range(1, len(policies) + 1))
    parts = ax3.violinplot(survival_data, positions=positions, widths=0.6,
                           showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(POLICY_COLORS[policies[i]])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    parts['cmeans'].set_color('black'); parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('red'); parts['cmedians'].set_linewidth(2)
    parts['cbars'].set_color('black')
    parts['cmaxes'].set_color('black')
    parts['cmins'].set_color('black')
    ax3.set_ylabel('Survival Time (months)', fontweight='bold', fontsize=12)
    ax3.set_title('C. Survival Distribution', fontweight='bold', loc='left',
                  fontsize=13, pad=10)
    ax3.set_xticks(positions)
    ax3.set_xticklabels(policies, fontweight='medium')
    ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax3.set_axisbelow(True)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=2, label='Mean'),
        Line2D([0], [0], color='red', linewidth=2, label='Median'),
    ]
    ax3.legend(handles=legend_elements, loc='upper left', frameon=True,
               fancybox=True, shadow=True, edgecolor='black')

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'policy_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / 'policy_comparison.pdf', bbox_inches='tight')
    plt.close()


def plot_ablation(results_sac: Dict, ablation_results: Dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Reward Design Ablation Study', fontweight='bold', fontsize=15, y=1.02)

    ablation_labels = ["Full Reward\n(Toxicity Penalty)", "Survival-Only\n(Ablated)"]

    # A: Survival comparison
    ax1 = axes[0]
    means_abl = [results_sac["mean_survival"], ablation_results["mean_survival"]]
    stds_abl = [results_sac["std_survival"], ablation_results["std_survival"]]
    bars = ax1.bar(ablation_labels, means_abl, yerr=stds_abl, capsize=8,
                   color=ABLATION_COLORS, edgecolor='black', linewidth=1.5, alpha=0.85,
                   error_kw={'linewidth': 2.5, 'elinewidth': 2.5, 'capthick': 2.5})
    for bar, mean, std in zip(bars, means_abl, stds_abl):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + std + 0.4,
                 f'{mean:.1f}\u00b1{std:.1f}', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')
    pct_diff = ((means_abl[0] - means_abl[1]) / means_abl[1]) * 100 if means_abl[1] else 0
    mid_height = (means_abl[0] + means_abl[1]) / 2
    ax1.annotate('', xy=(0, means_abl[0]), xytext=(1, means_abl[1]),
                 arrowprops=dict(arrowstyle='<->', lw=2, color='#34495E'))
    ax1.text(0.5, mid_height, f'{abs(pct_diff):.1f}% difference',
             ha='center', fontsize=10, fontweight='bold', color='#2C3E50',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                       edgecolor='#34495E', linewidth=1.5))
    ax1.set_ylabel('Mean Survival (months)', fontweight='bold', fontsize=12)
    ax1.set_title('A. Impact of Reward Design on Survival', fontweight='bold',
                  loc='left', fontsize=13, pad=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, max(means_abl) + max(stds_abl) + 2)

    # B: Action distribution comparison
    ax2 = axes[1]
    action_labels_short = ["Observe", "TMZ", "RT", "Combined"]
    x_actions = np.arange(len(action_labels_short))
    width = 0.35
    counts_full = Counter(results_sac["actions"])
    freqs_full = [counts_full.get(j, 0) / len(results_sac["actions"]) for j in range(4)]
    counts_abl = Counter(ablation_results["actions"])
    freqs_abl = [counts_abl.get(j, 0) / len(ablation_results["actions"]) for j in range(4)]
    ax2.bar(x_actions - width / 2, freqs_full, width, label='Full Reward',
            color=ABLATION_COLORS[0], alpha=0.85, edgecolor='black', linewidth=1.2)
    ax2.bar(x_actions + width / 2, freqs_abl, width, label='Survival-Only',
            color=ABLATION_COLORS[1], alpha=0.85, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Action Probability', fontweight='bold', fontsize=12)
    ax2.set_title('B. Treatment Strategy Comparison', fontweight='bold',
                  loc='left', fontsize=13, pad=10)
    ax2.set_xticks(x_actions)
    ax2.set_xticklabels(action_labels_short, fontweight='medium')
    ax2.legend(frameon=True, fancybox=True, shadow=True, edgecolor='black',
               loc='upper right', ncol=1)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'ablation_study.png', dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / 'ablation_study.pdf', bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    cfg = Config()
    p = argparse.ArgumentParser(
        description="Evaluate SAC against random/heuristic baselines and a reward ablation."
    )
    p.add_argument('--checkpoint', type=Path, required=True,
                   help='Path to a SAC agent checkpoint (.pt) produced by train_sac.py.')
    p.add_argument('--scaler', type=Path, required=True,
                   help='Pickled StandardScaler that matches the checkpoint.')
    p.add_argument('--data', type=Path, default=Path(cfg.DATA_PATH),
                   help='PCA-reduced radiomics CSV used to build the env.')
    p.add_argument('--out', type=Path, default=Path('outputs/eval'),
                   help='Directory to write figures and summary.json into.')
    p.add_argument('--episodes', type=int, default=1000,
                   help='Number of evaluation episodes per policy.')
    p.add_argument('--seed', type=int, default=0,
                   help='RNG seed for the random baseline.')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.data)
    with open(args.scaler, 'rb') as f:
        scaler = pickle.load(f)

    env = GliomaTwinEnv(df, scaler)
    agent: SACAgent = load_agent(args.checkpoint)
    rng = np.random.default_rng(args.seed)

    print("\n" + "=" * 70)
    print("POLICY EVALUATION (Random, Heuristic, SAC)")
    print("=" * 70)

    results: Dict[str, Dict] = {}

    print("\n[1/3] Evaluating Random Policy...")
    results["Random"] = evaluate_policy(env, lambda s: random_policy(s, rng),
                                        n_episodes=args.episodes)

    print("\n[2/3] Evaluating Heuristic Policy...")
    results["Heuristic"] = evaluate_policy(
        env, lambda s: heuristic_policy(s, env.i_volume), n_episodes=args.episodes
    )

    print("\n[3/3] Evaluating SAC Policy...")
    results["SAC"] = evaluate_policy(
        env, lambda s: sac_policy(s, agent.actor), n_episodes=args.episodes
    )

    print("\n" + "=" * 70)
    print("POLICY PERFORMANCE SUMMARY")
    print("=" * 70)
    comparison_df = pd.DataFrame({
        policy: {
            "Mean Survival": f"{data['mean_survival']:.2f} \u00b1 {data['std_survival']:.2f}",
            "Median Survival": f"{data['median_survival']:.1f}",
            "Range": f"[{data['min_survival']:.0f}, {data['max_survival']:.0f}]",
            "Mean Reward": f"{data['mean_reward']:.2f}",
        }
        for policy, data in results.items()
    }).T
    print("\n", comparison_df)

    plot_policy_comparison(results, args.out)

    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 70)
    significance: Dict[str, Dict] = {}
    for p1, p2 in [("SAC", "Random"), ("SAC", "Heuristic")]:
        statistic, p_value = stats.mannwhitneyu(
            results[p1]["survival_data"],
            results[p2]["survival_data"],
            alternative='two-sided',
        )
        sig_level = ("***" if p_value < 0.001
                     else "**" if p_value < 0.01
                     else "*" if p_value < 0.05 else "ns")
        mean_diff = results[p1]["mean_survival"] - results[p2]["mean_survival"]
        significance[f"{p1}_vs_{p2}"] = {
            "mean_diff": float(mean_diff),
            "p_value": float(p_value),
            "significance": sig_level,
        }
        print(f"\n{p1} vs {p2}:")
        print(f"  Mean difference: {mean_diff:.2f} months")
        print(f"  p-value: {p_value:.4f} {sig_level}")

    print("\n" + "=" * 70)
    print("REWARD ABLATION STUDY")
    print("=" * 70)
    env_ablation = AblationEnv(df, scaler)
    print("\nEvaluating SAC policy with ablated reward...")
    ablation_results = evaluate_policy(
        env_ablation, lambda s: sac_policy(s, agent.actor), n_episodes=args.episodes
    )

    plot_ablation(results["SAC"], ablation_results, args.out)

    _, p_ablation = stats.mannwhitneyu(
        results["SAC"]["survival_data"], ablation_results["survival_data"]
    )
    print("\nAblation Study Results:")
    print(f"  Full Reward:   {results['SAC']['mean_survival']:.2f} \u00b1 "
          f"{results['SAC']['std_survival']:.2f} months")
    print(f"  Survival-Only: {ablation_results['mean_survival']:.2f} \u00b1 "
          f"{ablation_results['std_survival']:.2f} months")
    diff = abs(results['SAC']['mean_survival'] - ablation_results['mean_survival'])
    print(f"  Difference:    {diff:.2f} months")
    print(f"  p-value:       {p_ablation:.4f}")

    summary = {
        "policies": {
            name: {k: v for k, v in r.items()
                   if k not in ("survival_data", "reward_data", "actions")}
            for name, r in results.items()
        },
        "ablation": {
            k: v for k, v in ablation_results.items()
            if k not in ("survival_data", "reward_data", "actions")
        },
        "significance": significance,
        "ablation_p_value": float(p_ablation),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Figures and summary written to {args.out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
