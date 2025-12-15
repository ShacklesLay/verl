# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = ["AlgoConfig", "FilterGroupsConfig", "KLControlConfig"]


@dataclass
class KLControlConfig(BaseConfig):
    """Configuration for KL control.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        type (str): Type of KL control. Can be "fixed" or "adaptive".
        kl_coef (float): Initial coefficient for KL penalty.
        horizon (int): Horizon value for adaptive controller.
        target_kl (float): Target KL divergence for adaptive controller.
    """

    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1


@dataclass
class FilterGroupsConfig(BaseConfig):
    """Configuration for filter groups (used in DAPO and Entropy).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        enable (bool): Whether to enable filter groups.
        metric (Optional[str]): Metric to use for filtering: "acc", "score", "seq_reward", "seq_final_reward", etc.
        max_num_gen_batches (int): Non-positive values mean no upper limit.
    """

    enable: bool = False
    metric: Optional[str] = None
    max_num_gen_batches: int = 0


@dataclass
class AlgoConfig(BaseConfig):
    """Configuration for the algorithm.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        gamma (float): Discount factor for future rewards.
        lam (float): Trade-off between bias and variance in the GAE estimator.
        adv_estimator (str): Advantage estimator type: "gae", "grpo", "reinforce_plus_plus", etc.
        norm_adv_by_std_in_grpo (bool): Whether to normalize advantages by std (specific to GRPO).
        use_kl_in_reward (bool): Whether to enable in-reward KL penalty.
        kl_penalty (str): How to estimate KL divergence: "kl"/"k1", "abs", "mse"/"k2", "low_var_kl"/"k3", "full", "top-k", or "top-k-unnorm".
        kl_topk_size (Optional[int]): Number of top tokens to consider for top-k and top-k-unnorm KL computation.
            Required when using kl_penalty="top-k" or "top-k-unnorm".
        kl_ctrl (KLControlConfig): KL control configuration.
        use_pf_ppo (bool): Whether to enable preference feedback PPO.
        pf_ppo (dict[str, Any]): Preference feedback PPO settings.
        filter_groups (Optional[FilterGroupsConfig]): Filter groups configuration, used in DAPO and Entropy
        rollout_is_threshold (Optional[float]): Upper threshold for IS weights. null = disabled,
            float value = enabled (compute weights and metrics). This is the main on/off switch.
        rollout_is_threshold_lower (Optional[float]): Lower threshold for IS weights. If None, defaults to 1/upper.
        rollout_is_level (str): Aggregation level: "token", "sequence", or "geometric".
        rollout_is_mode (str): Bounding mode: "truncate" (cap upper only) or "mask" (zero outside bounds).
        rollout_is_veto_threshold (float or None): Per-token veto threshold for catastrophic outliers. None to disable.
        rollout_is (bool): Whether to apply IS weights to policy loss. True = apply weights,
            False = compute metrics only (useful for monitoring before enabling correction). Default: False.
    """

    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_topk_size: Optional[int] = None
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)
    use_pf_ppo: bool = False
    pf_ppo: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate that kl_topk_size is set when using top-k KL penalty variants
        if self.kl_penalty in ("top-k", "top-k-unnorm"):
            if self.kl_topk_size is None or self.kl_topk_size <= 0:
                raise ValueError(
                    f"kl_penalty='{self.kl_penalty}' requires kl_topk_size to be set to a positive integer. "
                    f"Got kl_topk_size={self.kl_topk_size}"
                )
    filter_groups: Optional[FilterGroupsConfig] = None
    # Rollout Importance Sampling
    # Controls computation of IS weights and mismatch metrics
    rollout_is_threshold: Optional[float] = None  # null = disabled, float = enabled
    rollout_is_threshold_lower: Optional[float] = None
    rollout_is_level: str = "token"
    rollout_is_mode: str = "truncate"
    rollout_is_veto_threshold: Optional[float] = None
    # Controls whether to apply IS weights to policy loss (only if rollout_is_threshold is set)
    # True = apply weights to loss, False = compute metrics only (no weight application)
    rollout_is: bool = False
