set -x

project_name='GRPO-TOPK'

gsm8k_train_path=data/gsm8k/train.parquet
gsm8k_test_path=data/gsm8k/test.parquet

export WANDB_MODE=offline
export WANDB_API_KEY=1fd190887edce4f569d2a445cea4dde17c1491a3

# set the paths
train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

kl_topk_size=100
SFT_MODEL=$(basename $MODEL_PATH)
loss_mode="ppo-top${kl_topk_size}"
exp_name="${loss_mode}-RL"
CKPTS_DIR=rl/new/${loss_mode}/${exp_name}

MODEL_PATH=/inspire/hdd/project/video-understanding/public/share/models/Qwen2.5-7B-Instruct


max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_type=top-k \
    +actor_rollout_ref.actor.kl_topk_size=${kl_topk_size} \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb","tensorboard"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=2 \
    $@
