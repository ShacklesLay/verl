MODEL_PATH=/inspire/hdd/project/video-understanding/public/share/models/Qwen2.5-7B-Instruct

export WANDB_MODE=offline
export WANDB_API_KEY=1fd190887edce4f569d2a445cea4dde17c1491a3

SFT_MODEL=$(basename $MODEL_PATH)
loss_mode=ppo
exp_name="${loss_mode}-epslow-${clip_ratio_low}-epshigh-${clip_ratio_high}-${SFT_MODEL}-RL"
CKPTS_DIR=rl/new/${loss_mode}/${exp_name}

max_prompt_length=$((1024 * 2))

gsm8k_train_path=data/gsm8k/train.parquet
gsm8k_test_path=data/gsm8k/test.parquet

# set the paths
train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${MODEL_PATH}\
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb","tensorboard"]' \
    trainer.project_name="verl_qw25-example" \
    trainer.experiment_name='Qwen2.5-7B-Instruct_ppo' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=2 \
    $@
