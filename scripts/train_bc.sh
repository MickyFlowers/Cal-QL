export http_proxy=http://192.168.32.11:18000
export https_proxy=http://192.168.32.11:18000


CUDA_VISIBLE_DEVICES=0 

XLA_PYTHON_CLIENT_PREALLOCATE=false python -m bc.train_bc \
    logging.online=true \
    logging.prefix=bc \
    seed=0 \
    logging.project=screwdriver-il \
    policy_obs_head_arch=256-256 \
    policy_hidden_dim=256 \
    policy_out_head_arch=256-256 \
    batch_size=16 \
    num_workers=4 \
    trainer.learning_rate=1e-6 \
    data_root=/mnt/pfs/dataset/screwdriver-il-converted \
    n_train_epochs=1000 \
    save_model=True \
    save_ckpt_dir=/root/workspace/Cal-QL/checkpoints/ \
    save_model_interval=200 \
    # load_ckpt_dir=/root/workspace/Cal-QL/checkpoints/bc--screwdriver-il/20251105-101205 \

