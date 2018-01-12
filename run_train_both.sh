export CUDA_VISIBLE_DEVICES=0
python main.py --mode='train' --dataset='BothFer' --c_dim=7 --c2_dim=7 --image_size=64 --batch_size=32 --d_train_repeat=5 \
                 --num_iters=200000 --num_iters_decay=100000 --log_step=100 --sample_step=1000 --model_save_step=1000 \
                 --use_tensorboard=True \
                 --sample_path='stargan_both/samples' --log_path='stargan_both/logs' \
                 --model_save_path='stargan_both/models' --result_path='stargan_both/results'
