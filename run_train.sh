export CUDA_VISIBLE_DEVICES=0
main_path="stargan_fer2013_res_pixel_dropout"
python main.py --mode='train' --dataset='fer2013' --c_dim=7 --image_size=64 --batch_size=32 --d_train_repeat=5 \
                 --num_epochs=200 --num_epochs_decay=100 --log_step=50 --sample_step=800 --model_save_step=800 \
                 --use_tensorboard=True \
                 --sample_path="${main_path}/samples" --log_path="${main_path}/logs" \
                 --model_save_path="${main_path}/models" --result_path="${main_path}/results"
