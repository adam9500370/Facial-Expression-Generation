export CUDA_VISIBLE_DEVICES=0
python main.py --mode='test' --dataset='BothFer' --c_dim=7 --c2_dim=7 --image_size=64 --batch_size=32 \
               --test_model='200000' --fer2013_image_path='data/fer2013/test' --ferg_db_image_path='data/ferg_db/test' \
               --sample_path='stargan_both/samples' --log_path='stargan_both/logs' \
               --model_save_path='stargan_both/models' --result_path='stargan_both/results'
