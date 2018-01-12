export CUDA_VISIBLE_DEVICES=0
main_path="stargan_fer2013_res_pixel_dropout"
python main.py --mode='test' --dataset='fer2013' --c_dim=7 --image_size=64 --batch_size=32 \
               --test_model='200_800' --fer2013_image_path='data/fer2013/test' \
               --sample_path="${main_path}/samples" --log_path="${main_path}/logs" \
               --model_save_path="${main_path}/models" --result_path="${main_path}/results"
