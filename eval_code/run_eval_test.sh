export CUDA_VISIBLE_DEVICES=0
main_path="stargan_fer2013_res_pixel_dropout"
python main.py --resume=checkpoints/checkpoint.pth.tar -e --data="../${main_path}/results/fer2013"
