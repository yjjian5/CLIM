cd src
CUDA_VISIBLE_DEVICES=0,1 python train.py mot --exp_id mix_ft_ch_dla34_CC --load_model '../models/crowdhuman_dla34.pth' --data_cfg '../src/lib/cfg/data.json'
cd ..