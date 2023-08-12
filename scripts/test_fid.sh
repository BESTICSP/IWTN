set -ex
python test_fid.py \
--dataroot /home/user/disk_3/xzx/dataset/night2day/myday2night/ \
--checkpoints_dir ./checkpoints \
--name day2night \
--gpu_ids 1 \
--model sc \
--num_test 0
