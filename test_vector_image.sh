bash init_test.sh
python train_vector.py --num_point 2048 \
--batch_size 12 --num_decoder_layers 3 \
--dataset vector --data_root growvector_# \
--num_target 64 --log_dir experiment --terminal_classification \
--checkpoint ckpt_pointnet.pth \
--image_model_path image_model_config_ckpt.pkl --preserve_scale --test_only --num_workers 1