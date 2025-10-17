python train_vector_ptv3.py --num_gpus 1 --num_point 2048 --batch_size 12 \
--num_decoder_layers 3 \
--dataset vector --data_root growvector_# --num_target 64 \
--log_dir experiment \
--checkpoint ckpt_ptv3.pth \
--image_model_path image_model_config_ckpt.pkl --preserve_scale --terminal_classification --test_only