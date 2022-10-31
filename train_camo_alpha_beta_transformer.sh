python3 train_camo_alpha_beta_transformer.py    --exp_name camo_transformer \
                                                --batch_size 60 \
                                                --features_path features/x152++_faster_rcnn \
                                                --annotation_folder annotations \
                                                --workers 2 \
                                                --alpha 0.1 \
                                                --beta 0.2 \
                                                --saved_folder saved_models