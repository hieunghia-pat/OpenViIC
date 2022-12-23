python3 train_albert_language_model.py  --exp_name albert_model \
                                        --batch_size 60 \
                                        --pretrained_name vinai/phobert-base \
                                        --features_path features/UIT-ViIC/x152++_faster_rcnn \
                                        --annotation_folder annotations/UIT-ViIC \
                                        --workers 2