python3 train_bert_language_model.py    --exp_name bert_model \
                                        --batch_size 60 \
                                        --pretrained_name vinai/bert-base-uncased \
                                        --features_path features/UIT-ViIC/x152++_faster_rcnn \
                                        --annotation_folder annotations/UIT-ViIC \
                                        --workers 2