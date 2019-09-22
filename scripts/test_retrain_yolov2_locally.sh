
python retrain_yolov2.py \
  --images_num=100 \
  --optimizer=adam \
  --workers=10 \
  --table_name_images=val_boxable_images \
  --table_name_image_boxes=val_images_boxes \
  --copy_db_to_scratch=f \
  --save_checkpoint_every_n_minutes=5 \
  --use_multitarget_learning=f \
  --continue_from_last_checkpoint=f \
  --validation_data_use_percentage=0.001 \
  --unfreeze_top_k_layers=all
``
