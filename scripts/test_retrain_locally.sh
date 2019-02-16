
python retrain.py \
  --images_num=100 \
  --model=inceptionV3 \
  --optimizer=adam \
  --workers=3 \
  --copy_db_to_scratch=f \
  --save_checkpoint_every_n_minutes=3 \
  --use_multitarget_learning=f \
  --continue_from_last_checkpoint=f \
  --validation_data_use_percentage=0.001 \
  --unfreeze_top_k_layers=all
