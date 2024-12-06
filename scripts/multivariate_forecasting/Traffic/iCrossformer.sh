export CUDA_VISIBLE_DEVICES=0

model_name=iCrossformer
seg_num=3

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --dynamic_routing 0 \
  --seg_num 3 \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --dynamic_routing 0 \
  --seg_num 3 \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --dynamic_routing 0 \
  --seg_num 3 \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --dynamic_routing 0 \
  --seg_num 3 \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1