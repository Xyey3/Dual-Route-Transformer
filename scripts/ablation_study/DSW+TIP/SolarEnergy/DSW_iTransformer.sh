
fix_seed=${FIX_SEED}
echo "Running DSW_iTransformer with fix_seed=$fix_seed"
model_name=DSW_iTransformer
seg_num=6

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_96 \
  --model $model_name \
  --data Solar \
  --features M \
  --seg_num $seg_num\
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 1e-4 \
  --itr 1 \
  --fix_seed $fix_seed

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_192 \
  --model $model_name \
  --data Solar \
  --features M \
  --seg_num $seg_num\
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 1e-4 \
  --itr 1 \
  --fix_seed $fix_seed

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_336 \
  --model $model_name \
  --data Solar \
  --features M \
  --seg_num $seg_num \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 1e-4 \
  --itr 1 \
  --fix_seed $fix_seed

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_720 \
  --model $model_name \
  --data Solar \
  --features M \
  --seg_num $seg_num \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --patience 1 \
  --learning_rate 1e-4 \
  --itr 1 \
  --fix_seed $fix_seed