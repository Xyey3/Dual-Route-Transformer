
fix_seed=${FIX_SEED}
echo "Running DRFormer with fix_seed=$fix_seed"
model_name=DRFormer
seg_num=6

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seg_num $seg_num \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --fix_seed $fix_seed \

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seg_num $seg_num \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --fix_seed $fix_seed \

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seg_num $seg_num \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --fix_seed $fix_seed \

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seg_num $seg_num \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --fix_seed $fix_seed \
  
# fix_seed=${FIX_SEED}
# echo "Running DRFormer with fix_seed=$fix_seed"
# model_name=DRFormer
# seg_num=3
# batch_size=8

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seg_num $seg_num \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 1 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 256 \
#   --itr 1 \
#   --fix_seed $fix_seed \
#   --batch_size $batch_size \
#   --learning_rate 5e-5

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_96_192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seg_num $seg_num \
#   --seq_len 96 \
#   --pred_len 192 \
#   --e_layers 1 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 256 \
#   --itr 1 \
#   --fix_seed $fix_seed \
#   --batch_size $batch_size \
#   --learning_rate 5e-5

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_96_336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seg_num $seg_num \
#   --seq_len 96 \
#   --pred_len 336 \
#   --e_layers 1 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --itr 1 \
#   --fix_seed $fix_seed \
#   --batch_size $batch_size \
#   --learning_rate 5e-5

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/exchange_rate/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_96_720 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seg_num $seg_num \
#   --seq_len 96 \
#   --pred_len 720 \
#   --e_layers 1 \
#   --enc_in 8 \
#   --dec_in 8 \
#   --c_out 8 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --itr 1 \
#   --fix_seed $fix_seed \
#   --batch_size $batch_size \
#   --learning_rate 5e-6
  