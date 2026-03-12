# python train_antsr_fp32.py \
#   --data_root /home/namnguyen/projects/MobileAI/data/DIV2K \
#   --out_dir runs_mobileone_24_12_safe \
#   --preset balanced \
#   --device cuda --seed 1234 \
#   --rep_type mobileone --channels 24 --n_rep 12 --rep_act_mode relu \
#   --mo_branches 2 --mo_use_1x1 --mo_use_identity --rep_use_bn \
#   --skip_mode add --use_global_add \
#   --out_clamp_fp32 none --out_clamp_export minclip --out_clamp_qat minclip \
#   --epochs1 0 --epochs2 0 --epochs3 150 \
#   --lr2 2e-5 --lr3 3e-6 \
#   --patch2 160 --patch3 128 \
#   --s2_loss charbdct --dct_w_s2 0.02 \
#   --s3_loss charbdct --dct_w_s3 0.03 \
#   --scheduler2 step_halve --scheduler3 step_halve \
#   --grad_clip 1.0 \
#   --ema --ema_decay 0.999 \
#   --qat --deploy_before_qat --qat_mode fx --qat_backend qnnpack \
#   --qat_disable_observer_ep 8 --qat_freeze_fakequant_ep 60 \
#   --teacher_type mambair \
#   --mambair_repo /home/namnguyen/projects/MobileAI/bbox_sr3/MambaIR \
#   --mambair_teacher_ckpt /home/namnguyen/projects/MobileAI/bbox_sr3/mambairv2_lightSR_x3.pth \
#   --kd_loss l1 \
#   --kd_w_s2 0.03 \
#   --kd_w_s3 0.04 \
#   --kd_w_s3_start 0.05 --kd_w_s3_end 0.02 \
#   --kd_freq_w_s3 0.0 \
#   --kd_conf_gamma 12.0 --kd_conf_min 0.05 --kd_conf_max 1.0 \
#   --kd_sched_p1 0.3 --kd_sched_p2 0.7 --kd_low_floor 0.04 \
#   --kd_use_residual_s3 \
#   --fqkd_w 0.03 --fqkd_layers 0,2,4 \
#   --enable_pact_qat \
#   --pact_init_min 0.0 --pact_init_max 32.0 \
#   --pact_l2 5e-6 \
#   --amp_fp32 --channels_last \
#   --val_shaves 0,3 --best_key psnr_sr_rgb_sh0 \
#   --report_ssim --ssim_win 11 --ssim_sigma 1.5 \
#   --workers 2 --batch 16 --val_every 1 \
#   --log_every 200 \
#   --no-channel_shuffle_s2s3 



python train_antsr_fp32.py \
  --data_root /home/namnguyen/projects/MobileAI/data/DIV2K \
  --out_dir runs_mobileone_24_12_push \
  --preset balanced \
  --device cuda --seed 1234 \
  --rep_type mobileone --channels 24 --n_rep 12 --rep_act_mode relu \
  --mo_branches 2 --mo_use_1x1 --mo_use_identity --rep_use_bn \
  --skip_mode add --use_global_add \
  --out_clamp_fp32 none --out_clamp_export minclip --out_clamp_qat minclip \
  --epochs1 0 --epochs2 0 --epochs3 180 \
  --lr3 3e-6 \
  --patch3 128 \
  --s3_loss charbdct --dct_w_s3 0.03 \
  --scheduler3 step_halve \
  --grad_clip 1.0 \
  --ema --ema_decay 0.999 \
  --qat --deploy_before_qat --qat_mode fx --qat_backend qnnpack \
  --qat_disable_observer_ep 8 --qat_freeze_fakequant_ep 60 \
  --teacher_type mambair \
  --mambair_repo /home/namnguyen/projects/MobileAI/bbox_sr3/MambaIR \
  --mambair_teacher_ckpt /home/namnguyen/projects/MobileAI/bbox_sr3/mambairv2_lightSR_x3.pth \
  --kd_loss l1 \
  --kd_w_s3 0.045 \
  --kd_w_s3_start 0.055 --kd_w_s3_end 0.025 \
  --kd_freq_w_s3 0.0 \
  --kd_conf_gamma 12.0 --kd_conf_min 0.05 --kd_conf_max 1.0 \
  --kd_sched_p1 0.3 --kd_sched_p2 0.7 --kd_low_floor 0.04 \
  --kd_use_residual_s3 \
  --fqkd_w 0.04 --fqkd_layers 0,2,4 \
  --enable_pact_qat \
  --pact_init_min 0.0 --pact_init_max 32.0 \
  --pact_l2 3e-6 \
  --amp_fp32 --channels_last \
  --val_shaves 0,3 --best_key psnr_sr_rgb_sh0 \
  --report_ssim --ssim_win 11 --ssim_sigma 1.5 \
  --workers 2 --batch 16 --val_every 1 \
  --log_every 200 \
  --no-channel_shuffle_s2s3