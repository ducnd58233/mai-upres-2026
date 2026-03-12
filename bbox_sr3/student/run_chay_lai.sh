# python train_antsr_fp32.py \
#   --data_root /home/namnguyen/projects/MobileAI/data/DIV2K \
#   --out_dir /home/namnguyen/projects/MobileAI/bbox_sr3/student/runs_mobileone_32_10_cleanfix_plus \
#   --preset balanced \
#   --device cuda --seed 1234 \
#   --rep_type mobileone --channels 32 --n_rep 10 --rep_act_mode relu \
#   --mo_branches 2 --mo_use_1x1 --mo_use_identity --rep_use_bn \
#   --skip_mode add --use_global_add \
#   --out_clamp_fp32 none --out_clamp_export minclip --out_clamp_qat minclip \
#   --epochs1 600 --epochs2 220 --epochs3 120 \
#   --lr1 1e-3 --lr2 2e-5 --lr3 2e-6 \
#   --patch1 128 --patch2 160 --patch3 144 \
#   --s1_loss l1 \
#   --s2_loss charbdct --dct_w_s2 0.02 \
#   --s3_loss charbdct --dct_w_s3 0.03 \
#   --scheduler1 cos_warmup --scheduler2 step_halve --scheduler3 step_halve \
#   --grad_clip 1.0 \
#   --ema --ema_decay 0.999 \
#   --qat --deploy_before_qat --qat_mode fx --qat_backend qnnpack \
#   --qat_disable_observer_ep 8 --qat_freeze_fakequant_ep 40 \
#   --teacher_type mambair \
#   --mambair_repo /home/namnguyen/projects/MobileAI/bbox_sr3/MambaIR \
#   --mambair_teacher_ckpt /home/namnguyen/projects/MobileAI/bbox_sr3/mambairv2_lightSR_x3.pth \
#   --kd_loss l1 \
#   --kd_w_s2 0.03 \
#   --kd_w_s3 0.025 \
#   --kd_w_s3_start 0.03 --kd_w_s3_end 0.01 \
#   --kd_freq_w_s3 0.0 \
#   --kd_conf_gamma 12.0 --kd_conf_min 0.05 --kd_conf_max 1.0 \
#   --kd_sched_p1 0.3 --kd_sched_p2 0.7 --kd_low_floor 0.04 \
#   --kd_use_residual_s3 \
#   --fqkd_w 0.015 --fqkd_layers 1,4,7 \
#   --amp_fp32 --channels_last \
#   --val_shaves 0,3 --best_key psnr_sr_rgb_sh0 \
#   --report_ssim --ssim_win 11 --ssim_sigma 1.5 \
#   --workers 2 --batch 16 --val_every 1 \
#   --log_every 200 \
#   --freeze_bn_epoch 80 \
#   --bn_recalib_batches 64 \
#   --no-channel_shuffle_s2s3 \
#   --resume /home/namnguyen/projects/MobileAI/bbox_sr3/student/runs_mobileone_32_10_cleanfix_plus/balanced/ckpt_best_s2_fp32.pt



python train_antsr_fp32.py \
  --data_root /home/namnguyen/projects/MobileAI/data/DIV2K \
  --out_dir /home/namnguyen/projects/MobileAI/bbox_sr3/student/runs_mobileone_32_10_bestbet_v3 \
  --preset balanced \
  --device cuda --seed 1234 \
  --rep_type mobileone --channels 32 --n_rep 10 --rep_act_mode relu \
  --mo_branches 2 --mo_use_1x1 --mo_use_identity --rep_use_bn \
  --skip_mode add --use_global_add \
  --out_clamp_fp32 none --out_clamp_export minclip --out_clamp_qat minclip \
  --epochs1 600 --epochs2 240 --epochs3 90 \
  --lr1 1e-3 --lr2 1.5e-5 --lr3 1e-6 \
  --patch1 128 --patch2 160 --patch3 128 \
  --s1_loss l1 \
  --s2_loss charbdct --dct_w_s2 0.02 \
  --s3_loss charbdct --dct_w_s3 0.015 \
  --scheduler1 cos_warmup --scheduler2 step_halve --scheduler3 step_halve \
  --grad_clip 1.0 \
  --ema --ema_decay 0.999 \
  --qat --deploy_before_qat --qat_mode fx --qat_backend qnnpack \
  --qat_disable_observer_ep 6 --qat_freeze_fakequant_ep 24 \
  --teacher_type mambair \
  --mambair_repo /home/namnguyen/projects/MobileAI/bbox_sr3/MambaIR \
  --mambair_teacher_ckpt /home/namnguyen/projects/MobileAI/bbox_sr3/mambairv2_lightSR_x3.pth \
  --kd_loss l1 \
  --kd_w_s2 0.03 \
  --kd_w_s3 0.012 \
  --kd_w_s3_start 0.015 --kd_w_s3_end 0.006 \
  --kd_freq_w_s3 0.0 \
  --kd_conf_gamma 10.0 --kd_conf_min 0.10 --kd_conf_max 0.75 \
  --kd_sched_p1 0.25 --kd_sched_p2 0.60 --kd_low_floor 0.03 \
  --fqkd_w 0.0 \
  --amp_fp32 --channels_last \
  --val_shaves 0,3 --best_key psnr_sr_rgb_sh0 \
  --report_ssim --ssim_win 11 --ssim_sigma 1.5 \
  --workers 2 --batch 16 --val_every 1 \
  --log_every 200 \
  --freeze_bn_epoch -1 \
  --bn_recalib_batches 64 \
  --no-channel_shuffle_s2s3


python train_antsr_v2.py \
  --data_root /home/namnguyen/projects/MobileAI/data/DIV2K \
  --out_dir /home/namnguyen/projects/MobileAI/bbox_sr3/student/runs_mobileone_32_4_ver2_bien_the \
  --preset balanced \
  --device cuda --seed 1234 \
  --rep_type mobileone --channels 32 --n_rep 4 --rep_act_mode relu \
  --mo_branches 4 --mo_use_1x1 --mo_use_identity --rep_use_bn \
  --skip_mode add --use_global_add \
  --no-image_residual \
  --no-use_block_res_scale \
  --out_clamp_fp32 none --out_clamp_export minclip --out_clamp_qat minclip \
  --epochs1 600 --epochs2 240 --epochs3 120 \
  --lr1 1e-3 --lr2 1.5e-5 --lr3 1e-6 \
  --patch1 128 --patch2 160 --patch3 144 \
  --s1_loss l1 \
  --s2_loss charbdct --dct_w_s2 0.02 \
  --s3_loss charbdct --dct_w_s3 0.015 \
  --scheduler1 cos_warmup --scheduler2 step_halve --scheduler3 step_halve \
  --grad_clip 1.0 \
  --ema --ema_decay 0.999 \
  --qat --deploy_before_qat --qat_mode fx --qat_backend qnnpack \
  --qat_disable_observer_ep 6 --qat_freeze_fakequant_ep 24 \
  --teacher_type mambair \
  --mambair_repo /home/namnguyen/projects/MobileAI/bbox_sr3/MambaIR \
  --mambair_teacher_ckpt /home/namnguyen/projects/MobileAI/bbox_sr3/mambairv2_lightSR_x3.pth \
  --kd_loss l1 \
  --kd_w_s2 0.03 \
  --kd_res_w_s2 0.0 \
  --kd_wave_w_s2 0.0 \
  --kd_w_s3 0.01 \
  --kd_res_w_s3 0.0 \
  --kd_wave_w_s3 0.0 \
  --kd_freq_w_s3 0.0 \
  --kd_w_s3_start 0.010 --kd_w_s3_end 0.005 \
  --kd_conf_gamma 10.0 --kd_conf_min 0.10 --kd_conf_max 0.75 \
  --kd_sched_p1 0.25 --kd_sched_p2 0.60 --kd_low_floor 0.01 \
  --fqkd_w 0.0 \
  --amp_fp32 --channels_last \
  --val_shaves 0 --best_key psnr_sr_rgb_sh0 \
  --report_ssim --ssim_win 11 --ssim_sigma 1.5 \
  --workers 2 --batch 16 --val_every 1 \
  --log_every 200 \
  --freeze_bn_epoch -1 \
  --bn_recalib_batches 64 \
  --no-channel_shuffle_s2s3