python train_MMNet.py --BaseStation 1 --Antenna 1 --User 1 \
--modulation 'QAM_16' --channel 'AWGN' \
--SNRdB_min 1 --SNRdB_max 20 --train_size 12800 --batch_size 64 --test_size 20 \
--linear_name  'MMNet_linear' --denoiser_name 'MMNet_Denoiser' \
--epochs 200 --test_every 40 --log_every 20 --log_dir 'experiments/SISO_QAM16_AWGN_LINEAR_MMNet'