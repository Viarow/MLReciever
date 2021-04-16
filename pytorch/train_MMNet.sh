python train_MMNet.py --BaseStation 1 --Antenna 1 --User 1 \
--modulation 'QAM_256' --channel 'AWGN' --amplifier 'WienerHammerstein' \
--SNRdB_min 1 --SNRdB_max 20 --train_size 25600 --batch_size_train 256 --test_size 20 --batch_size_test 256 \
--linear_name  'MMNet_linear' --denoiser_name 'MMNet_Denoiser' --num_layers 10 \
--epochs 500 --test_every 100 --log_every 20 --log_dir 'experiments_AWGN/SISO_QAM256_AWGN_NONLINEAR_MMNet_500epochs'