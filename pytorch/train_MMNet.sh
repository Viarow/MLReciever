python train_MMNet.py --BaseStation 1 --Antenna 1 --User 1 \
--modulation 'QAM_64' --channel 'AWGN' --amplifier 'WienerHammerstein' \
--SNRdB_min 1 --SNRdB_max 20 --train_size 65536 --batch_size_train 1024 --test_size 20 --batch_size_test 1024 \
--linear_name  'MMNet_linear' --denoiser_name 'MMNet_Denoiser' --num_layers 10 \
--epochs 500 --test_every 100 --log_every 20 --log_dir 'experiments_AWGN_order2_sharp/SISO_QAM64_AWGN_NONLINEAR_MMNet_500epochs'