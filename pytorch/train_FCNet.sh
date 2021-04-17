python train_FCNet.py --BaseStation 1 --Antenna 1 --User 1 \
--modulation 'QAM_256' --channel 'AWGN' --amplifier 'WienerHammerstein' \
--SNRdB_min 1 --SNRdB_max 20 --train_size 25600 --batch_size_train 256 --test_size 20 --batch_size_test 256 \
--upstream 1 --downstream 1 \
--epochs 500 --test_every 100 --log_every 20 --log_dir 'experiments_AWGN_order2_sharp/SISO_QAM256_AWGN_NONLINEAR_FCNet_500epochs'