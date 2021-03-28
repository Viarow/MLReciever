python train_FCNet.py --BaseStation 1 --Antenna 10 --User 10 \
--modulation 'QAM_16' --channel 'AWGN' \
--SNRdB_min 1 --SNRdB_max 20 --train_size 25600 --batch_size_train 128 --test_size 20 --batch_size_test 128 \
--upstream 4 --downstream 5 --dropout True \
--epochs 500 --test_every 100 --log_every 20 --log_dir 'experiments/10I10O_QAM16_AWGN_LINEAR_FCNet_500epochs'