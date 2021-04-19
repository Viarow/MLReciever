python distribution.py --BaseStation 1 --Antenna 1 --User 1 \
--modulation 'QAM_16' --channel 'AWGN' --SNRdB_min 1 --SNRdB_max 20 \
--train_size 10240 --batch_size_train 64 --test_size 20 --batch_size_test 64 \
--fig_dir 'dataset/distribution_eval'