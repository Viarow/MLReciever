python train_FCNet.py --BaseStation 1 --Antenna 1 --User 1 \
--modulation 'QAM_16' --channel 'RayleighFading' \
--SNRdB_min 5 --SNRdB_max 30 --train_size 25600 --batch_size_train 256 --test_size 25 --batch_size_test 256 \
--upstream 1 --downstream 1 \
--epochs 200 --test_every 50 --log_every 20 --log_dir 'experiments_RayleighFading/SISO_QAM16_Rayleigh_LINEAR_FCNet_200epochs'