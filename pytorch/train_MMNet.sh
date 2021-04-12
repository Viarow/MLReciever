python train_MMNet.py --BaseStation 1 --Antenna 1 --User 1 \
--modulation 'QAM_16' --channel 'RayleighFading' \
--SNRdB_min 5 --SNRdB_max 30 --train_size 25600 --batch_size_train 256 --test_size 25 --batch_size_test 256 \
--linear_name  'MMNet_linear' --denoiser_name 'MMNet_Denoiser' --num_layers 10 \
--epochs 200 --test_every 50 --log_every 20 --log_dir 'experiments_RayleighFading/SISO_QAM16_Rayleigh_LINEAR_MMNet_200epochs'