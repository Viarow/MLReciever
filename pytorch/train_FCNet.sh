python train_FCNet.py --BaseStation 1 --Antenna 10 --User 10 \
--modulation 'QAM_16' --channel 'AWGN' \
--SNRdB_min 1 --SNRdB_max 20 --train_size 12800 --batch_size 64 --test_size 20 \
--upstream 4 --downstream 5 --dropout False \
--epochs 200 --test_every 40 --log_every 20 --log_dir './10I10O_QAM16_AWGN_LINEAR_FCNet_Constant'