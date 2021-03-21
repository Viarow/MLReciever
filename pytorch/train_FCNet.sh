python train_FCNet.py --BaseStation 1 --Antenna 10 --User 10 \
--modulation 'QAM_16' --channel 'AWGN' \
--SNRdB_min 5 --SNRdB_max 25 --train_size 12800 --batch_size 64 --test_size 128 \
--upstream 4 --downstream 5 --dropout False \
--epochs 500 --test_every 100 --log_every 20 --log_dir './10I10O_QAM16_AWGN_LINEAR_FCNet'