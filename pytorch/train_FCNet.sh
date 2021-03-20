python train_FCNet.py --modulation 'QAM_16' --channel 'AWGN' \
--SNRdB_min 5 --SNRdB_max 25 --train_size 12800 --batch_size 64 --test_size 128 \
--upstream 4 --downstream 5 --dropout False \
--epochs 500 --test_every 100 --log_every 20 --log_dir './SISO_QAM16_AWGN_LINEAR_FCNet'