python train_test.py --NT 16 --NR 64 --modulation 'QAM_16' --snr_min 4. --snr_max 10.0 --test_points 100 -lr 1e-3 --batch_size 2 --seq_len 2 --epochs 5 --mapping_file './simulated_data/mappings.txt' --gpu 0,1 --fig_dir './figures'