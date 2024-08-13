# PhosphoricAcidDetect

# insurct to use
python train_P_model.py --data data/test.fasta --num_seq_labels 2 --epochs 5 --resume pretrained --write_validate_to_log\
python train_P_model.py --data data/data_1000_60232_partition.fasta --num_seq_labels 2 --save_model --write_validate_to_log --resume pretrianed \
python train_P_model.py --data data/test.fasta --num_seq_labels 2 --epochs 5 --write_validate_to_log --resume pretrianed


