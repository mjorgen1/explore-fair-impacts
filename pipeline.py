
import os
import sys
if __name__ == "__main__":
    print("prepare data")
    os.system(f'python create_data.py --data_dir data/raw/ --output_path data/final/ --file_name test_1.csv')
    print("model grid classification")
    os.system(f'python train_grid.py --data_path data/final/test_1.csv --output_path data/results/test_1 --weight_idx 1 --testset_size 0.3')
