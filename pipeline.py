
import os
import sys
if __name__ == "__main__":
    print("prepare data")
    os.system(f'python create_data.py')
    print("model classification")
    os.system(f'python classification.py')
