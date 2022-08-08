
import os
import sys
if __name__ == "__main__":
    print("prepare data")

    print
    os.system(f"python create_data.py -input data/raw/ -output data/final")
