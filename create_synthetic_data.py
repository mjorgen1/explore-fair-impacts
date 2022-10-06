from scripts.data_creation_utils import load_args,load_sample_and_save
import argparse


if __name__ == '__main__':

    # load arguments
    parser = argparse.ArgumentParser(description='Specify the path to your config file.')
    parser.add_argument('-config', type=str,help="Path to where your config yaml file is stored.")
    args = parser.parse_args()
    try:
        args = load_args(f'configs/{args.config}.yaml')
    except:
        print(f'File does not exist: configs/{args.config}.yaml')
    # create dataset
    load_sample_and_save(**args)