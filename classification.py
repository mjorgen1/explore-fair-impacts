from scripts.classification_utils import classify,load_args
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Specify the path to your config file.')
    parser.add_argument('-config', type=str,
                        help="Path to where your config yaml file is stored.")
    args = parser.parse_args()

    try:
        args = load_args(f'configs/{args.config}.yaml')
    except:
        print(f'File does not exist: configs/{args.config}.yaml')

    classify(**args)
