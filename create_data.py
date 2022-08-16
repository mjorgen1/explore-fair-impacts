from scripts.data_creation_utils import load_args,load_sample_and_save

if __name__ == '__main__':
    args = load_args('configs/data_creation.yaml')
    load_sample_and_save(**args)
