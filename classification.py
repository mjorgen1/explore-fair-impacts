from scripts.classification_utils import classify,load_args

if __name__ == '__main__':
    args = load_args('configs/classification.yaml')
    classify(**args)
