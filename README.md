# Supposedly Fair Classification Systems and Their Impacts
Mackenzie Jorgensen, Elizabeth Black, Natalia Criado, & Jose Such

The algorithmic fairness field has boomed with discrimination mitigation methods to make Machine Learning (ML) model
predictions fairer across individuals and groups. However, recent research shows that these measures can sometimes lead
to harming the very people Artificial Intelligence practitioners want to uplift. In this paper, we take this research a step
further by including real ML models, multiple fairness metrics, and discrimination mitigation methods in our experiments to
understand their relationship with the impact on groups being classified. We highlight how carefully selecting a fairness
metric is not enough when taking into consideration later effects of a model’s predictions–the ML model, discrimination
mitigation method, and domain must be taken into account. Our experiments show that most of the mitigation methods,
although they produce “fairer” predictions, actually do not improve the impact for the disadvantaged group, and for those
methods that do improve impact, the improvement is minimal. We highlight that using mitigation methods to make models
more “fair” can have unintended negative consequences, particularly on groups that are already disadvantaged.

We owe a great deal to Liu et al.'s work, [*Delayed Impact of Fair Machine Learning*](https://arxiv.org/abs/1803.04383). We extended their [code](https://github.com/lydiatliu/delayedimpact) here to solve a classification problem with 
multiple ML models, fairness metrics, and mitigation methods. 

**Problem Domain**: loan repayment

**Datasets**:
Our simulated datasets are based on Hardt et al.'s 2016 dataset. 
- Download the data folder from the Github repository for [fairmlbook](https://github.com/fairmlbook/fairmlbook.github.io/tree/master/code/creditscore) (Barocas, Hardt and Narayanan 2018)
- Save it to the root directory of this repository (csvs should be in the folder 'data')
- Then run: ```delayedimpact/Liu_paper_code/FICO-figures.ipynb```

# Repo Structure
  Files:
    - requirements.txt contains the required python packages for the project
    - create_data.py, classification.py, pipeline.py -> run from cmd line
  Folder:
    - Liu_paper_code: contains the forged code from https://github.com/lydiatliu/delayedimpact (indirectly used for data collection)
    - scripts: contais all functions used for data collection, classification, evaluation and visualisations (stored in seperate py files)
    - configs: contains yaml files, which entail configurations for data collection and classification from cmd line
    - notebooks: contais the notebooks to run the code ( data collection, classification, evaluation/statistics and visualizations)

# Project Pipeline

This project can be divided into three stages:
1. Generation of (potentially synthetic) simulated dataset
2. Training and testing ML models
3. Visualizing and performing statistical analyses on results

This section gives a high-level overview of the workflow of each section and what is needed to run the code.
Stage 1 and 2 of the pipeline can be eiher run via notebook or via cmd line. The third part is only executable via notebooks.

## 1. Dataset Generation

This section prepares the simulated (potentially synthetic) dataset that will be used for training and testing the unmitigated and mitigated models. 

**Parameters**:
- Need to be set:
  - directory of the raw data from Hardt et al. (2016)
  - directory for the created synthetic dataset
  - filename of the created dataset
- Can be changed:
  - set_size: absolute size of the dataset (number of samples)
  - group_size_ratio: ratio of race in the dataset (black to white samples)
  - black_label_ratio: ratio of balck samples with true and false labels.
  - order_of_magnitude: number of samples with are drawn by step
  - shuffle_seed: controls the shuffle of the dataset
  - round_num_scores: indicator of how the scores are rounded
  
**Key details**:
- The original dataset according to Hardt et al. (2016) has the group_size_ratio: [0.12;0.88] and black_label_ratio: [0.66;0.34]. 
  By changing those params we interfere tith the score distributions and create a synthetic dataset.
- The ```delayedimpact/scripts/data_creation_utils.py``` is the pyfile that includes all of the helpful functions for the data collection
- How to run:
  - Way 1: Run the notebook (```delayedimpact/notebooks/simData_collection```) and set params in the third cell
  - Way 2: Set params in ```configs/data_creation``` and run ```python create_data.py``` from any cmd line.


## 2. Training and Testing ML Models

This section trains ML models on the simulated data and trains unmitigated and mitigated models on it for comparison. 

**Parameters**:
- Need to be set:
  - datapath to the dataset created in Stage1
  - directory for saving the results for the models
- Can be changed:
  - weight_idx:
  - testset_size: absolute size of the dataset (number of samples)
  - demo_ratio: ratio of race in the dataset (black to white samples)
  - label_ratio: ratio of black samples with true and false labels.
  - balance_test_set: 
  - set_bound: upper boundary of the absolute trainset and testset size
  - models: classifier that should be trained and tested
  - constraints: fairness constraints that should be applied o the classifiers
  - reduction_algorithms: algos that should be applied
  - save: (bool) indicator if results should be saved
  
**Key details**:
- The ```delayedimpact/scripts/classification_utils.py``` and ```delayedimpact/scripts/evaluation_utils.py``` are the pyfiles that include all of the helpful functions for the classification.
- How to run:
  - Way 1: Run the notebook (```delayedimpact/notebooks/simData_classification```) and set params in the second cell
  - Way 2: Set params in ```configs/classification``` and run ```python classification.py``` from any cmd line.


## 3. Performing statistical analyses on results

This work is under construction.

<!-- NOTES -->
## Notes/Resources:
- For the reduction algorithm code see: [Grid Search](https://github.com/fairlearn/fairlearn/blob/main/fairlearn/reductions/_grid_search/grid_search.py) and [Exponentiated Gradient](https://github.com/fairlearn/fairlearn/blob/main/fairlearn/reductions/_exponentiated_gradient/exponentiated_gradient.py)
- Reduction algorithms and fairness constraints: 'disparity constraints are cast as Lagrange multipliers, which cause the reweighting and relabelling of the input data. This *reduces* the problem back to standard machine learning training.'
- Fairness constraint options: DP refers to demographic parity, EO to equalized odds, TPRP to true positive rate parity, FPRP to false positive rate parity, ERP to error rate parity, and BGL to bounded group loss.
- The ML models available (these sklearn models' fit functions take in sample weights which is necessary for Fairlearn): gaussian naive bayes, decision tree, logistic regression, and svm. Currently, all samples equally (weight_index=1).
- The sklearn confusion matrix looks like:
  ```
  [[TN FP]
   [FN TP]]
  ```
- Impact score changes: TPs' scores increase by 75, FPs' scores drop by 150, and TNs and FNs do not change currently. Also, for aggregate analyses, we use the average score change of each (racial) group.
- Race features: Black is 0 and White it 1.   

<!-- CONTACT -->
## Contact
* Mackenzie Jorgensen - mackenzie.jorgensen@kcl.ac.uk
* Hannah Richert - hrichert@ous.de

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgments
Thank you to Lydia for helping me get started using her code!

<!-- License -->
## License
Lydia's [repository](https://github.com/lydiatliu/delayedimpact) is licensed under the BSD 3-Clause "New" or "Revised" [License](https://github.com/lydiatliu/delayedimpact/blob/master/LICENSE).
