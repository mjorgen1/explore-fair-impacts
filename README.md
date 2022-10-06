# Supposedly Fair Classification Systems and Their Impacts
Mackenzie Jorgensen, Hannah Richert, Elizabeth Black, Natalia Criado, & Jose Such

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

Our real-world dataset is the German Credit Dataset from the UCI ML Repo.
- Download the ```german.data``` file from https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data) into a data folder

# Repo Structure
 - Files:
    - requirements.txt contains the required python packages for the project
    - create_synthetic_data.py, create_german_data.py, classification.py, classification_german.py -> run from cmd line
 - Folder:
    - Liu_paper_code: contains the forged code from https://github.com/lydiatliu/delayedimpact (indirectly used for data collection)
    - configs: contains yaml files, which entail configurations for data collection and classification from cmd line
    - scripts: contains all functions used for data collection, classification, evaluation and visualisations (stored in seperate py files)
    - notebooks: contains the notebooks to run the code ( data collection, classification, evaluation/statistics and visualizations)

# Project Pipeline

This project can be divided into three stages:
1. Generation/collection of (potentially synthetic) a simulated dataset or real world dataset
2. Training and testing ML models
3. Visualizing and performing statistical analyses on results

This section gives a high-level overview of the workflow of each section and what is needed to run the code.
Stage 1 and 2 of the pipeline can be eiher run via notebook or via cmd line. The third part is only executable via jupyter notebooks.

## 1. Dataset Generation

This section prepares the simulated, synthetic dataset (or the German Credit dataset) that will be used for training and testing the unmitigated and mitigated models. 
  
**Key details--Synthetic datasets**:
- The original dataset according to Hardt et al. (2016) has the group_size_ratio: [0.12;0.88] and black_label_ratio: [0.66;0.34]. 
  By changing those parameters, we interfere with the score distributions and create a synthetic dataset.
- The ```delayedimpact/scripts/data_creation_utils.py``` is the pyfile that includes all of the helpful functions for the data collection.
- How to run:
  - Way 1: Run the notebook (```/notebooks/simData_collection```) and set parameters in the third cell.
  - Way 2: Set parameters in ```configs/data_creation``` or create your own .yaml file in the folder and run ```python create_synthetic_data.py -config data_creation``` from any cmd line (you can substitute the -config parameter with your own yaml-file name).

**Key details--German Credit dataset**:
- The original dataset according to UCI ML repo has the group_size_ratio: [0.31;0.69], a disadv-label-ratio: [0.35,0.65], and an adv_label_ratio: [0.28,0.72]. 
- How to run:
  - Run ```python create_german_data.py``` from any cmd line (set correct parameters, like the path at the beginning of the file).

## 2. Training and Testing ML Models

This section describes training ML models on the simulated data and training unmitigated and mitigated models on the data for comparison. 

**Key details--Synthetic dataset**:
- The ```/scripts/classification_utils.py``` and ```/scripts/evaluation_utils.py``` are the pyfiles that include all of the helpful functions for the classification.
- How to run:
  - Way 1: Run the notebook (```/notebooks/simData_classification```) and set params in the second cell
  - Way 2: Set params in ```configs/classification``` or create your own .yaml file in the folder and run ```python classification.py -config classification``` from any cmd line (you can substitude the -config parameter with your own yaml-file name).
  
**Key details--German Credit dataset**:
- The ```/scripts/classification_utils.py``` and ```/scripts/evaluation_utils.py``` are the pyfiles that include all of the helpful functions for classification.
- How to run:
  -  Set parameters in ```configs/classification_german``` or create your own .yaml file in the folder and run ```python classification_german.py -config classification_german``` from any cmd line (you can substitude the -config parameter with your own yaml-file name).


## 3. Performing statistical analyses on results

In this section, we investigate the immediate and delayed impact results, check the score distributions for Normality and then their significance based on different aspects of the experiments.
- How to run (set parameters in the second cell): 
 - Run the notebook (```/notebooks/simData_classification```)

<!-- NOTES -->
## Notes/Resources:
- Fairness constraint options: DP refers to demographic parity, EO to equalized odds, TPRP to true positive rate parity, FPRP to false positive rate parity, ERP to error rate parity, and BGL to bounded group loss.
- The ML models available (these sklearn models' fit functions take in sample weights which is necessary for Fairlearn): gaussian naive bayes, decision tree, logistic regression, and svm. Currently, all samples equally (weight_index=1).
- The sklearn confusion matrix looks like:
  ```
  [[TN FP]
   [FN TP]]
  ```

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
