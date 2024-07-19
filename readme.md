# Set of Sets

## Overview

This codebase allows you to perform experiments with the Set of Sets. It constitutes an evolutionary approach to compression of deep neural networks. The algorithmic core is based on MO-MFEA I/II. Experiments are performed on the following datasets:

1) Multilingual Translation (Custom Transformer - Not reported): https://www.statmt.org/wmt18/translation-task.html
2) Multilingual Translation (M2M100-418M - Reported): https://www.statmt.org/wmt19/translation-task.html
3) Beijing Air Quality Regression (Non-Timeseries, Feedforward Neural Network - Not Reported): https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data
4) Beijing Air Quality Regression (Timeseries, LSTM - Reported)
5) CIFAR-10
6) MNIST

The structure of the codebase is as follows:
```
SoS_Final_Code
│   results_analysis.ipynb # Jupyter notebook for plotting the results of all experiments
│   run_all.py # Python subprocess file for executing all experiments in order. Hyperparameters
| 			   # are changed here.
└─── NLP # Multilingual Translation experiment code
│   └───MFEA
│   |	└───data
|   |	|   └───evo_data
|   |	|	| 	└───extraction_set # Data used for evolutionary fitness assessment.
|   |   |	|	|			   # 50k datapoints per language.
|	|	|	|	|	|	cs_only.pkl # Czech Language in pickled format - A tuple of two arrays. First array
|	|	|	|	|	|			    # is source sentences. Second array is target sentences. Format
|	|	|	|	|	|			    #standard.
|	|	|	|	|	|	de_only.pkl # German Language in identical format as above.
|	|	|	|	└───test_set # Small test dataset of 10k datapoints. For final assessment of evolved |	|	|	|	|	|	|	|		 # models.
|	|	|	|	|	| ...
|	|	|	└───finetuning_data # Data used for finetuning after evolution.
|	|	|	|	└───extraction_set # Training data. 200k datapoints per language.
|	|	|	|	|	| ...
│	|   └───models
|	|   |   |	base.pth # Pre-trained transformer model, trained for German and Czech.
│	|   └───prepared_model_info # Information about the pre-prepared models. Used to select pareto-optimal
│	|   │	|					# models to jumpstart evolution
|	|   |   └───cs_only
|	|	|	|	|	stats_dict.pkl # Array of arrays of tuples. There is one array per language/task.
|	|	|	|	| 			   	   # Within that array is tuples, where first element is performance and
|	|	|	| 	|			   	   # second element is size.
|	|   |   └───de_only
|	|	|	|	|	stats_dict.pkl
|	|   |   |	mask_dict.pkl # All masks, which define pre-prepared models. Further explanation below.
│	|   └───results # Folder for storing experimental results.
│	|   |	chromosome.py # Chromosomes/binary mask definitions located here.
│	|   |	compare_solutions.py # Pareto selection used in MFEA here.
│	|   |	hyp.py # Functions for calculating hypervolumes here.
│	|   |	model.py # Transformer model definition here.
│	|   |	multitask.py # Main evolutionary code here.
│	|   |	operators.py # Evolutionary operators and RMP learning defined here.
│	|   |	prep_models.py # Code for pre-preparation of models here.
│   └───Prep_data_and_model # Prepare base transformer model and process data.
│	|   └───data # Processed data
|	|	|	| train.pkl # Processed data for training transformer
|	|	|	| val.pkl # Processed data for testing trained transformer
│	|   └───dataprocessing
|	|	|	└───cs-en # Raw Czech-English data from WMT18
|	|	|	└───de-en # Raw German-English data from WMT18
|	|	|	|	dataprocessing.ipynb # Notebook for preprocessing data
│	|   └───results
│	|   |	main.py # Training functions for transformer
|	|	|   model.py # Transformer definition
```
## Getting Started

Codebases for each experiment are identical except for choice of metric and loss function. Hence, once familiarized with the code for one experiment, familiarizing with the others should be straightforward. The code for the MNIST experiments is annotated and its functions and general flow are shared across all experiments. Please refer to the MNIST experiment for clarification.

### If using provided data
1) Open the .zip file containing all pre-prepared datasets. There should be four folders.
2) Place the contents of each folder in the corresponding /data/ folder in the main codebase.
3) Open run_all.py. Modify hyperparameters and select experiments as desired. Hyperparameters are as follows:
	* run_name = 'Test_Run' # Name of experiment - Folder containing results will be named this.
	* pretrained = '1' # Whether or not to use pre-prepared models to jumpstart evolution. Set to '0' if running comparison between singletask and multitask. Otherwise hypervolumes will be squashed.
	* rmp = '0.4' # rmp setting - If using MFEA-II, value does not matter.
	* final_finetune_switch = '1' # '1' enables finetuning. '0' disables finetuning
	* MFEA_II = '0' # '1' uses MFEA-II with automated rmp learning. '0' uses MFEA-I
	* finetune_epochs = '20' # Number of epochs to finetune per model.
	* preprepared_population = '1000' # Number of models to assess for pre-preparation.
4) Open ./*/main.py, where * refers to an experiment like NLP or Regression. Evolutionary hyperparameter settings are in a dictionary called hyperparams. This includes Generations for evolution, size of population for each task, and SBXDI and PMDI settings. Change this according to your requirements.

### If not using provided data
1) Download raw data from provided sources.
2) Run all boxes in order in Prep_data_and_model/dataprocessing.ipynb notebook (For NLP) or run all boxes in order in the prep_data_and_model.ipynb notebook (For all other experiments).
3) Modify data preprocessing to suit your needs.
4) Retrain models.
5) Edit hyperparams dictionary in main.py.
6) Rerun all experiments in run_all.py.

## General Process Overview

1) Acquire raw data.
2) Preprocess data. You will need a training set and a testing set for preparing the JAT. You will also need a training set for evolution, and a test set for final assessment after evolution. It is recommended that each of these be subsets of the full training and testing datasets.
	* For evolution, smaller datasets are acceptable. Larger datasets will of course provide better finetuned performance.
	* For NLP, there is a separate training dataset for evolution and for finetuning. This is because the NLP experiment requires large amounts of data for training, but needs less data for evolution. This is important for reducing computational expense.
3) Choose a model, and train it on the full datasets.
4) Pre-prepare models. This means creating a very large population of randomly generated candidates, and then evaluating them on the training data. The Pareto-optimal members of this population will form the initial population for evolution. This is equivalent to running evolution with a very large initial population, then cutting it down in subsequent generations. This will jumpstart evolution, and can result in better finetuned performance. It is not recommended to use this when evaluating multi-task vs single-task, because it will blur the differences between them.
	* Skip this step if it is not necessary for your purposes.
5) Use trained model and prepared datasets for evolution.
6) Finetune evolved models.
7) Plot results.

## Binary Masking

The binary mask is our means of simulating a compressed model. This is a vector of continuous variables from 0 to 1, which is then rounded and applied to a model. Each model used contains an apply_mask function, and is accompanied by a size_mask function. The size_mask function examines the structure of the model and then determines the appropriate size of the binary mask. It also segments the binary mask by layer. For example, in a 3 layer network with 5 nodes in layer 1, 10 in layer 2, and 10 in layer 3. You may have a binary mask of size 150 (5x10 + 10x10). The first fifty will correspond to the first layer, and the last hundred will correspond to the second layer. Each variable represents an individual parameter.

Alternatively, you may have a 3 layer network with 10,000 nodes in each layer. This would result in a binary mask of 2x10e-8. This is not feasible. So you could have a binary mask of dimensionality 200. Then, the first 100 variables will correspond to the first layer, and each variable accounts for 1,000,000 parameters. This means if a variable is deactivated (Set to 0), it will shut off 1,000,000 parameters. The granularity of this masking is a hyperparameter to be tuned. Higher specificity (Bigger mask) can provide better final results at the cost of making evolution costlier.

Please examine the apply_mask (In model.py) and size_mask (In multitask.py) functions in detail to understand how this works. With RNNs, Transformers, and Fully Connected Models, you can have each variable represent a group of parameters. For CNNs, you can have each variable represent a group of filters.

## Creating your own experiments

Each domain has idiosyncracies and requires a unique approach suited for the specific model being used, and to the specific metrics of interest. Here are some notes regarding the domains considered so far.

### NLP
1) NLP is the most expensive experiment by far.
2) We train a transformer, then we apply binary masking on all linear layers. Do not touch embeddings - Data processing should give you an indexed vocabulary. To complete compression, you can use this indexed vocabulary to remove the embedding parameters corresponding to the unneeded languages. For example:
	* Vocabulary with 5 german words and 5 czech words. German words indexed from 0 to 4. Czech words indexed from 5 to 9. Embedding vector of dimension 10x512. I can remove the embeddings at index 0 to 4, and thus have embeddings that only account for Czech language.
3) Multilingual translation is tough, so scaling to more than 2 languages will require extensive resources. Recommend reducing the length of sentences and cutting down vocabulary to compensate - This is of course dependent on your specific needs.
4) Task specialization is simply about restricting the dataset from multiple languages to a subset of them. So, here, we trained a model for Czech and German translation. When specializing, we give evolution a dataset containing only one language.

### Regression
1) This is the least expensive since the data is lowest dimensionality.
2) We used a fully-connected model. Binary mask is applied to groups of parameters in all layers.
3) Task specialization is about restricting to geographic area. Hence, in evolution we exclude the data which didn't come from a specific geographical area.

### Image Classification
1) We used CNNs. Binary mask can be applied to filters directly. Can also be extended to the fully-connected layers (if any).
2) Specialization is about selecting a subset of labels of interest. So, we provide datasets containing only a few classes per task. We also remove the output nodes corresponding to the tasks which we are no longer interested in.

## Notes

### Pre-preparation of models using prep_model.py

Pre-preparation of models is countrolled with the pretrain setting in run_all.py This amounts to creating a large population of candidates, assessing them on training data, and then selecting the pareto optimal members as the first generation of evolution. This speeds up the process and can result in better finetuned performance. However, it reduces the differences between multitasking and singletasking (Because it starts evolution off in a more optimal position). Hence, enable it when the focus is on acquiring better finetuned performance. Disable it when the focus is on revealing efficiency differential between multitasking and singletasking.

### Constraints on Candidate Masks

The size/dimensionality of candidate masks is controlled by the initialization of the Chromosome class. To generate a candidate mask of dimension 2048, where the number of 1s is constrained to be between 0.2 x 2048 and 0.99 x 2048, you would initialize the Chromosome in the following way:

```
temp = Chromosome()
temp.initialize(2048, 0.2, 0.99)
```

The degree of constraint is problem and model dependent. Intuitively, larger models with greater degrees of overparametrization and larger binary masks allow for smaller lower bounds on size. In the Regression experiments, an LSTM of size 0.56 MB is sufficient to achieve good performance. Hence we constrain its size to be at least 20% of the original model. If the constraint is removed, many poorly performing models are generated. On the other hand, in the MNIST experiments, we use ResNet-18, which is extremely powerful relative to the dataset on which it is used. Hence, we constrain the candidate masks to between 0% and 10% of the original model size. In other words, this constraint is dependent upon how many parameters the model can afford to lose before it breaks down.

This hyperparameter helps to focus the search, and can result in significant performance and efficiency gains.
