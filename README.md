# GENIUS Project
This project aims to help users analyze multiple sources of multi-omics-data. The framework will create a matrix for every data source included and by stacking, create multi-channel images (Genome Images). Those images can be analyzed with computer vision, algorithms of your choice or you can use a model design we used for our study.

# Abstract 
The application of next-generation sequencing (NGS) has transformed cancer research. As costs have decreased, NGS has increasingly been applied to generate multiple layers of molecular data from the same samples, covering genomics, transcriptomics, and methylomics. Integrating these types of multi-omics data in a combined analysis is now becoming a common issue with no obvious solution, often handled on an ad-hoc basis, with multi-omics data arriving in a tabular format and analyzed using computationally intensive statistical methods. These methods particularly ignore the spatial orientation of the genome and often apply stringent p-value corrections that likely result in the loss of true positive associations. Here, we present GENIUS (GEnome traNsformatIon and spatial representation of mUltiomicS data), a framework for integrating multi-omics data using deep learning models developed for advanced image analysis. The GENIUS framework is able to transform multi-omics data into images with genes displayed as spatially connected pixels and successfully extract relevant information with respect to the desired output. Here, we demonstrate the utility of GENIUS by applying the framework to multi-omics datasets from the Cancer Genome Atlas. Our results are focused on predicting the development of metastatic cancer from primary tumours, and demonstrate how through model inference, we are able to extract the genes which are driving the model prediction and likely associated with metastatic disease progression. We anticipate our framework to be a starting point and strong proof of concept for multi-omics data transformation and analysis without the need for statistical correction. 
# Citation

https://www.biorxiv.org/content/10.1101/2023.02.09.525144v1

@article {Sokač}2023.02.09.525144,
	author = {Mateo Sokač} and Lars Dyrskj{\o}t and Benjamin Haibe-Kains and Hugo J.W.L. Aerts and Nicolai J Birkbak},
	title = {GENIUS: GEnome traNsformatIon and spatial representation of mUltiomicS data},
	elocation-id = {2023.02.09.525144},
	year = {2023},
	doi = {10.1101/2023.02.09.525144},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/02/10/2023.02.09.525144},
	eprint = {https://www.biorxiv.org/content/early/2023/02/10/2023.02.09.525144.full.pdf},
	journal = {bioRxiv}
}
# Folder organization
### The code is organized into four folder
 * GenomeImage: Code for transforming multiomics data into GenomeImages
 * Training: Code for training model
   * Models: Model design implemented in pytorch
 * ExtractWithIG: Code for using Integrated Gradients in order to retrieve attribution scores
   * This folder also includes a script to encode your genome image into a L vector
 * data: This folder is for input data
   * Contains subset of TCGA for example runs

# How to use GENIUS?
### Download the repository
```
git clone https://github.com/mxs3203/GENIUS
```
### Step 0 
We recomend using conda environment using environment.yml file. If you want to name the environment differently replace GENIUS in --name parameter
```
cd GENIUS
conda env create --name GENIUS --file environment.yml
```
If pytroch version does not work on your hardware, install the latest PyTorch distribution which works with your hardware. make sure it is compatible with latest numpy, pandas, pickle, sklearn, torchvision and all needed dependencies

### Running Example data
* The follwing script summarizes the process 
```
cd GenomeImage/
python3 make_images.py --clinical_data ../data/tcga/clinical.csv --ascat_data ../data/tcga/ascat.csv --all_genes_included ../data/tcga/all_genes_ordered_by_chr_no_sex_chr.csv --mutation_data ../data/tcga/muts.csv --gene_exp_data ../data/tcga/gene_exp_matrix.csv --gene_methyl_data ../data/tcga/methylation.csv
cd ../Training/
python3 train.py --config_file config
cd ../ExtractWithIG
python3 analyze_network.py --genome_images ../data/tcga/genome_images/ --config_file ../Training/config --clinical_data ../data/tcga/clinical.csv --model_checkpoint ../Training/saved_models/48.pb --all_genes_file ../data/tcga/all_genes_ordered_by_chr_no_sex_chr.csv --attribution_n_steps 10
```
* You can also use bash script which contains the code above
```
cd GENIUS
./run_genius.sh
```
# Running GENIUS one step at the time
### Step 1: Raw data
Organize your raw data into data folder
* Check example input 
### Step 2: Map genome data to images
* First we enter the directory with the scripts for making images
```
cd GenomeImage
```
* We can call a script using default arguments which will automatically use a subset of TCGA data we curated as an example
```
python3 make_images.py
```
* If you want to use your data check input format of data/tcga files and use argument for your data. Replace paths with your data
```
python3 make_images.py --clinical_data ../data/tcga/clinical.csv --ascat_data ../data/tcga/ascat.csv --all_genes_included ../data/tcga/all_genes_ordered_by_chr_no_sex_chr.csv --mutation_data ../data/tcga/muts.csv --gene_exp_data ../data/tcga/gene_exp_matrix.csv --gene_methyl_data ../data/tcga/methylation.csv
```
* NOTE: This step takes approx 10mins for every 500images
* Script arguments:
	* --clinical_data: Path to CSV file that must contain ID and label column we will use for prediction
 	* --ascat_data: Path to output matrix of ASCAT tool. Check the example input for required columns
  	* --all_genes_included: Path to the CSV file that contains ordered genes which will be used to create Genome Image
  	* --mutation_data: Path CSV file representing mutation data. This file should contain Polyphen2 score and HugoSymbol
  	* --gene_exp_data: Path to the csv file representing gene expression data where columns=sample_ids and there should be a column named "gene" representing the HugoSymbol of the gene
  	* --gene_methyl_data: Path to the csv file representing gene methylation data wherecolumns=sample_ids and there should be a column named "gene1" representing the HugoSymbol of the gene
## Before next step
* This should result with new folders "genome_images" and "genome_vector" inside of data folder
  * Genome images are squared images made using all gene provided in all_gene_included file
  * Genome images are required as input in the next step, therefore make sure that this step finishes.
  * Genome images are paired with clinical_data where genome_image file name is used to indetify a sample in clinical_data
### Step 3: Prepare for Training
* For training the model we will use train.py script located in Training folder.
* Assuming that we are still in GenomeImage folder we should change directory to Training folder
```
cd ../Training/
```
## Config File
The config file can be found in training folder (config) and it is used to change training hyperparameters. This should be changed based on the hardware you are using to train the model If you are running GENIUS on your own data, make sure that training input data (genome images and clinical) is stored appropriate folder and linked via config file
* The model will be saved if the best loss is detected AND if F1 score is > config['save_model_score']
  * This is used to speed up the run if we are not expecting satisfactory F1 score in early epochs
* config['predictor_column'] and config['response_column'] are zero indexed column from metadata file. In our example we use clinical.csv where column = 1 is ID, used to find a genome image and column = 4 is binary label (metastatic cancer)
* config['early_stop_patience'] is parameter that controls the tolerance for how many epochs needs to pass with no improvement of loss value. If loss is not improved after that number, training will stop.
### Step 4: Training the model
* In this step we train the model for predicting metastatic disease by running the following code:
```
python3 train.py --config_file config
```
Config arguments:
### Step 5: Inspect the model performance 
* If you are happy with model performance continue with the next step
* If you do not "trust" the model because it does not predict outcome variable with high certainty, try tweaking the model parameters or hyperparameters.

### Step 6: Integrated Gradients (IG)
* Assuming we are still in Training folder we have to change working directory into ExtractWithIG
```
cd ../ExtractWithIG
```
* The training script outputs model with the best loss in .pb format. This file is needed for IG step
* We run the following script to get the gradients of a model
```
python3 analyze_network.py --genome_images ../data/tcga/genome_images/ --config_file ../Training/config --clinical_data ../data/tcga/clinical.csv --model_checkpoint ../Training/saved_models/best_model.pb --all_genes_file ../data/tcga/all_genes_ordered_by_chr_no_sex_chr.csv --attribution_n_steps 10
```
* This script will produce csv files for every cancer type included in clinical file.
* This script will also produce PNG image representing attribution of each gene in genome image, for each data source.
* Make sure you use the same config you used for training the model
* Script arguments:
	* --genome_images: Path to a folder containing all genome images
 	* --config_file: path to a config file used for training the model (previous step)
  	* --clinical_data: Path to the CSV file that contains ID and output label
  	* --model_checkpoint: Path to a .pb file produced in the previous step. Should be Training/saved_models/best_model.pb
  	* --all_genes_file: Path to the CSV file that contains ordered genes which will be used to create Genome Image
  	* --attribution_n_steps: Number of steps integrated gradients will do in order to evaluate attribution
### Step 7: Downstream Analysis
* Use attribution score in downstream analysis of your choice



