echo 'Entering GenomeImage folder'
cd GenomeImage/
echo 'Running make_images.py'
python3 make_images.py --clinical_data ../data/tcga/clinical.csv --ascat_data ../data/tcga/ascat.csv --all_genes_included ../data/tcga/all_genes_ordered_by_chr_no_sex_chr.csv --mutation_data ../data/tcga/muts.csv --gene_exp_data ../data/tcga/gene_exp_matrix.csv --gene_methyl_data ../data/tcga/methylation.csv
echo 'make_image.py finished'
echo 'Entering Training folder'
cd ../Training/
echo 'Training the model using Genome Images created in previous step...'
python3 train.py --config_file config
echo 'Training finished'
echo 'Entering ExtractWithIG folder...'
cd ../ExtractWithIG
echo 'Analyzing network using Integrated gradients and producing .csv files for each cancer type included in the study...'
python3 analyze_network.py --genome_images ../data/tcga/genome_images/ --config_file ../Training/config --clinical_data ../data/tcga/clinical.csv --model_checkpoint ../Training/saved_models/best_model.pb --all_genes_file ../data/tcga/all_genes_ordered_by_chr_no_sex_chr.csv --attribution_n_steps 10

