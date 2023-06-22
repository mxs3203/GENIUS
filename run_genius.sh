echo 'Entering GenomeImage folder'
cd GenomeImage/
echo 'Running make_images.py'
python3 make_images.py --clinical_data ../data/example_data/clinical.csv --ascat_data ../data/example_data/ascat.csv --all_genes_included ../data/example_data/all_genes_ordered_by_chr_no_sex_chr.csv --mutation_data ../data/example_data/muts.csv --gene_exp_data ../data/example_data/gene_exp_matrix.csv --gene_methyl_data ../data/example_data/methylation.csv
echo 'make_image.py finished'
echo 'Entering Training folder'
cd ../Training/
echo 'Training the model using Genome Images created in previous step...'
python3 train.py --config_file config
echo 'Training finished'
echo 'Entering ExtractWithIG folder...'
cd ../ExtractWithIG
echo 'Analyzing network using Integrated gradients and producing .csv files for each cancer type included in the study...'
python3 analyze_network.py --genome_images ../data/example_data/genome_images/ --config_file ../Training/config --clinical_data ../data/example_data/clinical.csv --model_checkpoint ../Training/saved_models/best_model.pb --all_genes_file ../data/example_data/all_genes_ordered_by_chr_no_sex_chr.csv --attribution_n_steps 10

