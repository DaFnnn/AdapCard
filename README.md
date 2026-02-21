# DeepCard: A Data-Driven Cardinality Estimation Method based on Adaptive Sum-Product Network

This is the soure code for AdapCard.

# Setup

Tested with python3.7

```
git clone https://github.com/DaFnnn/AdapCard.git
cd AdapCard
sudo apt install -y libpq-dev gcc python3-dev
conda env create -f environment.yml
conda activate adapcard
```

# Reproduce Experiments
The single table dataset is included in ./imdb-benchmark/dataset_name. For imdb dataset, please download the IMDB dataset. Unzip csv files to ./imdb-benchmark.

The code supports AdapCard-RSPN and AdapCard-TC-SPN with or without MetaATPM. For convenience, we introduce the experiments of AdapCard-TC-SPN with MetaATPM on four datasets.


## Learning MetaATPM
Firstly, we need to train MetaATPM, then we can learn TC-SPN with MetaATPM on any dataset without retraining.

```

python gnn_regression_double_threshold.py

```

## Reproduce Results on Power Dataset


Generate hdf files from csvs.

```
python3 maqp.py --generate_hdf
    --dataset power
    --csv_seperator ,
    --csv_path ./imdb-benchmark/power
    --hdf_path ./imdb-benchmark/gen_single_light/power
```

Generate sampled hdf files from csvs.

```
python3 maqp.py --generate_sampled_hdfs
    --dataset power
    --hdf_path ./imdb-benchmark/gen_single_light/power
    --max_rows_per_hdf_file 100000000
    --hdf_sample_size 10000
```

Learn ensembles

```
python3 maqp.py --generate_ensemble
    --dataset power
    --samples_per_spn 10000000 10000000 10000000 10000000 10000000
    --ensemble_strategy single
    --hdf_path ./imdb-benchmark/gen_single_light/power
    --ensemble_path ./imdb-benchmark/spn_ensembles/power
    --max_rows_per_hdf_file 100000000
    --post_sampling_factor 10 10 5 1 1
```

Evaluate performance for queries.

```
python3 maqp.py --evaluate_cardinalities 
    --max_variants 1
    --dataset power
    --target_path ./baselines/cardinality_estimation/results/deepCard/power_model_based_budget_5.csv
    --ensemble_location ./imdb-benchmark/spn_ensembles/power/ensemble_single_power_10000000.pkl
    --query_file_location ./benchmarks/power/sql/power.sql
    --ground_truth_file_location ./benchmarks/power/sql/power_true_card.csv
```


## Reproduce Results on Forest Dataset


Generate hdf files from csvs.

```
python3 maqp.py --generate_hdf
    --dataset forest
    --csv_seperator ,
    --csv_path ./imdb-benchmark/forest
    --hdf_path ./imdb-benchmark/gen_single_light/forest
    --max_rows_per_hdf_file 100000000
```

Generate sampled hdf files from csvs.

```
python3 maqp.py --generate_sampled_hdfs
    --dataset forest
    --hdf_path ./imdb-benchmark/gen_single_light/forest
    --max_rows_per_hdf_file 100000000
    --hdf_sample_size 10000
```

Learn ensembles

```
python3 maqp.py --generate_ensemble
    --dataset forest
    --samples_per_spn 10000000 10000000 10000000 10000000 10000000
    --ensemble_strategy single
    --hdf_path ./imdb-benchmark/gen_single_light/forest
    --ensemble_path ./imdb-benchmark/spn_ensembles/forest
    --max_rows_per_hdf_file 100000000
    --post_sampling_factor 10 10 5 1 1
```

Evaluate performance for queries.

```
python3 maqp.py --evaluate_cardinalities 
    --max_variants 1
    --dataset forest
    --target_path ./baselines/cardinality_estimation/results/deepCard/forest_model_based_budget_5.csv
    --ensemble_location ./imdb-benchmark/spn_ensembles/forest/ensemble_single_forest_10000000.pkl
    --query_file_location ./benchmarks/forest/sql/forest.sql
    --ground_truth_file_location ./benchmarks/forest/sql/forest_true_card.csv
```


## Reproduce Results on Census Dataset


Generate hdf files from csvs.

```
python3 maqp.py --generate_hdf
    --dataset census
    --csv_seperator ,
    --csv_path ./imdb-benchmark/census
    --hdf_path ./imdb-benchmark/gen_single_light/census
```

Generate sampled hdf files from csvs.

```
python3 maqp.py --generate_sampled_hdfs
    --dataset census
    --hdf_path ./imdb-benchmark/gen_single_light/census
    --max_rows_per_hdf_file 100000000
    --hdf_sample_size 10000
```

Learn ensembles

```
python3 maqp.py --generate_ensemble
    --dataset census
    --samples_per_spn 10000000 10000000 10000000 10000000 10000000
    --ensemble_strategy single
    --hdf_path ./imdb-benchmark/gen_single_light/census
    --ensemble_path ./imdb-benchmark/spn_ensembles/census
    --max_rows_per_hdf_file 100000000
    --post_sampling_factor 10 10 5 1 1
```

Evaluate performance for queries.

```
python3 maqp.py --evaluate_cardinalities 
    --max_variants 1
    --dataset census
    --target_path ./baselines/cardinality_estimation/results/deepCard/census_model_based_budget_5.csv
    --ensemble_location ./imdb-benchmark/spn_ensembles/census/ensemble_single_census_10000000.pkl
    --query_file_location ./benchmarks/census/sql/census.sql
    --ground_truth_file_location ./benchmarks/census/sql/census_true_card.csv
```





## Reproduce Results on IMDB Dataset

Download the IMDB dataset. Unzip csv files to ./imdb-benchmark.

Generate hdf files from csvs.

```
python3 maqp.py --generate_hdf
    --dataset imdb-light
    --csv_seperator ,
    --csv_path ./imdb-benchmark
    --hdf_path ./imdb-benchmark/gen_single_light/job-light
    --max_rows_per_hdf_file 100000000
```

Generate sampled hdf files from csvs.

```
python3 maqp.py --generate_sampled_hdfs
    --dataset imdb-light
    --hdf_path ./imdb-benchmark/gen_single_light/job-light
    --max_rows_per_hdf_file 100000000
    --hdf_sample_size 100000
```

Learn ensembles

```
python3 maqp.py --generate_ensemble
    --dataset imdb-light 
    --samples_per_spn 10000000 10000000 1000000 1000000 1000000
    --ensemble_strategy rdc_based
    --hdf_path ./imdb-benchmark/gen_single_light/job-light
    --max_rows_per_hdf_file 100000000
    --samples_rdc_ensemble_tests 100000 
    --ensemble_path ./imdb-benchmark/spn_ensembles/job-light
    --post_sampling_factor 10 10 5 1 1
    --ensemble_budget_factor 5
    --ensemble_max_no_joins 3
    --pairwise_rdc_path ./imdb-benchmark/spn_ensembles/job-light/pairwise_rdc.pkl
```

Evaluate performance for queries.

```
python3 maqp.py --evaluate_cardinalities 
    --rdc_spn_selection
    --max_variants 1
    --pairwise_rdc_path ./imdb-benchmark/spn_ensembles/job-light/pairwise_rdc.pkl
    --dataset imdb-light
    --target_path ./baselines/cardinality_estimation/results/deepCard/imdb_light_model_based_budget_5.csv
    --ensemble_location ./imdb-benchmark/spn_ensembles/job-light/ensemble_join_3_budget_5_10000000.pkl
    --query_file_location ./benchmarks/job-light/sql/job_light_queries.sql
    --ground_truth_file_location ./benchmarks/job-light/sql/job_light_true_cardinalities.csv
```

## Updates

Conditional incremental learning (i.e., initial learning of all films before 2011, newer films learn incremental)

```
python3 maqp.py  --generate_ensemble
    --dataset imdb-light
    --samples_per_spn 10000000 10000000 1000000 1000000 1000000
    --ensemble_strategy rdc_based
    --hdf_path ./imdb-benchmark/gen_single_light/job-light
    --max_rows_per_hdf_file 100000000
    --samples_rdc_ensemble_tests 100000
    --ensemble_path ./imdb-benchmark/spn_ensembles/job-light
    --post_sampling_factor 10 10 5 1 1
    --ensemble_budget_factor 0
    --ensemble_max_no_joins 3
    --pairwise_rdc_path ./imdb-benchmark/spn_ensembles/job-light/pairwise_rdc.pkl
    --incremental_condition "title.production_year<2011"
```

