# dataset construction
This is the data and code we used for dataset construction. 

**NOTE:** 
Please note that all directory paths mentioned in this README are **relative to the ./dataset_construction/ root directory**.  
This means that when a path such as ./data/ is referenced, it should be understood as ./dataset_construction/data/. This convention is used throughout this document to maintain clarity and brevity.

## Ontology
As our event dataset are constructed based on CAMEO ontology,
especially that our event extraction uses the multi-level relation information,
we convert information in CAMEO document into python dictionaries and store them under the following folder:
```
./data/
```

## News articles
To distribute the event extraction pipeline on multiple cards simultaneously, we split all original news articles based on their publication years.
For example, all raw data for the country Egypt(EG) in Mideast are now convert to files in the following folder:
```
./data_cleaned/
```

## Multi-level event extraction
First, set up environment for [Vicuna-13b](https://github.com/lm-sys/FastChat):
```
conda create -n vicuna-cuda11 python=3.9
conda activate vicuna-cuda11
conda install pytorch==1.12.0 -c pytorch

pip3 install fschat
pip install openai

pip install notebook
```
Then, conduct the multi-level event extraction prompt and response parsing, where Y is an int number from 2015 to 2022:
- Level 1
```
python3 -m fastchat.src_all.prompt_level1 --model your_vicuna_model_path --load-8bit --year Y --output_path your_output_path
python3 -m fastchat.src_all.parse_level1 --year Y --output_path your_output_path
```
- Level 2
```
python3 -m fastchat.src_all.prompt_level2 --model your_vicuna_model_path --load-8bit --year Y --output_path your_output_path
python3 -m fastchat.src_all.parse_level2 --year Y --output_path your_output_path
```
- Level 3
```
python3 -m fastchat.src_all.prompt_level3 --model your_vicuna_model_path --load-8bit --year Y --output_path your_output_path
python3 -m fastchat.src_all.parse_level3 --year Y --output_path your_output_path
```

The raw responses for each level prompt will be saved under:
```
your_output_path/Y/results
your_output_path/Y/results_level2
your_output_path/Y/results_level3
```
The parsed structured event for each level responses will be saved under:
```
./results/level1/Y_events.csv
./results/level2/Y_events.csv
./results/level3/Y_events.csv
```
Then, we merge all extracted events for all years and all levels:
```
python ./fastchat/src_all/merge_levels.py
```
The merged results will be saved at:
```
./results/all_events_df_EG.csv
```
After the same extraction pipeline is applied to all EG, IR, and IS we used in phase3, we obtain all raw extracted events in MidEast, this data is saved at (can be downloaded from S3):
```
./data_final/EGIRIS_raw.csv
```
And we use all md5s in EGIRIS_raw.csv to filter the structured event in original GDELT dataset, and save this data at (can be downloaded from S3):
```
./data_final/GDELT_EGIRIS.csv
```

## Data cleaning and entity linking
We conduct several steps for Vicuna-extracted data cleaning:
- Drop NA entity
- Filter entity by term frequency >= 10 and document frequency >= 10
- Entity linking using the mapping dictionary generated from GPT4 (this dictionary is also saved under ./data_final)

This pipeline can be found in the following notebooks:
```
./data_final/final_data.ipynb
./data_final/entlink.ipynb
```
The cleaned data is saved at:
```
./data_final/EGIRIS_entlink.csv
```

## Complex event clustering
We first prepare the document embedding and time embedding for each atomic event.

All documents are too large to be uploaded to github, so needed to be downloaded from S3 and put under folder as following:
```
./clustering/all_vicuna_docs.json
```
It contains all text documents present in ./data_final/EGIRIS_entlink.csv. It also corresponds to ./clustering/all_vicuna_md5_list.json.


### Feature extraction
We use [SimCSE](https://github.com/princeton-nlp/SimCSE) to extract document embedding:
```
python ./clustering/get_simcse_emb.py --output_path your_raw_doc_embs_output_path
```

Then, we construct doc and time embeddings for all atomic events in the cleaned dataset:
```
python ./clustering/prepare_data.py --input_path ./data_final/EGIRIS_entlink.csv --output_path ./clustering/entlink --doc_embed_path your_raw_doc_embs_output_path
```
After this step, they will be saved at:
```
./clustering/entlink/doc_embs.npy
./clustering/entlink/time_embs.npy
```

### Time-aware clustering
We adopt the UMAP + HDBSCAN pipeline in [BERTopic](https://github.com/MaartenGr/BERTopic) and add the time feature into its original code.

This pipeline with the final adopted parameter is implement in the following notebook:
```
./BERTopic/notebooks/entlink/noprob_time_t1.0_n200_m10.ipynb
```
After this step, all atomic events with raw complex event id (named as topic id under BERTopic) are saved at:
```
./clustering/ce/t1_m10_raw.csv
```

### Data cleaning and preparation
To construct the final complex event dataset, we conduct several steps:
- We merge atomic events if they have the same (subject, relation, object, time, ceid)
- We split too large complex events by max date range = 78, and max atomic event number = 112
- We filter out complex events into outliers if they do not have min date range = 2 and min atomic event number = 10
- We split complex events into train, valid and test by their center date, and both valid and test set have one year of data
- We filter out entities and relations in valid and test set if they do not present in train set


All these steps and stats for intermediate and final results in this pipeline can be found in the following notebooks for MidEast-CE(EGIRIS) and GDELT-CE:
```
./clustering/ce/clean_ce_EGIRIS.ipynb
./clustering/ce/clean_ce_GDELT.ipynb
```
Final event data will be saved at:
```
./clustering/ce/EGIRIS_ce_final.csv
./clustering/ce/EGIRIS_ce_train.csv
./clustering/ce/EGIRIS_ce_valid_final.csv
./clustering/ce/EGIRIS_ce_test_final.csv
./clustering/ce/EGIRIS_ce_outliers_final.csv

./clustering/ce/GDELT_ce_final.csv
./clustering/ce/GDELT_ce_train.csv
./clustering/ce/GDELT_ce_valid_final.csv
./clustering/ce/GDELT_ce_test_final.csv
./clustering/ce/GDELT_ce_outliers_final.csv
```

Then, we convert these event data into id data files for graph construction and training:
```
For TKG baseline data:
./clustering/ce/prepare_data_tkg_EGIRIS.ipynb
./clustering/ce/prepare_data_tkg_GDELT.ipynb

For CE model data:
./clustering/ce/prepare_data_ce_EGIRIS.ipynb
./clustering/ce/prepare_data_ce_GDELT.ipynb
```
The generated id structured data will be saved at:
```
./clustering/ce/data_tkg_EGIRIS/
./clustering/ce/data_tkg_GDELT/

./clustering/ce/data_ce_EGIRIS/
./clustering/ce/data_ce_GDELT/
```
These are the final files we used to train and test TKGBaselines and LoGo.
In particular, here, ./clustering/ce/data_ce_EGIRIS/ contains all information included in MIDEAST_CE and ./clustering/ce/data_ce_GDELT/ contains all information included in GDELT_CE.
