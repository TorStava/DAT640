# dat640-project

## Introduction

This repo contains the code accompanying the final report in DAT640 Autumn 2023 at the University of Stavanger. 

## Dataset
The dataset must be downloaded and placed in the `data/collection` folder. 

See [README](data/README.md) for more details.

## Requirements
* Elasticsearch version 7.17 or higher

Python requirements can be installed running `conda env create -f environment.yml` in the main folder.

## Indexing
Indexing is a one-time operation that populates Elasticsearch with the parsed documents for the data collection. 

See [README](treccast/indexer/README.md) for more details.

## Baseline
The code for the baseline method using PyTerrier and Random Forest can be found here: [baseline](baseline).

## Advanced method
The code for the advanced method based CTS and MVR and using Elasticsearch and Huggingface transformers can be found here: [advanced](treccast).


## References
A lot of coding inspiration has been taken from these Github repositories:
* https://github.com/iai-group/ecir2023-reproducibility/tree/master
* https://github.com/microsoft/MSMARCO-Conversational-Search
* https://github.com/CodingTil/2023_24---IRTM---Group-Project
* https://github.com/novasearch/conversational-search-assistant-transformers
