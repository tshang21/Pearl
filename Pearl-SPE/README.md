# On the Expressivity of Stable Positional Encodings for Graphs

## About

Edits on the paper: [On the Expressivity of Stable Positional Encodings for Graphs](https://arxiv.org/abs/2310.02579). 

## Code usage

### Requirements

See requirements.txt for necessary python environment.


### Dataset

Download all required datasets from [here](https://drive.google.com/drive/folders/17nVALCgTz0LV8pVuoM0xQnRqwRH3Bz7a?usp=drive_link). The downloaded 'data' directory should be placed in the root direcotry. For example, './data/drugood', etc.

### Reproduce experiments

To reproduce experiments on ZINC, cd to ./zinc and run
```
python runner.py --config_dirpath ../configs/zinc --config_name SPE_gine_gin_mlp_pe37.yaml --seed 0
```

If you want to run REDDIT, cd to ./reddit and run:

```
python runner.py --config_dirpath ../configs/zinc --config_name reddit.yaml --seed 0
```


To reproduce experiments on Alchemy, cd to ./alchemy and run
```
python --config_dirpath ../configs/alchemy --config_name SPE_gine_gin_mlp_pe12.yaml --seed 0
```

To reproduce experiments on DrugOOD, cd to ./drugood and run
```
python --config_dirpath ../configs/assay --config_name SPE_gine_gin_mlp_pe32_zeropsi.yaml --dataset assay --seed 0
python --config_dirpath ../configs/scaffold --config_name SPE_gine_gin_mlp_pe32_standard_dropout.yaml --dataset scaffold --seed 0
python --config_dirpath ../configs/scaffold --config_name SPE_gine_gin_mlp_pe32_standard_dropout.yaml --dataset size --seed 0
```

To reproduce substructures counting, cd to ./count and run
```
bash run.sh
```
