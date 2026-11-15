# Predicting Xe/Kr Gas Adsorption in MOFs with 3D CNN

Separating Xe and Kr is industrially costly due to cryogenic distillation requirements. Metal-Organic Frameworks (MOFs) offer a selective-adsorption alternative, but identifying optimal structures demands extensive simulation. This repo accelerates MOF screening by predicting Xe/Kr mixture adsorption directly from **3D energy grids** of guest/host interaction energies using CNN-based models.

Two model families are provided:

1. **Single-pressure CNN** : a separate model per pressure level; achieves Xe adsorption R² up to 0.961 and Xe selectivity R² ~0.8.
2. **Multi-pressure Sequence RCNN** : one model trained across all pressures simultaneously; improves Xe selectivity R² from 0.867 to 0.894 by capturing inter-pressure dependencies in the adsorption isotherm.

## Setup

Requires **Python 3.11**. Install the packages listed in `requirements.txt`
```bash
pip install -r requirements.txt
```
## Sample data 
`sample_data/` contains a small MOF subset for a quick demo. Voxel grids (`clean.npy`) are under `sample_data/{train,val,test}/`; pressure-specific label CSVs are under `sample_data/<pressure>/` (e.g. `sample_data/1bar/`); multi-pressure labels are under `sample_data/all_seq/`.

---

## 1. Single-Pressure CNN

Train and predict separately for **Kr** and **Xe** at each pressure, then derive Xe selectivity. Example below uses `1bar`.

### Train (run once per target)

```bash
 python train_gas_ads_model.py \
  --pressure 1bar \
  --grid-path ./sample_data \
  --load-dir ./sample_data/1bar \
  --model-save-dir ./single_pressure_models \
  --output-dir ./single_pressure_models \
  --target Kr \
  --seed 42
```

Repeat with `--target Xe`.

### Predict (run once per target)

```bash
python predict_gas_ads.py \
  --model-path /path/to/kr_model.pt \
  --grid-path ./sample_data \
  --load-dir ./sample_data/1bar \
  --target Kr \
  --seed 42

python predict_gas_ads.py \
  --model-path /path/to/xe_model.pt \
  --grid-path ./sample_data \
  --load-dir ./sample_data/1bar \
  --target Xe \
  --seed 42
```

### Derived Xe Selectivity

```bash
python predict_Xe_sel.py \
  --kr-pred /path/to/kr_predictions.csv \
  --xe-pred /path/to/xe_predictions.csv \
  --output-csv ./single_pressure_models/xe_selectivity.csv
```

Output CSV contains the merged Kr/Xe true and predicted adsorption columns plus `Xe_selectivity` and `Xe_selectivity_pred`.

---

## 2. Multi-Pressure Sequence RCNN

A single model trained over all six pressure levels at once. Train and predict for **Kr** and **Xe**, then derive per-pressure Xe selectivity.

### Train (run once per target)

```bash
 python train_gas_ads_model_seq.py \
  --grid-path ./sample_data \
  --load-dir ./sample_data/all_seq \
  --model-save-dir ./multi_pressure_models \
  --output-dir ./multi_pressure_models \
  --target Kr \
  --seed 168
```

Repeat with `--target Xe`.

### Predict (run once per target)

```bash
 python predict_gas_ads_seq.py \
  --model-path /path/to/kr_model.pt \
  --grid-path ./sample_data \
  --load-dir ./sample_data/all_seq \
  --target Kr \
  --arch multioutput \
  --seq-order forward \
  --seed 168

 python predict_gas_ads_seq.py \
  --model-path /path/to/xe_model.pt \
  --grid-path ./sample_data \
  --load-dir ./sample_data/all_seq \
  --target Xe \
  --arch multioutput \
  --seq-order forward \
  --seed 168
```

### Derived Xe Selectivity (all pressures)

```bash
python predict_Xe_sel_seq.py \
  --kr-pred /path/to/kr_seq_predictions.csv \
  --xe-pred /path/to/xe_seq_predictions.csv \
  --output-csv ./multi_pressure_models/xe_selectivity_seq.csv
```

Output CSV contains per-pressure Kr/Xe true and predicted adsorption columns plus `sel_true_<pressure>` and `sel_pred_<pressure>` for each of the six pressure levels.


## Developer Team

The code was developed by Youjia Li from the <a href="http://cucis.ece.northwestern.edu/">CUCIS</a> group at the Electrical and Computer Engineering Department at Northwestern University.

Copyright (C) 2026, Northwestern University.
