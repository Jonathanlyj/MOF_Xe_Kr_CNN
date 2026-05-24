### Single-pressure Model
  1. Preprocessing: preprocess_local.py
  2. Training: training_property_all.py model_property_all.py
  3. Inference: 
    - predict_property_all.py predicts MOF-energy_grid level raw predictions and then aggregate at MOF level 
    - predict_gas_ads.py evaluates Xe/Kr adsorption checkpoints (see its module docstring for 1 bar example commands).
    - predict_sel.ipynb estimates derived Xe selectivity from gas adsorption prediction

### Multi-pressure Sequence Model
  1. Preprocessing: preprocess_local.py
  2. Training: training_property_all_seq.py model_property_all.py
  3. Inference: 
      - predict_property_all_seq.py predicts MOF-energy_grid level raw predictions and then aggregate at MOF level 
      - predict_sel.ipynb estimates derived Xe selectivity from gas adsorption prediction

### Multi-pressure Attention Model
  1. Preprocessing: preprocess_local.py
  2. Training: training_property_all_seq.py model_property_all.py
  3. Inference: 
      - predict_property_all_seq.py predicts MOF-energy_grid level raw predictions and then aggregate at MOF level 
      - predict_sel.ipynb estimates derived Xe selectivity from gas adsorption prediction

