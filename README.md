# Cross_Stance
Cross-Target stance detection 

# Requirements:

- python==3.7.10
- numpy==1.19.2
- matplotlib==3.5.2
- seaborn==0.11.2
- pandas==1.2.4
- tqdm==4.59.0
- scikit-learn==0.24.2
- pytorch==1.9.0 (Cuda 10.2)
- transformers==4.7.0

# Training a model:

General syntax:
```
python train_model.py -m train -c [config_file] -t [train_data] -v [valid_data] -p [test_data] -n [model_name] -e [early_stopping] -s [save_checkpoints]
```

Example:
```
python train_model.py -m train -c ../../config/BiCondBertLstm_example.txt -t ../../data/UStanceBR/v2/hold1topic_out/final_bo_train.csv -v ../../data/UStanceBR/v2/hold1topic_out/final_bo_valid.csv -n bo -e 5 -s 1
```

Contact:

[Matheus Camasmie Pavan](linkedin.com/in/matheus-camasmie-pavan)
[matheus.pavan@usp.br](matheus.pavan@usp.br)
