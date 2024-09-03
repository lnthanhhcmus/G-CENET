# G-CENET: Generation Mode and Contrastive Event Network

![architecture](architecture.png)

## Authors
1. Le An Huynh: 20120028@student.hcmus.edu.vn - 0971031912
2. Thien An Nguyen: 20120030@student.hcmus.edu.vn - 0772697828
3. Thanh Le: lnthanh@fit.hcmus.edu.vn

## Preprocessing
```bash
cd data/YAGO
python get_history_graph.py
```

## Training and Testing
### ICEWS14
```bash
python main.py -d ICEWS14 --max-epochs 30 --valid-epochs 5 --alpha 0.2 --beta 0.1 --lambdax 2.0 --batch-size 1024 --lr 0.001 --save_dir SAVE --eva_dir SAVE --time-stamp 1 --entity subject
python main.py -d ICEWS14 --max-epochs 30 --valid-epochs 5 --alpha 0.2 --beta 0.1 --lambdax 2.0 --batch-size 1024 --lr 0.001 --save_dir SAVE --eva_dir SAVE --time-stamp 1 --entity object
```

### ICEWS18
```bash
python main.py -d ICEWS18 --max-epochs 30 --valid-epochs 5 --alpha 0.2 --beta 0.1 --lambdax 2.0 --batch-size 1024 --lr 0.001 --save_dir SAVE --eva_dir SAVE --time-stamp 24 --entity subject
python main.py -d ICEWS18 --max-epochs 30 --valid-epochs 5 --alpha 0.2 --beta 0.1 --lambdax 2.0 --batch-size 1024 --lr 0.001 --save_dir SAVE --eva_dir SAVE --time-stamp 24 --entity object
```

### WIKI
```bash
python main.py -d WIKI --max-epochs 30 --valid-epochs 5 --alpha 0.2 --beta 0.1 --lambdax 10.0 --batch-size 1024 --lr 0.001 --save_dir SAVE --eva_dir SAVE --time-stamp 1 --entity subject
python main.py -d WIKI --max-epochs 30 --valid-epochs 5 --alpha 0.2 --beta 0.1 --lambdax 10.0 --batch-size 1024 --lr 0.001 --save_dir SAVE --eva_dir SAVE --time-stamp 1 --entity object
```

### YAGO
```bash
python main.py -d YAGO --max-epochs 30 --valid-epochs 5 --alpha 0.1 --beta 0.1 --lambdax 10.0 --batch-size 1024 --lr 0.001 --save_dir SAVE --eva_dir SAVE --time-stamp 1 --entity subject
python main.py -d YAGO --max-epochs 30 --valid-epochs 5 --alpha 0.1 --beta 0.1 --lambdax 10.0 --batch-size 1024 --lr 0.001 --save_dir SAVE --eva_dir SAVE --time-stamp 1 --entity object
```
