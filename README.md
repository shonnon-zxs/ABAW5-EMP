# ABAW5-EMP
#  ABAW3 @ CVPR 2022
## Task: EXPRESSION CLASSIFICATION

[ABAW5 @ CVPR 2023](https://ibug.doc.ic.ac.uk/resources/cvpr-2023-5th-abaw/)


### prepare
```bash
pip install -r requirements.txt
```
+ Activate new_env environment
```bash
conda activate new_env
```
### train

```bash
python data_preparation.py --root_dir path/to/dataset-folder --out_dir path/to/out-data-folder
```

```bash
python main.py --cfg ./conf/EXPR_baseline.yaml
```

### test
```bash
python prepare_test_data.py --root_video_dir path/to/batch-1-2-folder --dataset_dir path/to/out-data-folder
```

```bash
python main.py --cfg /path/to/config-yaml-file
```

### reference 
[https://github.com/kimngan-phan/Affwild2-ABAW3-EXPR-PRLAB](https://github.com/kimngan-phan/Affwild2-ABAW3-EXPR-PRLAB)
