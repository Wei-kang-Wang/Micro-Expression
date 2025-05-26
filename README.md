## [Learning from Macro-expressions: A Micro-expression Recognition Framework (ACMMM 2020)](https://dl.acm.org/doi/abs/10.1145/3394171.3413774)


## Installation
```bash
conda create -n micro-net python=3.8
conda activate micro-net
conda install pytorch cudatoolkit -c pytorch
pip install -r requirement.txt
```


## Dataset

In this paper, we use the datasets [CASME2](http://casme.psych.ac.cn/casme/e2), [SMIC](https://service.tib.eu/ldmservice/dataset/smic) and [SAMM](https://helward.mmu.ac.uk/STAFF/M.Yap/dataset.php). Please download them from their official website respectively.


## Train

```bash
python train.py
```
