# T2G-Former

This is the implementation and experiment results of the AAAI 2023 oral paper *T2G-FORMER: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction* [link](https://arxiv.org/abs/2211.16887)

Feel free to report any [issues](https://github.com/jyansir/t2g-former/issues) and post [questions](https://github.com/t2g-former/discussions).

The implementation of T2G-Former in the original paper is `bin/t2g_former.py`.

## 1. Experiment Results

Here are some directories for experiment results.

- `results`: results of main experiment, including scores and running configs of all models on all datasets. Here are some important keys in experiment output (json file).
    - `final`: final test score after finetune.
    - `best`: best test score during finetune.
    - `cfg`: model configuration used for the experiment.
- `configs`: tuned configuration of each model on each dataset.
- `ablations`: coming soon ...
- `figures`: coming soon ...

## 2. Runtime Environment

- Basic packages: `requirements.txt` contains some basic packages to execute the script, some package versions are strict and will be told in the following instructions.

- TabNet (TF version): experiment follows the TabNet version in original paper, if you want to use script for TabNet here, `tensorflow 1.x` is needed. We execute TabNet with `tensorflow-gpu==1.14.0` on GeForce RTX 2080 Ti.

- XGBoost: We execute XGBoost with `xgboost==1.7.2`. When you encounter problems with script for XGBoost, consider to downgrade the version, it depends on its basic API.

- Other pytorch models: We execute other models with `torch==1.9.0`. Any version >= 1.9.0 is executable.
    - DANets: need `tensorboard` >= 1.15.0

- We execute all models (except TabNet TF) on NVIDIA GeForce RTX 3090.

## 3. Datasets

*LICENSE*: by downloading our dataset you accept licenses of all its components. We do not impose any new restrictions in addition to those licenses. You can find the list of sources in the section "References" of our paper.

1. Download the data: `coming soon ...`
2. Move the archive to the `data` directory: `mv xxx ./data`
3. Unpack the archive: `tar -xvf xxx`

## 4. Reproduce Experiments

- `run_experiment`: contains all scripts for our experiments.
    - `run_baselines.sh`: scripts to finetune all models on datasets. If tuned config is not given, it will load a default configuration we provided.
    - `tune_baselines.sh`: scripts to tune all model configurations on datasets.

You can modify the script, and execute it as follows:

```
CUDA_VISIBLE_DEVICES=XXX bash ./run_experiment/run_baselines.sh

# ignore system out
CUDA_VISIBLE_DEVICES=XXX nohup bash ./run_experiment/run_baselines.sh >/dev/null 2>&1 &
```

## 5. How to test your model

You can test your models by adding them to `bin` directory and `bin/__init__.py`. Keep the same API we used in other models, and execute your model with script `run_baselines.py`.

## 6. Acknowledgement

We sincerely appreciate the benchmark provided by [Yura52](https://github.com/Yura52/tabular-dl-revisiting-models) for fair comparison and script implementation.

## 7. How to cite T2G-Former

For now, cite [the Arxiv paper](https://arxiv.org/abs/2211.16887):

```
@article{yan2022t2g,
  title={T2G-Former: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction},
  author={Yan, Jiahuan and Chen, Jintai and Wu, Yixuan and Chen, Danny Z and Wu, Jian},
  journal={arXiv preprint arXiv:2211.16887},
  year={2022}
}
```
