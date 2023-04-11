
# Africa-Biomass-Challenge-final
Cocoa farming has driven massive deforestation in West Africa: Cote d’Ivoire, the world’s leading 
exporter, has seen a [loss of 80%](https://jp.reuters.com/article/us-ivorycoast-forests-cocoa-idUSKCN11C0IB) of its forests since its independence in 1960. While cocoa is
traditionally grown in monoculture, it can thrive under the shade of tall trees. To restore some 
of the lost tree cover, the planting of shade trees is a high priority for the country as well as 
the private sector, to reverse the impacts of deforestation and improve [carbon sequestration](https://www.usgs.gov/faqs/what-carbon-sequestration)
by African tropical forests.

In this challenge, your objective is to predict biomass in shaded regions in Cote d’Ivoire based on [GEDI](https://gedi.umd.edu/),
[Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) and ground truth biomass data. 
Remote monitoring of the increase of biomass will help measure the impact of reforestation efforts in 
Cote d’Ivoire as well as detect degradation of forests due to cocoa, without requiring expensive and labor-intensive biomass estimates on the ground.


## Installation

Python 3.7+ is required to run code from this repo.

```console
$ git clone https://github.com/biniyam369/Africa-Biomass-Challenge-final.git
$ cd Africa-Biomass-Challenge-final
```

Now let's install the requirements. But before we do that, we **strongly**
recommend creating a virtual environment with a tool such as
[virtualenv](https://virtualenv.pypa.io/en/stable/):

```console
$ virtualenv -p python3 .venv
$ source .venv/bin/activate
$ pip install -r src/requirements.txt
```

> This instruction assumes that DVC is already installed, as it is frequently
> used as a global tool like Git. If DVC is not installed, see the
> [DVC installation guide](https://dvc.org/doc/install) on how to install DVC.

This DVC project comes with a preconfigured DVC
[remote storage](https://dvc.org/doc/commands-reference/remote) that holds raw
data (input), intermediate, and final results that are produced. This is a
read-only HTTP remote.

```console
$ dvc remote list
storage https://remote.dvc.org/get-started
```

You can run [`dvc pull`](https://man.dvc.org/pull) to download the data:

```console
$ dvc pull
```

## Running in your environment

Run [`dvc repro`](https://man.dvc.org/repro) to reproduce the
[pipeline](https://dvc.org/doc/commands-reference/pipeline):

```console
$ dvc repro
Data and pipelines are up to date.
```

If you'd like to test commands like [`dvc push`](https://man.dvc.org/push),
that require write access to the remote storage, the easiest way would be to set
up a "local remote" on your file system:

> This kind of remote is located in the local file system, but is external to
> the DVC project.

```console
$ mkdir -p /tmp/dvc-storage
$ dvc remote add local /tmp/dvc-storage
```

You should now be able to run:

```console
$ dvc push -r local
```

## Existing stages

This project with the help of the Git tags reflects the sequence of actions that
are run in the DVC [get started](https://dvc.org/doc/get-started) guide. Feel
free to checkout one of them and play with the DVC commands having the
playground ready.

- `0-git-init`: Empty Git repository initialized.
- `1-dvc-init`: DVC has been initialized. `.dvc/` with the cache directory
  created.
- `2-track-data`: Raw data file `data.xml` downloaded and tracked with DVC using
  [`dvc add`](https://man.dvc.org/add). First `.dvc` file created.
- `3-config-remote`: Remote HTTP storage initialized. It's a shared read only
  storage that contains all data artifacts produced during next steps.
- `4-import-data`: Use `dvc import` to get the same `data.xml` from the DVC data
  registry.
- `5-source-code`: Source code downloaded and put into Git.
- `6-prepare-stage`: Create `dvc.yaml` and the first pipeline stage with
  [`dvc run`](https://man.dvc.org/run). It transforms XML data into TSV.
- `7-ml-pipeline`: Feature extraction and train stages created. It takes data in
  TSV format and produces two `.pkl` files that contain serialized feature
  matrices. Train runs random forest classifier and creates the `model.pkl` file.
- `8-evaluation`: Evaluation stage. Runs the model on a test dataset to produce
  its performance AUC value. The result is dumped into a DVC metric file so that
  we can compare it with other experiments later.
- `9-bigrams-model`: Bigrams experiment, code has been modified to extract more
  features. We run [`dvc repro`](https://man.dvc.org/repro) for the first time
  to illustrate how DVC can reuse cached files and detect changes along the
  computational graph, regenerating the model with the updated data.
- `10-bigrams-experiment`: Reproduce the evaluation stage with the bigrams based
  model.
- `11-random-forest-experiments`: Reproduce experiments to tune the random
  forest classifier parameters and select the best experiment.

There are three additional tags:

- `baseline-experiment`: First end-to-end result that we have performance metric
  for.
- `bigrams-experiment`: Second experiment (model trained using bigrams
  features).
- `random-forest-experiments`: Best of additional experiments tuning random
  forest parameters.

These tags can be used to illustrate `-a` or `-T` options across different
[DVC commands](https://man.dvc.org/).

## Project structure

The data files, DVC files, and results change as stages are created one by one.
After cloning and using [`dvc pull`](https://man.dvc.org/pull) to download
data, models, and plots tracked by DVC, the workspace should look like this:

```console
$ tree
.
├── README.md
├── data                  # <-- Directory with raw and intermediate data
│   ├── data.xml          # <-- Initial XML StackOverflow dataset (raw data)
│   ├── data.xml.dvc      # <-- .dvc file - a placeholder/pointer to raw data
│   ├── features          # <-- Extracted feature matrices
│   │   ├── test.pkl
│   │   └── train.pkl
│   └── prepared          # <-- Processed dataset (split and TSV formatted)
│       ├── test.tsv
│       └── train.tsv
├── dvc.lock
├── dvc.yaml              # <-- DVC pipeline file
├── eval
│   ├── importance.png    # <-- Feature importance plot
│   ├── live
│   │   ├── metrics.json  # <-- Binary classifier final metrics (e.g. AUC)
│   │   └── plots         # <-- Data points for ROC, confusion matrix
│   │       └── sklearn
│   │           ├── cm
│   │           │   ├── test.json
│   │           │   └── train.json
│   │           └── roc
│   │               ├── test.json
│   │               └── train.json
│   └── prc               # <-- Data points for custom PRC
│       ├── test.json
│       └── train.json
├── model.pkl             # <-- Trained model file
├── params.yaml           # <-- Parameters file
└── src                   # <-- Source code to run the pipeline stages
    ├── evaluate.py
    ├── featurization.py
    ├── prepare.py
    ├── requirements.txt  # <-- Python dependencies needed in the project
    └── train.py
```
