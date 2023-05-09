# TRACE for Cardiac Arrest Prediction 

This repository is our attempt at reproducing the experiments in [Learning Unsupervised Representations for ICU Timeseries](https://proceedings.mlr.press/v174/weatherhead22a/weatherhead22a.pdf).


>Addison Weatherhead, Robert Greer, Michael-Alice
    Moga, Mjaye Mazwi, Danny Eytan, Anna Golden-
    berg, and Sana Tonekaboni. 2022. [Learning Unsupervised Representations for ICU Timeseries](https://proceedings.mlr.press/v174/weatherhead22a/weatherhead22a.pdf). In *Pro-
    ceedings of the Conference on Health, Inference, and
    Learning*, volume 174 of *Proceedings of Machine
    Learning Research*, pages 152–168. PMLR.


Much of our code is sourced from [TRACE Github Repo](https://github.com/Addison-Weatherhead/TRACE) which is the original implementation of the experiments from this paper.

## Descriptive Notebook

For a more succinct overview of our project, we have created a Jupyter Notebook which highlights the main goals and findings of our experiments. The notebook will walk users through the process of our experiments and allow users to reproduce our experiments easily. The notebook can be found in the root directory in `Descriptive_Notebook_Group62.ipynb`. Alternatively, to directly see the results of the notebook, check out `Descriptive_Notebook_Group62.html`.

## Requirements

From the repository folder, set up a virtual Python environment:

```
python -m venv env
```

Then, in order to start it run this command:

Windows:
```
env\Scripts\activate.bat
```

Mac/Linux:
```
source env/bin/activate
```

To install requirements/dependencies:

```setup
pip install -r requirements.txt
```

Dependencies: 
```
numpy==1.21.6
scikit_learn==0.21.2
statsmodels==0.10.0
torch==1.13.1
```

Note: This may require Python version 3.7.3.

## Data Download Instructions

The preprocessed data is already in this repository in the folder `/mimic-database` and does not need to be downloaded separately. To view a shorter version of the raw database, check out `/mimic-database-small-version`. The original raw data can be downloaded from [MIMIC Database](https://github.com/Addison-Weatherhead/TRACE) and was not uploaded to this Github repository because it is over 25 GB.

> Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P.C., Mark, R., Mietus, J.E., Moody, G.B., Peng, C.K. and Stanley, H.E., 2000. PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220

## Preprocessing

To see how we generated our cardiac arrest labels, check out `/mimic-database/generate_labels.py`. The labels have already been generated and are stored in `/mimic-database/labels.csv`. To observe the label generation process on a smaller version of the dataset, run the following command from the `/mimic-database-small-version` directory:
```
python generate_labels.py
```
The generated labels can then be found in `/mimic-database-small-version/labels.csv`.

To see how we cleaned our data and generated valid patient datasets, check out `/mimic-database/generate_data.py`. The valid patient data has already been generated and is stored in the folder `/mimic-database/valid_patient_data` which has one csv file per valid patient containing 3 hours of the patient's physiological signals. To observe the data generation process on a smaller version of the dataset, run the following command from the `/mimic-database-small-version` directory:
```
python generate_data.py
```
The generated data can then be found in the `/mimic-database-small-version/valid_patient_data` folder.

## Training and Evaluation

To train and evaluate the TRACE model, run this command from the root directory `/TRACE_mimic`:
```
python main.py
```

This script will train and evaluate the TRACE model on the MIMIC ICU database. It will also train and evaluate the CPC model, the Triplet-Loss model, and the End-to-End model for comparison with our baselines. Please note that we use random seeding to shuffle the data so results may vary between runs. However, the TRACE model consistently outperforms the baselines and is comparable to the supervised End-to-End model.

## Pre-trained Model

Our pretrained models for TRACE can be found in `pretrained_discriminator.pt` and `pretrained_encoder.pt`. They are trained on the MIMIC database with the parameters in `main.py`. These files are formatted using Python's `pickle` module. For more information, see [PyTorch Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html).

## Results

Our model achieves the following performance on the MIMIC database for predicting cardiac arrest:


| Model              | AUROC           | AUPRC          |
| ------------------ |---------------- | -------------- |
| TRACE              |  0.67 ± 0.16    |  0.81 ± 0.09   |
| CPC                |  0.38 ± 0.00    |  0.69 ± 0.00   |
| T-Loss             |  0.38 ± 0.00    |  0.69 ± 0.00   | 
| E2E                | 0.167 ± 0.118   |  0.581 ± 0.039 |

With these results, we have confirmed the hypothesis from the original paper ( [Learning Unsupervised Representations for ICU Timeseries](https://proceedings.mlr.press/v174/weatherhead22a/weatherhead22a.pdf) ) which was that TRACE would outperform all other baseline models in predicting patient conditions using physiological signals from ICU data. As our results show, TRACE outperformed the CPC, Triplet-Loss, and End-to-End models in the task of predicting cardiac arrest using a patient's physiological signals from ICU data.

## Contributors

Akhil Bhamidipati (akhilsb2@illinois.edu)
Hojin Ryoo (hojinr2@illinois.edu)
