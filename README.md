# SerotoninLevelDetection

Development of Neural Networks to classify FSCV serotonin data.

## Installation

To create an environment and install the dependencies required for the project, navigate to base folder and use:

```bash
make create_environment

conda activate SerotoninLevelDetection

make requirements
```

If the dependencies fail to install automatically for any reason, the missing ones will be displayed and can be installed manually.

## Usage

To run the application, navigate to the `/src` folder.

Once the dependencies are installed, use the following script to run the application locally:

```
python .
```

The application will run in the command line. The simple interface is provided.

CLI steps:

1. Select the function: Train or Predict. (Type the word)
2. Select the model (Type an option number)
3. For training, select whether cross validation in required (Type an option number)
4. Enter the version

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. (Used for testing and statistical analysis)
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── reports
    │   ├── results        <- Model prediction results
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── partition_dataset.py
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   ├── CustomDataGenerator.py
    │   │   ├── CustomDataGenerator_5s.py
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to build, train and test models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   ├── test_model.py
    │   │   ├── build_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

---

## Authors

```
Author: Tomas Andriuskevicius
Supervisor: Dr. Parastoo Hashemi, Sergio Mena, and Melissa Hexter

MSc Project, Imperial College London, 2022
```

## License

[MIT](https://choosealicense.com/licenses/mit/)

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
