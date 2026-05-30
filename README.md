# CV2 Final Project

This project is based on the domain of robust Head Pose Estimation (HPE). Specifically, it aims to demonstrate how facial occlusions reduce the performance of head pose estimation models and to find a viable solution through data augmentation by creating synthetic facial occlusions.

## Project Overview

The workflow of the project is organized around three main notebooks:

- `main.ipynb` is the core notebook of the project. It contains the main experiments, the fine-tuning stages, the validation and test evaluation and the error cases.
- `occlusion_demo.ipynb` is a demonstration notebook for the custom occlusion system. It shows how facial landmarks are extracted, synthetic occlusions are generated and the custom dataset is created from the I2Head database from the Public University of Navarre (UPNA).
- `models_demo.ipynb` is where the three candidate models are executed and compared before the final experiments TokenHPE is selected as the final model for the project experiments.

## Repository Structure

- `main.ipynb`: main notebook for training and experiments.
- `occlusion_demo.ipynb`: occlusion system demonstration and custom dataset generation.
- `models_demo.ipynb`: comparison notebook for the three candidate models.
- `dataset.py`: dataset definition.
- `utils.py`: helper functions for training, evaluation, visualization and occlusion generation.
- `models/`: model definitions and support code.
- `weights/`: pretrained and fine-tuned model weights.
- `dataset/`: dataset files.
- `test_failure/`: manually collected failure examples and edge cases.
- `report.pdf`: Contains the detailed report of our work.

## Requirements

The project is based on Python with the following main dependencies:

- PyTorch
- torchvision
- NumPy
- SciPy
- OpenCV
- Matplotlib
- scikit-learn
- PIL / Pillow
- face-alignment
- tqdm
- peft

Install the dependencies in your preferred environment before running the notebooks.

## Typical Workflow

1. Prepare or verify the datasets under `dataset/`. The dataset can be downloaded through [this link](https://www.kaggle.com/datasets/705dc64640b143c1e31acbef73281cd9b809e111f1314882932dbc39a96c992d).
2. Run `occlusion_demo.ipynb` if you need to regenerate or inspect the occluded samples.
3. Use `models_demo.ipynb` to compare candidate models and confirm the selection of TokenHPE.
4. Run `main.ipynb` to train, fine-tune and evaluate the final experiments. Our fine-tuned model's weights can be downloaded in [this link.](https://huggingface.co/enricbc/Fully_Fine-tuned_TokenHPE_occlusion)

## Authors

Antonio Vila Leis and Enric Baixauli Casañ.
