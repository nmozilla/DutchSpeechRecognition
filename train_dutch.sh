#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=125GB


python3 train_dutch.py /data/s3757994/DutchSpeechRecognition/config/las_dutch_config.yaml
