# stroke-fim-prediction-2025
# Prediction of Discharge FIM Scores using Routine Admission Variables

## Overview
This repository contains the Python code used for the retrospective observational study predicting discharge Functional Independence Measure (FIM) scores in patients with stroke.

## AI-Assisted Workflow (ChatGPT)
To ensure transparency and reproducibility, the core Python script (`fim_fim_predict.py`) was developed with the assistance of ChatGPT (OpenAI, GPT-4). 
- **Prompting Strategy:** The authors provided ChatGPT with the specific clinical variables, required modeling approach (Linear Regression), and internal validation requirements (e.g., 5-fold cross-validation).
- **Review Process:** The generated code was thoroughly reviewed, tested, and validated by the human authors to ensure clinical and statistical accuracy before execution.

## Files
- `fim_fim_predict.py`: The main script for data preprocessing, model training, and evaluation.

## Requirements
- Python 3.x
- pandas, numpy, scikit-learn, openpyxl
