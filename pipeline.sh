#!/bin/bash

# Запуск генерации данных
echo "Generating data..."
python3 data_generation.py

# Запуск предобработки данных
echo "Preprocessing data..."
python3 model_preprocessing.py

# Запуск обучения модели
echo "Training model..."
python3 model_preparation.py

# Запуск тестирования модели
echo "Testing model..."
python3 model_testing.py

echo "Pipeline completed!"