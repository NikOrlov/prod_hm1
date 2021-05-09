Репозиторий с решением задачи предсказания наличия заболевания. используя
Данные для построения модели быля взяты [здесь](https://www.kaggle.com/ronitf/heart-disease-uci)

Команды необходимо запускать из папки ml_project

Настройка окружения:
* ```conda create -n .prod_hw1```
* ```conda activate .prod_hw1```
* ```conda install pip```
* ```pip3 install -r requirements.txt```

Запуск пайплайна:
```python train_pipeline.py config/train_config.yaml```