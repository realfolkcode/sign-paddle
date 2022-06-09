## Инструкция для инференса

0) Развертываем окружение

`conda env create -f paddleHelix.yml`

`conda activate paddleHelix`

`pip install pandarallel`

1) Копируем saved_model.pt (чекпоинт с моделью) в директорию models

2) dataset_file - это путь к датафрейму с (относительными) путями файлов для белков, лигандов и покетов

dataset_name - название для датасета (может быть любым)

`python preprocess_dataset.py --dataset_file ./data/example.csv --dataset_name example --output_path ./data/`

3) dataset - должно совпадать с dataset_name из предыдущего пункта 

`python generate_prediction.py --cuda -1 --model_dir ./models --data_dir ./data --dataset example --cut_dist 5 --num_angle 6 --batch_size 8`

На выходе получаем файл prediction.csv в директории models
