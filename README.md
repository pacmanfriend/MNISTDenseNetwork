# MNIST Dense Network

Desktop-приложение для распознавания рукописных цифр MNIST. Проект содержит собственную реализацию полносвязной нейросети на NumPy, GUI на Tkinter, preprocessing пользовательских изображений через OpenCV и сохранение модели в HDF5.

## Возможности

- загрузка MNIST через `tensorflow.keras.datasets`;
- создание, обучение, сохранение и загрузка dense-модели;
- асинхронное обучение без блокировки GUI;
- настройка размера обучающей выборки, количества эпох, batch size, learning rate и числа worker-процессов;
- отображение progress/status по эпохам;
- загрузка внешнего изображения цифры и предсказание класса;
- просмотр примеров из тренировочного MNIST-набора;
- экспериментальный PyTorch baseline для сравнения.

## Стек

- Python
- NumPy
- Tkinter
- Matplotlib
- OpenCV
- SciPy
- TensorFlow/Keras datasets
- h5py
- PyTorch для экспериментального baseline

## Установка

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

На Windows активация окружения будет отличаться:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск GUI

```bash
.venv/bin/python main.py
```

При первом запуске приложение скачает MNIST через Keras. После загрузки данных можно создать или загрузить модель, настроить параметры обучения и запустить обучение.

## Запуск тестов

```bash
.venv/bin/python -m unittest discover -s tests
```

## PyTorch baseline

В `experimental_networks/pytorch_network.py` лежит простая reference-модель на PyTorch. Ее можно запустить отдельно:

```bash
.venv/bin/python experimental_networks/pytorch_network.py --epochs 5 --batch-size 128
```

## Архитектура

```text
main.py                    точка входа GUI-приложения
app/
  gui.py                   Tkinter-интерфейс и UI callbacks
  trainer.py               запуск обучения в отдельном потоке
  state.py                 состояние приложения: модель и MNIST-датасет
  config.py                TrainingConfig и валидация параметров
  datasets.py              загрузка MNIST
  preprocessing.py         нормализация изображений и labels
neural/
  dense_network.py         NumPy dense network, fit/predict/evaluate/save/load
  tools.py                 OpenCV preprocessing внешних изображений
experimental_networks/
  pytorch_network.py       экспериментальный PyTorch baseline
tests/
  test_config_preprocessing.py
```

Основной поток приложения отвечает за GUI. Загрузка MNIST и обучение выполняются асинхронно, а результаты возвращаются в UI через callbacks. Подготовка данных вынесена отдельно, чтобы один и тот же preprocessing использовался в обучении, тестах и предсказании.

## Модель

Собственная модель `DenseNetwork` реализована на NumPy:

- вход: 784 признака, изображение 28x28;
- два скрытых dense-слоя;
- `tanh` activation;
- dropout на первом скрытом слое во время обучения;
- softmax output на 10 классов;
- сохранение и загрузка весов через HDF5.