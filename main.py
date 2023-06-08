import argparse
import time
import numpy as np
from neural import DenseNetwork
from neural.tools import load_image_cv

import tkinter as tk
from tkinter import filedialog

from tkinter import messagebox

from multiprocessing import cpu_count, current_process

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def main():
    p = argparse.ArgumentParser(description="MNIST dense network (GUI or CLI).")
    sub = p.add_subparsers(dest="mode")

    # GUI mode (default)
    sub.add_parser("gui", help="Start Tkinter GUI (default).")

    # CLI training/evaluation mode
    train_p = sub.add_parser("train", help="Train/evaluate and optionally classify an image.")
    train_p.add_argument("--model-path", type=str, default="models/dense_mnist.h5")
    train_p.add_argument("--image", type=str, default=None, help="Path to an image to classify (optional).")
    train_p.add_argument("--train-size", type=int, default=60000)
    train_p.add_argument("--batch-size", type=int, default=32)
    train_p.add_argument("--epochs", type=int, default=50)
    train_p.add_argument("--alpha", type=float, default=0.2)

    args = p.parse_args()

    if args.mode in (None, "gui"):
        gui = GUI()
        gui.load_mnist_data()
        gui.start()
        return

    if args.mode == "train":
        train(args)
        return

    raise RuntimeError(f"Unknown mode: {args.mode}")


def train(args) -> None:
    data_size = args.train_size
    model_path = args.model_path

    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    images = x_train[:data_size].reshape(data_size, 28 * 28).astype(np.float32) / 255.0
    labels = np.eye(10, dtype=np.float32)[y_train[:data_size]]

    test_images = x_test.reshape(len(x_test), 28 * 28).astype(np.float32) / 255.0
    test_labels = np.eye(10, dtype=np.float32)[y_test]

    model = DenseNetwork()
    model.init_weights()
    model.fit(x_train=images, y_train=labels, batch_size=args.batch_size, epochs=args.epochs, alpha=args.alpha)
    test_acc = model.evaluate(x_test=test_images, y_test=test_labels, verbose=True)
    print(f"Final test accuracy: {test_acc:.4f}")

    model.save_model(model_path)

    loaded = DenseNetwork()
    loaded.load_model(model_path)
    loaded_acc = loaded.evaluate(x_test=test_images, y_test=test_labels, verbose=False)
    print(f"Loaded model test accuracy: {loaded_acc:.4f}")

    if args.image:
        img28 = load_image_cv(args.image).astype(np.float32) / 255.0
        x = img28.reshape(1, 28 * 28)
        pred = int(loaded.predict(x)[0])
        print(f"Predicted digit: {pred}")

class GUI:
    def __init__(self):
        self.__root = tk.Tk()

        self.grayscale_image: np.ndarray = None
        self.model: DenseNetwork = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.max_train_size = 0
        self.max_test_size = 0

        self.result = tk.StringVar()
        self.threads_count = tk.IntVar()

        self.load_mnist_data()

        self.buttons_frame = tk.Frame(self.__root, width=50, pady=10, padx=10)
        self.result_frame = tk.Frame(self.__root, width=50, padx=10, pady=10)
        self.max_params_frame = tk.Frame(self.__root, padx=5, pady=5)
        self.train_frame = tk.Frame(self.__root, padx=5, pady=5)
        self.threads_counter_frame = tk.Frame(self.__root, padx=5, pady=5)

        self.load_img_btn = tk.Button(self.buttons_frame, text="Загрузить изображение", padx=2, pady=2, width=30, height=1,
                                   bg='white', fg='black', command=self.load_image_from_filesystem)
        self.create_model_btn = tk.Button(self.buttons_frame, text="Создать нейросеть", padx=2, pady=2,
                                       width=30, height=1, bg='white', fg='black', command=self.create_model)
        self.train_model_btn = tk.Button(self.threads_counter_frame, text="Начать обучение нейросети", padx=2, pady=2,
                                      width=30, height=1, bg='white', fg='black', command=self.train_model)
        self.save_model_btn = tk.Button(self.buttons_frame, text="Сохранить нейросеть", padx=2, pady=2,
                                     width=30, height=1, bg='white', fg='black', command=self.save_model)
        self.show_model_info_btn = tk.Button(self.buttons_frame, text='Показать информацию о нейросети', padx=2, pady=2,
                                          width=30, height=1, bg='white', fg='black')

        self.load_model_btn = tk.Button(self.buttons_frame, text='Загрузить нейросеть', padx=2, pady=2,
                                     width=30, height=1, bg='white', fg='black', command=self.load_model)
        self.predict_btn = tk.Button(self.buttons_frame, text='Определить цифру', padx=2, pady=2,
                                  width=30, height=1, bg='white', fg='black', command=self.predict)

        self.show_train_btn = tk.Button(self.buttons_frame, text="Тренировочные данные", padx=2, pady=2,
                                     width=30, height=1, bg='white', fg='black', command=self.show_mnist)

        self.result_label = tk.Label(self.result_frame, text="", font=("Arial", 16))

        self.max_threads_count_label = tk.Label(self.max_params_frame,
                                             text=f"Доступное количество потоков: {cpu_count()}",
                                             font=("Arial", 10))

        self.max_train_size_label = tk.Label(self.max_params_frame,
                                          text=f"Доступный размер обучающих данных: {len(self.x_train)}",
                                          font=("Arial", 10))

        self.train_size_label = tk.Label(self.train_frame, text="Размер обучающих данных", font=("Arial", 10))
        self.train_size_entry = tk.Entry(self.train_frame)

        self.threads_count_label = tk.Label(self.threads_counter_frame, text="Количество потоков", font=("Arial", 10))
        self.threads_count_entry = tk.Entry(self.threads_counter_frame)

        self.image_plot = None
        self.canvas = None

    def start(self):
        self.__root.title("Image Classifier")
        self.__root.geometry("1280x720+50+50")

        self.buttons_frame.pack(anchor=tk.NW, side=tk.LEFT)

        self.max_params_frame.pack(anchor=tk.NW, side=tk.TOP, expand=tk.FALSE)
        self.max_threads_count_label.pack(anchor=tk.NW, side=tk.TOP)
        self.max_train_size_label.pack(anchor=tk.NW, side=tk.TOP)
        self.train_model_btn.pack(anchor=tk.NW, side=tk.BOTTOM)

        self.train_frame.pack(anchor=tk.NW, side=tk.TOP, expand=tk.FALSE)
        self.train_size_label.pack(anchor=tk.NW, side=tk.LEFT)
        self.train_size_entry.pack(anchor=tk.NE, side=tk.RIGHT)

        self.threads_counter_frame.pack(anchor=tk.NW, side=tk.TOP, expand=tk.FALSE)
        self.threads_count_label.pack(anchor=tk.NW, side=tk.LEFT)
        self.threads_count_entry.pack(anchor=tk.NE, side=tk.RIGHT)

        self.load_img_btn.pack(anchor=tk.NW)
        self.load_model_btn.pack(anchor=tk.NW)
        self.create_model_btn.pack(anchor=tk.NW)
        self.save_model_btn.pack(anchor=tk.NW)
        self.predict_btn.pack(anchor=tk.NW)
        self.show_train_btn.pack(anchor=tk.NW)

        fig = plt.figure(figsize=(6, 4))
        self.image_plot = fig.add_subplot()
        self.canvas = FigureCanvasTkAgg(fig, master=self.__root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(anchor=tk.NE, side=tk.TOP)

        self.result_frame.pack(anchor=tk.NE)
        self.result_label.pack(anchor=tk.NE)

        self.__root.mainloop()

    def load_model(self, event=None):
        model_path = filedialog.askopenfilename()

        if model_path != "":
            self.model = DenseNetwork()
            self.model.load_model(model_path)

    def predict(self, event=None):
        if self.model is None:
            self.__root.title("Image Classifier — сначала загрузите модель")
            return
        if self.grayscale_image is None:
            self.__root.title("Image Classifier — сначала загрузите изображение")
            return

        x = (self.grayscale_image.astype(np.float32) / 255.0).reshape(1, 28 * 28)
        pred = int(self.model.predict(x)[0])

        self.result_label['text'] = f'Результат: {pred}'

        self.__root.title(f"Image Classifier — Predicted: {pred}")

    def load_image_from_filesystem(self, event=None):
        filepath = filedialog.askopenfilename()

        if filepath != "":
            self.grayscale_image = load_image_cv(filepath)

            self.image_plot.imshow(self.grayscale_image)

            self.canvas.draw()
            self.canvas.get_tk_widget().pack(anchor=tk.NE)

    def load_mnist_data(self):
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.max_train_size = len(self.x_train)
        self.max_test_size = len(self.x_test)

    def show_mnist(self, event=None):
        mnist_win = tk.Tk()
        mnist_win.geometry('1280x720+50+50')

        fig = plt.figure(figsize=(10, 10))
        for i in range(36):
            image_plot = fig.add_subplot(6, 6, i + 1)
            # image_plot.xticks([])
            # image_plot.yticks([])
            image_plot.grid(False)
            image_plot.imshow(self.x_train[i].reshape((28, 28)))

        canvas = FigureCanvasTkAgg(fig, master=mnist_win)
        canvas.draw()
        canvas.get_tk_widget().pack(anchor=tk.CENTER)

    def create_model(self, event=None):
        self.model = DenseNetwork()
        self.model.init_weights()

    def save_model(self):
        if self.model is None:
            self.__root.title("Image Classifier — нет модели для сохранения")
            return
        filepath = filedialog.asksaveasfilename()

        if filepath != "":
            self.model.save_model(filepath)

    def train_model(self):
        if self.model is None:
            messagebox.showerror("Ошибка", "Загрузите или создайте модель!")
            return

        try:
            train_size = int(self.train_size_entry.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Укажите корректный размер обучающих данных!")
            return

        if train_size <= 0 or train_size > self.max_train_size:
            messagebox.showerror("Ошибка", f"Размер должен быть от 1 до {self.max_train_size}")
            return

        images = self.x_train[:train_size].reshape(train_size, 28 * 28).astype(np.float32) / 255.0

        labels = np.zeros((train_size, 10), dtype=np.float32)
        for i, j in enumerate(self.y_train[:train_size]):
            labels[i][j] = 1

        start_time = time.monotonic()
        self.model.fit(images, labels, batch_size=16, epochs=40, validation_split=0.1, alpha=0.1)
        print(f"Training time: {time.monotonic() - start_time:.2f}s")

        test_images = self.x_test.reshape(len(self.x_test), 28 * 28).astype(np.float32) / 255.0
        test_labels = np.zeros((len(self.y_test), 10), dtype=np.float32)
        for i, j in enumerate(self.y_test):
            test_labels[i][j] = 1

        acc = self.model.evaluate(test_images, test_labels, verbose=True)
        print(f"Test accuracy: {acc:.4f}")

        messagebox.showinfo("Обучение", "Нейронная сеть обучена!")

    def show_model_info(self):
        pass

def print_proc(a, b, c, d, i):
    g = current_process().name

    print(f"{g} Epoch: {i} Train-Err: {a} Train-Acc: {b} Validation-Err: {c} Validation-Acc: {d}")

if __name__ == '__main__':
    main()
