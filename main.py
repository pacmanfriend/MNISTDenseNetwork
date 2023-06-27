import threading
import time
import numpy as np
from neural import DenseNetwork
from neural.tools import load_image_cv

import tkinter as tk
from tkinter import filedialog, messagebox

from multiprocessing import cpu_count

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def main():
    gui = GUI()
    gui.start()


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

        self.load_mnist_data()

        self.buttons_frame = tk.Frame(self.__root, width=50, pady=10, padx=10)
        self.result_frame = tk.Frame(self.__root, width=50, padx=10, pady=10)
        self.max_params_frame = tk.Frame(self.__root, padx=5, pady=5)
        self.train_params_frame = tk.Frame(self.__root, padx=5, pady=5)

        self.load_img_btn = tk.Button(self.buttons_frame, text="Загрузить изображение", padx=2, pady=2, width=30, height=1,
                                   bg='white', fg='black', command=self.load_image_from_filesystem)
        self.create_model_btn = tk.Button(self.buttons_frame, text="Создать нейросеть", padx=2, pady=2,
                                       width=30, height=1, bg='white', fg='black', command=self.create_model)
        self.train_model_btn = tk.Button(self.buttons_frame, text="Начать обучение нейросети", padx=2, pady=2,
                                      width=30, height=1, bg='white', fg='black', command=self.train_model)
        self.save_model_btn = tk.Button(self.buttons_frame, text="Сохранить нейросеть", padx=2, pady=2,
                                     width=30, height=1, bg='white', fg='black', command=self.save_model)
        self.show_model_info_btn = tk.Button(self.buttons_frame, text='Показать информацию о нейросети', padx=2, pady=2,
                                          width=30, height=1, bg='white', fg='black', command=self.show_model_info)
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

        def make_param_row(parent, label_text, default_value):
            frame = tk.Frame(parent, padx=5, pady=2)
            tk.Label(frame, text=label_text, font=("Arial", 10), width=26, anchor=tk.W).pack(side=tk.LEFT)
            entry = tk.Entry(frame, width=8)
            entry.insert(0, str(default_value))
            entry.pack(side=tk.RIGHT)
            frame.pack(anchor=tk.NW, fill=tk.X)
            return entry

        self.train_size_entry  = make_param_row(self.train_params_frame, "Размер обучающих данных", 60000)
        self.threads_entry     = make_param_row(self.train_params_frame, "Количество потоков", cpu_count())
        self.epochs_entry      = make_param_row(self.train_params_frame, "Количество эпох", 40)
        self.batch_size_entry  = make_param_row(self.train_params_frame, "Размер батча", 16)
        self.alpha_entry       = make_param_row(self.train_params_frame, "Скорость обучения (alpha)", 0.1)

        self.image_plot = None
        self.canvas = None

    def start(self):
        self.__root.title("Image Classifier")
        self.__root.geometry("1280x720+50+50")

        self.buttons_frame.pack(anchor=tk.NW, side=tk.LEFT)

        self.max_params_frame.pack(anchor=tk.NW, side=tk.TOP, expand=tk.FALSE)
        self.max_threads_count_label.pack(anchor=tk.NW, side=tk.TOP)
        self.max_train_size_label.pack(anchor=tk.NW, side=tk.TOP)

        self.train_params_frame.pack(anchor=tk.NW, side=tk.TOP, expand=tk.FALSE)

        self.load_img_btn.pack(anchor=tk.NW)
        self.load_model_btn.pack(anchor=tk.NW)
        self.create_model_btn.pack(anchor=tk.NW)
        self.train_model_btn.pack(anchor=tk.NW)
        self.save_model_btn.pack(anchor=tk.NW)
        self.show_model_info_btn.pack(anchor=tk.NW)
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
        if model_path:
            self.model = DenseNetwork()
            self.model.load_model(model_path)

    def predict(self, event=None):
        if self.model is None:
            self.__root.title("Image Classifier — сначала загрузите нейросеть")
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
        if filepath:
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
        mnist_win = tk.Toplevel(self.__root)
        mnist_win.geometry('1280x720+50+50')

        fig = plt.figure(figsize=(10, 10))
        for i in range(36):
            image_plot = fig.add_subplot(6, 6, i + 1)
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
            self.__root.title("Image Classifier — нет нейросети для сохранения")
            return
        filepath = filedialog.asksaveasfilename()
        if filepath:
            self.model.save_model(filepath)

    def _get_train_params(self):
        errors = []

        try:
            train_size = int(self.train_size_entry.get())
            if train_size <= 0 or train_size > self.max_train_size:
                errors.append(f"Размер данных: от 1 до {self.max_train_size}")
        except ValueError:
            errors.append("Размер данных: целое число")
            train_size = None

        try:
            num_workers = int(self.threads_entry.get())
            if not (1 <= num_workers <= cpu_count()):
                errors.append(f"Количество потоков: от 1 до {cpu_count()}")
        except ValueError:
            errors.append("Количество потоков: целое число")
            num_workers = None

        try:
            epochs = int(self.epochs_entry.get())
            if epochs <= 0:
                errors.append("Количество эпох: положительное целое число")
        except ValueError:
            errors.append("Количество эпох: целое число")
            epochs = None

        try:
            batch_size = int(self.batch_size_entry.get())
            if batch_size <= 0:
                errors.append("Размер батча: положительное целое число")
        except ValueError:
            errors.append("Размер батча: целое число")
            batch_size = None

        try:
            alpha = float(self.alpha_entry.get())
            if alpha <= 0:
                errors.append("Alpha: положительное число")
        except ValueError:
            errors.append("Alpha: число (например, 0.1)")
            alpha = None

        if errors:
            messagebox.showerror("Ошибка параметров", "\n".join(errors))
            return None
        return train_size, epochs, batch_size, alpha, num_workers

    def train_model(self):
        if self.model is None:
            messagebox.showerror("Ошибка", "Загрузите или создайте нейросеть!")
            return

        params = self._get_train_params()
        if params is None:
            return
        train_size, epochs, batch_size, alpha, num_workers = params

        images = self.x_train[:train_size].reshape(train_size, 28 * 28).astype(np.float32) / 255.0
        labels = np.eye(10, dtype=np.float32)[self.y_train[:train_size]]
        test_images = self.x_test.reshape(len(self.x_test), 28 * 28).astype(np.float32) / 255.0
        test_labels = np.eye(10, dtype=np.float32)[self.y_test]

        self.train_model_btn.config(state=tk.DISABLED)

        def run():
            start_time = time.monotonic()
            self.model.fit(images, labels, batch_size=batch_size, epochs=epochs, validation_split=0.1, alpha=alpha, num_workers=num_workers)
            elapsed = time.monotonic() - start_time
            acc = self.model.evaluate(test_images, test_labels, verbose=True)
            print(f"Training time: {elapsed:.2f}s  Test accuracy: {acc:.4f}")
            self.__root.after(0, lambda: self._on_train_done(acc))

        threading.Thread(target=run, daemon=True).start()

    def _on_train_done(self, acc: float):
        self.train_model_btn.config(state=tk.NORMAL)
        messagebox.showinfo("Обучение", f"Нейронная сеть обучена!\nТочность на тесте: {acc:.4f}")

    def show_model_info(self):
        if self.model is None:
            messagebox.showinfo("Информация", "Нейросеть не загружена.")
            return
        w01 = self.model.weights_0_1
        w12 = self.model.weights_1_2
        w23 = self.model.weights_2_3
        if w01 is not None and w12 is not None and w23 is not None:
            total = w01.size + w12.size + w23.size
            info = (
                f"Архитектура:\n"
                f"  Вход:      {self.model.input_size} нейронов\n"
                f"  Скрытый 1: {self.model.hidden_size} нейронов (tanh)\n"
                f"  Скрытый 2: {self.model.hidden_size} нейронов (tanh)\n"
                f"  Выход:     {self.model.output_size} нейронов (softmax)\n\n"
                f"Веса:\n"
                f"  weights_0_1: {w01.shape}\n"
                f"  weights_1_2: {w12.shape}\n"
                f"  weights_2_3: {w23.shape}\n"
                f"  Всего параметров: {total:,}"
            )
        else:
            info = "Нейросеть создана, но веса не инициализированы."
        messagebox.showinfo("Информация о нейросети", info)


if __name__ == '__main__':
    main()
