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

        self.buttons_frame      = tk.Frame(self.__root, width=50, pady=10, padx=10)
        self.result_frame       = tk.Frame(self.__root, width=50, padx=10, pady=10)
        self.max_params_frame   = tk.Frame(self.__root, padx=5, pady=5)
        self.train_params_frame = tk.Frame(self.__root, padx=5, pady=5)
        self.status_frame       = tk.Frame(self.__root, padx=5, pady=2)

        self.load_img_btn = tk.Button(
            self.buttons_frame, text="Загрузить изображение",
            padx=2, pady=2, width=30, height=1, bg='white', fg='black',
            command=self.load_image_from_filesystem)
        self.create_model_btn = tk.Button(
            self.buttons_frame, text="Создать нейросеть",
            padx=2, pady=2, width=30, height=1, bg='white', fg='black',
            command=self.create_model)
        self.train_model_btn = tk.Button(
            self.buttons_frame, text="Начать обучение нейросети",
            padx=2, pady=2, width=30, height=1, bg='white', fg='black',
            state=tk.DISABLED, command=self.train_model)
        self.save_model_btn = tk.Button(
            self.buttons_frame, text="Сохранить нейросеть",
            padx=2, pady=2, width=30, height=1, bg='white', fg='black',
            command=self.save_model)
        self.show_model_info_btn = tk.Button(
            self.buttons_frame, text='Показать информацию о нейросети',
            padx=2, pady=2, width=30, height=1, bg='white', fg='black',
            command=self.show_model_info)
        self.load_model_btn = tk.Button(
            self.buttons_frame, text='Загрузить нейросеть',
            padx=2, pady=2, width=30, height=1, bg='white', fg='black',
            command=self.load_model)
        self.predict_btn = tk.Button(
            self.buttons_frame, text='Определить цифру',
            padx=2, pady=2, width=30, height=1, bg='white', fg='black',
            command=self.predict)
        self.show_train_btn = tk.Button(
            self.buttons_frame, text="Тренировочные данные",
            padx=2, pady=2, width=30, height=1, bg='white', fg='black',
            state=tk.DISABLED, command=self.show_mnist)

        self.result_label = tk.Label(self.result_frame, text="", font=("Arial", 16))

        self.max_threads_count_label = tk.Label(
            self.max_params_frame,
            text=f"Доступное количество потоков: {cpu_count()}",
            font=("Arial", 10))
        self.max_train_size_label = tk.Label(
            self.max_params_frame,
            text="Доступный размер обучающих данных: —",
            font=("Arial", 10))

        self.status_label = tk.Label(
            self.status_frame, text="Загрузка данных MNIST…",
            font=("Arial", 9), fg='gray', anchor=tk.W)

        def make_param_row(parent, label_text, default_value):
            frame = tk.Frame(parent, padx=5, pady=2)
            tk.Label(frame, text=label_text, font=("Arial", 10), width=26, anchor=tk.W).pack(side=tk.LEFT)
            entry = tk.Entry(frame, width=8)
            entry.insert(0, str(default_value))
            entry.pack(side=tk.RIGHT)
            frame.pack(anchor=tk.NW, fill=tk.X)
            return entry

        self.train_size_entry = make_param_row(self.train_params_frame, "Размер обучающих данных", 60000)
        self.threads_entry    = make_param_row(self.train_params_frame, "Количество потоков", cpu_count())
        self.epochs_entry     = make_param_row(self.train_params_frame, "Количество эпох", 40)
        self.batch_size_entry = make_param_row(self.train_params_frame, "Размер батча", 16)
        self.alpha_entry      = make_param_row(self.train_params_frame, "Скорость обучения (alpha)", 0.1)

        self.image_plot = None
        self.canvas = None

    def start(self):
        self.__root.title("Image Classifier")
        self.__root.geometry("1280x720+50+50")

        self.buttons_frame.pack(anchor=tk.NW, side=tk.LEFT, fill=tk.Y)

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

        self.status_frame.pack(anchor=tk.NW, side=tk.TOP, fill=tk.X)
        self.status_label.pack(anchor=tk.NW, fill=tk.X)

        fig = plt.figure(figsize=(6, 4))
        self.image_plot = fig.add_subplot()
        self.canvas = FigureCanvasTkAgg(fig, master=self.__root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(anchor=tk.NE, side=tk.TOP)

        self.result_frame.pack(anchor=tk.NE)
        self.result_label.pack(anchor=tk.NE)

        self.__root.after(50, self._load_mnist_async)
        self.__root.mainloop()

    # ------------------------------------------------------------------ data

    def _load_mnist_async(self):
        def load():
            from tensorflow.keras.datasets import mnist
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            self.__root.after(0, lambda: self._on_mnist_loaded(x_train, y_train, x_test, y_test))
        threading.Thread(target=load, daemon=True).start()

    def _on_mnist_loaded(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.x_test  = x_test
        self.y_train = y_train
        self.y_test  = y_test
        self.max_train_size = len(x_train)
        self.max_train_size_label.config(
            text=f"Доступный размер обучающих данных: {self.max_train_size}")
        self.train_model_btn.config(state=tk.NORMAL)
        self.show_train_btn.config(state=tk.NORMAL)
        self._set_status("Готово")

    # ------------------------------------------------------------------ ui helpers

    def _set_status(self, text: str):
        self.status_label.config(text=text)

    # ------------------------------------------------------------------ model actions

    def load_model(self, event=None):
        model_path = filedialog.askopenfilename()
        if model_path:
            self.model = DenseNetwork()
            self.model.load_model(model_path)
            self._set_status("Нейросеть загружена")

    def create_model(self, event=None):
        self.model = DenseNetwork()
        self.model.init_weights()
        self._set_status("Нейросеть создана")

    def save_model(self):
        if self.model is None:
            messagebox.showwarning("Предупреждение", "Нет нейросети для сохранения.")
            return
        filepath = filedialog.asksaveasfilename()
        if filepath:
            self.model.save_model(filepath)
            self._set_status("Нейросеть сохранена")

    def show_model_info(self):
        if self.model is None:
            messagebox.showinfo("Информация", "Нейросеть не загружена.")
            return
        info = self.model.get_info()
        if not info['initialized']:
            text = "Нейросеть создана, но веса не инициализированы."
        else:
            shapes = info['shapes']
            text = (
                f"Архитектура:\n"
                f"  Вход:      {info['input_size']} нейронов\n"
                f"  Скрытый 1: {info['hidden_size']} нейронов (tanh)\n"
                f"  Скрытый 2: {info['hidden_size']} нейронов (tanh)\n"
                f"  Выход:     {info['output_size']} нейронов (softmax)\n\n"
                f"Веса:\n"
                f"  weights_0_1: {shapes['weights_0_1']}\n"
                f"  weights_1_2: {shapes['weights_1_2']}\n"
                f"  weights_2_3: {shapes['weights_2_3']}\n"
                f"  Всего параметров: {info['total_params']:,}"
            )
        messagebox.showinfo("Информация о нейросети", text)

    # ------------------------------------------------------------------ training

    def _get_train_params(self):
        if self.x_train is None:
            messagebox.showerror("Ошибка", "Данные MNIST ещё не загружены.")
            return None

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

        images      = self.x_train[:train_size].reshape(train_size, 28 * 28).astype(np.float32) / 255.0
        labels      = np.eye(10, dtype=np.float32)[self.y_train[:train_size]]
        test_images = self.x_test.reshape(len(self.x_test), 28 * 28).astype(np.float32) / 255.0
        test_labels = np.eye(10, dtype=np.float32)[self.y_test]

        self.train_model_btn.config(state=tk.DISABLED)
        self._set_status("Обучение…")

        def on_epoch_end(epoch: int, train_acc: float, val_acc: float):
            self.__root.after(
                0,
                lambda e=epoch, ta=train_acc, va=val_acc: self._set_status(
                    f"Эпоха {e + 1}/{epochs} | Train: {ta:.3f} | Val: {va:.3f}"
                ),
            )

        def run():
            start_time = time.monotonic()
            self.model.fit(
                images, labels,
                batch_size=batch_size, epochs=epochs,
                validation_split=0.1, alpha=alpha,
                num_workers=num_workers, on_epoch_end=on_epoch_end,
            )
            elapsed = time.monotonic() - start_time
            acc = self.model.evaluate(test_images, test_labels, verbose=True)
            print(f"Training time: {elapsed:.2f}s  Test accuracy: {acc:.4f}")
            self.__root.after(0, lambda: self._on_train_done(acc, elapsed))

        threading.Thread(target=run, daemon=True).start()

    def _on_train_done(self, acc: float, elapsed: float):
        self.train_model_btn.config(state=tk.NORMAL)
        self._set_status(f"Обучение завершено | Test acc: {acc:.4f} | {elapsed:.1f}s")
        messagebox.showinfo("Обучение", f"Нейронная сеть обучена!\nТочность на тесте: {acc:.4f}")

    # ------------------------------------------------------------------ image / predict

    def load_image_from_filesystem(self, event=None):
        filepath = filedialog.askopenfilename()
        if not filepath:
            return
        try:
            self.grayscale_image = load_image_cv(filepath)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение:\n{e}")
            return
        self.image_plot.imshow(self.grayscale_image, cmap='gray')
        self.canvas.draw()

    def predict(self, event=None):
        if self.model is None:
            self._set_status("Сначала загрузите нейросеть")
            return
        if self.grayscale_image is None:
            self._set_status("Сначала загрузите изображение")
            return
        x = (self.grayscale_image.astype(np.float32) / 255.0).reshape(1, 28 * 28)
        pred = int(self.model.predict(x)[0])
        self.result_label['text'] = f'Результат: {pred}'
        self.__root.title(f"Image Classifier — Predicted: {pred}")

    # ------------------------------------------------------------------ mnist viewer

    def show_mnist(self, event=None):
        if self.x_train is None:
            return
        win = tk.Toplevel(self.__root)
        win.title("Тренировочные данные MNIST")
        win.geometry('900x900+100+50')

        fig = plt.figure(figsize=(10, 10))
        win.protocol("WM_DELETE_WINDOW", lambda: (plt.close(fig), win.destroy()))

        for i in range(36):
            ax = fig.add_subplot(6, 6, i + 1)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(self.x_train[i].reshape((28, 28)), cmap='gray')

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == '__main__':
    main()
