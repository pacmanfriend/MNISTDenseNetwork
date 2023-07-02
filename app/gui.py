from __future__ import annotations

import threading

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox
from multiprocessing import cpu_count

from app.config import TrainingConfig
from app.datasets import load_mnist
from app.preprocessing import prepare_digit_image
from app.state import AppState
from app.trainer import TrainingController
from neural import DenseNetwork
from neural.tools import load_image_cv


class GUI:
    def __init__(self, state: AppState, trainer: TrainingController):
        self._state   = state
        self._trainer = trainer

        self.__root = tk.Tk()
        self._grayscale_image: np.ndarray | None = None
        self._image_plot = None
        self._canvas     = None

        self._build_widgets()

    # ------------------------------------------------------------------ build

    def _build_widgets(self):
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

    # ------------------------------------------------------------------ start

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
        self._image_plot = fig.add_subplot()
        self._canvas = FigureCanvasTkAgg(fig, master=self.__root)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(anchor=tk.NE, side=tk.TOP)

        self.result_frame.pack(anchor=tk.NE)
        self.result_label.pack(anchor=tk.NE)

        self.__root.after(50, self._load_mnist_async)
        self.__root.mainloop()

    # ------------------------------------------------------------------ data loading

    def _load_mnist_async(self):
        def load():
            try:
                self._state.mnist = load_mnist()
            except Exception as exc:
                self.__root.after(0, lambda e=exc: self._on_mnist_load_error(e))
            else:
                self.__root.after(0, self._on_mnist_loaded)
        threading.Thread(target=load, daemon=True).start()

    def _on_mnist_loaded(self):
        self.max_train_size_label.config(
            text=f"Доступный размер обучающих данных: {self._state.max_train_size}")
        self.train_model_btn.config(state=tk.NORMAL)
        self.show_train_btn.config(state=tk.NORMAL)
        self._set_status("Готово")

    def _on_mnist_load_error(self, exc: Exception):
        self._set_status("Ошибка загрузки MNIST")
        messagebox.showerror("Ошибка загрузки MNIST", str(exc))

    # ------------------------------------------------------------------ ui helpers

    def _set_status(self, text: str):
        self.status_label.config(text=text)

    def _get_train_params(self):
        try:
            train_size = int(self.train_size_entry.get())
        except ValueError:
            messagebox.showerror("Ошибка параметров", "Размер данных: целое число")
            return None

        try:
            num_workers = int(self.threads_entry.get())
        except ValueError:
            messagebox.showerror("Ошибка параметров", "Количество потоков: целое число")
            return None

        try:
            epochs = int(self.epochs_entry.get())
        except ValueError:
            messagebox.showerror("Ошибка параметров", "Количество эпох: целое число")
            return None

        try:
            batch_size = int(self.batch_size_entry.get())
        except ValueError:
            messagebox.showerror("Ошибка параметров", "Размер батча: целое число")
            return None

        try:
            alpha = float(self.alpha_entry.get())
        except ValueError:
            messagebox.showerror("Ошибка параметров", "Alpha: число (например, 0.1)")
            return None

        config = TrainingConfig(
            train_size=train_size,
            epochs=epochs,
            batch_size=batch_size,
            alpha=alpha,
            num_workers=num_workers,
        )
        errors = config.validate(self._state.max_train_size, cpu_count())
        if errors:
            messagebox.showerror("Ошибка параметров", "\n".join(errors))
            return None
        return config

    # ------------------------------------------------------------------ model actions

    def load_model(self, event=None):
        model_path = filedialog.askopenfilename()
        if model_path:
            self._state.model = DenseNetwork()
            self._state.model.load_model(model_path)
            self._set_status("Нейросеть загружена")

    def create_model(self, event=None):
        self._state.model = DenseNetwork()
        self._state.model.init_weights()
        self._set_status("Нейросеть создана")

    def save_model(self):
        if self._state.model is None:
            messagebox.showwarning("Предупреждение", "Нет нейросети для сохранения.")
            return
        filepath = filedialog.asksaveasfilename()
        if filepath:
            self._state.model.save_model(filepath)
            self._set_status("Нейросеть сохранена")

    def show_model_info(self):
        if self._state.model is None:
            messagebox.showinfo("Информация", "Нейросеть не загружена.")
            return
        info = self._state.model.get_info()
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

    def train_model(self):
        if self._state.model is None:
            messagebox.showerror("Ошибка", "Загрузите или создайте нейросеть!")
            return
        if self._trainer.is_running:
            return

        params = self._get_train_params()
        if params is None:
            return
        config = params

        self.train_model_btn.config(state=tk.DISABLED)
        self._set_status("Обучение…")

        def on_epoch_end(epoch: int, train_acc: float, val_acc: float):
            self.__root.after(
                0,
                lambda e=epoch, ta=train_acc, va=val_acc: self._set_status(
                    f"Эпоха {e + 1}/{config.epochs} | Train: {ta:.3f} | Val: {va:.3f}"
                ),
            )

        def on_done(acc: float, elapsed: float):
            self.__root.after(0, lambda: self._on_train_done(acc, elapsed))

        def on_error(exc: Exception):
            self.__root.after(0, lambda e=exc: self._on_train_error(e))

        try:
            self._trainer.start(
                config=config,
                on_epoch_end=on_epoch_end, on_done=on_done, on_error=on_error,
            )
        except Exception as exc:
            self._on_train_error(exc)

    def _on_train_done(self, acc: float, elapsed: float):
        self.train_model_btn.config(state=tk.NORMAL)
        self._set_status(f"Обучение завершено | Test acc: {acc:.4f} | {elapsed:.1f}s")
        messagebox.showinfo("Обучение", f"Нейронная сеть обучена!\nТочность на тесте: {acc:.4f}")

    def _on_train_error(self, exc: Exception):
        self.train_model_btn.config(state=tk.NORMAL)
        self._set_status("Ошибка обучения")
        messagebox.showerror("Ошибка обучения", str(exc))

    # ------------------------------------------------------------------ image / predict

    def load_image_from_filesystem(self, event=None):
        filepath = filedialog.askopenfilename()
        if not filepath:
            return
        try:
            self._grayscale_image = load_image_cv(filepath)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение:\n{e}")
            return
        self._image_plot.imshow(self._grayscale_image, cmap='gray')
        self._canvas.draw()

    def predict(self, event=None):
        if self._state.model is None:
            self._set_status("Сначала загрузите нейросеть")
            return
        if self._grayscale_image is None:
            self._set_status("Сначала загрузите изображение")
            return
        x = prepare_digit_image(self._grayscale_image)
        pred = int(self._state.model.predict(x)[0])
        self.result_label['text'] = f'Результат: {pred}'
        self.__root.title(f"Image Classifier — Predicted: {pred}")

    # ------------------------------------------------------------------ mnist viewer

    def show_mnist(self, event=None):
        if not self._state.data_loaded:
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
            ax.imshow(self._state.mnist.x_train[i].reshape((28, 28)), cmap='gray')

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
