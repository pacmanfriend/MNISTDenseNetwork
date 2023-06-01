import argparse

import numpy as np
from neural import DenseNetwork
from neural.tools import image_downsample, load_image

import tkinter as tk
from tkinter import filedialog

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
        img = load_image(args.image)
        img28 = image_downsample(img)  # (28, 28) in [0,1]
        x = img28.reshape(1, 28 * 28).astype(np.float32)
        pred = int(loaded.predict(x)[0])
        print(f"Predicted digit: {pred}")

class GUI:
    def __init__(self):
        self.__root = tk.Tk()

        self.loaded_image = None
        self.downsampled_image: np.ndarray = None
        self.grayscale_image: np.ndarray = None
        self.model: DenseNetwork = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.buttons_frame = tk.Frame(self.__root, width=50, pady=10, padx=10)
        self.load_img_btn = tk.Button(
            self.buttons_frame,
            text="Загрузить изображение",
            padx=2,
            pady=2,
            width=30,
            height=1,
            bg="white",
            fg="black",
        )
        self.convert_to_grey_btn = tk.Button(
            self.buttons_frame,
            text="Нормализовать (0..1)",
            padx=2,
            pady=2,
            width=30,
            height=1,
            bg="white",
            fg="black",
        )

        self.load_model_btn = tk.Button(
            self.buttons_frame,
            text="Загрузить модель",
            padx=2,
            pady=2,
            width=30,
            height=1,
            bg="white",
            fg="black",
        )
        self.predict_btn = tk.Button(
            self.buttons_frame,
            text="Определить цифру",
            padx=2,
            pady=2,
            width=30,
            height=1,
            bg="white",
            fg="black",
        )

        self.show_train_btn = tk.Button(
            self.buttons_frame,
            text="Тренировочные данные",
            padx=2,
            pady=2,
            width=30,
            height=1,
            bg="white",
            fg="black",
        )

        self.image_plot = None
        self.canvas = None

    def start(self):
        self.__root.title("Image Classifier")
        self.__root.geometry("1280x720+50+50")

        self.load_img_btn.bind("<Button-1>", self.load_image_from_filesystem)
        self.convert_to_grey_btn.bind('<Button-1>', self.convert_to_grayscale)
        self.load_model_btn.bind('<Button-1>', self.load_model)
        self.predict_btn.bind('<Button-1>', self.get_result)
        self.show_train_btn.bind('<Button-1>', self.show_mnist)

        self.buttons_frame.pack(anchor=tk.NW)
        self.load_img_btn.pack(anchor=tk.NW)
        self.convert_to_grey_btn.pack(anchor=tk.NW)
        self.load_model_btn.pack(anchor=tk.NW)
        self.predict_btn.pack(anchor=tk.NW)
        self.show_train_btn.pack(anchor=tk.NW)

        fig = plt.figure(figsize=(6, 4))
        self.image_plot = fig.add_subplot()
        self.canvas = FigureCanvasTkAgg(fig, master=self.__root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(anchor=tk.NE)

        self.__root.mainloop()

    def load_model(self, event):
        model_path = filedialog.askopenfilename()

        if model_path != "":
            self.model = DenseNetwork()
            self.model.load_model(model_path)

    def get_result(self, event):
        if self.model is None:
            self.__root.title("Image Classifier — сначала загрузите модель")
            return
        if self.grayscale_image is None:
            self.__root.title("Image Classifier — сначала загрузите изображение")
            return

        x = self.grayscale_image.reshape(1, 28 * 28).astype(np.float32)
        pred = int(self.model.predict(x)[0])
        self.__root.title(f"Image Classifier — Predicted: {pred}")

    def load_image_from_filesystem(self, event):
        filepath = filedialog.askopenfilename()

        if filepath != "":
            self.loaded_image = load_image(filepath)
            self.downsampled_image = image_downsample(self.loaded_image)  # (28, 28) float in [0,1]
            self.grayscale_image = self.downsampled_image

            self.image_plot.imshow(self.downsampled_image)

            self.canvas.draw()
            self.canvas.get_tk_widget().pack(anchor=tk.NE)

    def convert_to_grayscale(self, event):
        if self.downsampled_image is None:
            return

        img = self.downsampled_image.astype(np.float32)
        vmin = float(img.min())
        vmax = float(img.max())
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        self.grayscale_image = img

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

    def show_mnist(self, event):
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


if __name__ == '__main__':
    main()
