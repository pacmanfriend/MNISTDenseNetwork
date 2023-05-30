import argparse
import numpy as np
from neural import DenseNetwork
from neural.tools import image_downsample, load_image


def main():
    p = argparse.ArgumentParser(description="Train/evaluate DenseNetwork and optionally classify an image.")
    p.add_argument("--model-path", type=str, default="models/dense_mnist.h5")
    p.add_argument("--image", type=str, default=None, help="Path to an image to classify (optional).")
    p.add_argument("--train-size", type=int, default=60000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.2)
    args = p.parse_args()
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

if __name__ == '__main__':
    main()
