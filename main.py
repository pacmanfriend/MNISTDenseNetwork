import numpy as np
from neural import DenseNetwork


def main():
    data_size = 60000
    model_path = "models/dense_mnist.h5"

    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    images = x_train[:data_size].reshape(data_size, 28 * 28).astype(np.float32) / 255.0
    labels = np.eye(10, dtype=np.float32)[y_train[:data_size]]

    test_images = x_test.reshape(len(x_test), 28 * 28).astype(np.float32) / 255.0
    test_labels = np.eye(10, dtype=np.float32)[y_test]

    model = DenseNetwork()
    model.init_weights()
    model.fit(x_train=images, y_train=labels, batch_size=32, epochs=50, alpha=0.2)
    test_acc = model.evaluate(x_test=test_images, y_test=test_labels, verbose=True)
    print(f"Final test accuracy: {test_acc:.4f}")

    model.save_model(model_path)

    loaded = DenseNetwork()
    loaded.load_model(model_path)
    loaded_acc = loaded.evaluate(x_test=test_images, y_test=test_labels, verbose=False)
    print(f"Loaded model test accuracy: {loaded_acc:.4f}")

if __name__ == '__main__':
    main()
