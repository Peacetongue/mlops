import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse


def log_training_plots(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'], label='train')
    ax1.plot(history.history['val_accuracy'], label='validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(history.history['loss'], label='train')
    ax2.plot(history.history['val_loss'], label='validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    mlflow.log_figure(fig, "training_plots.png")
    plt.close(fig)


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

X_train, X_test = X_train / 255.0, X_test / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def build_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

model = build_cnn_model()

mlflow.set_experiment("neuro_experiment")

with mlflow.start_run(run_name="neuro_second"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    datagen = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    history = model.fit(datagen.flow(X_train, y_train, batch_size=args.batch_size),
          epochs=args.epochs,
          validation_data=(X_test, y_test))

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    f1 = f1_score(y_test, y_pred, average='micro')

    mlflow.log_params({
        "optimizer": "adam",
        "batch_size": args.batch_size,
        "epochs": args.epochs
    })

    mlflow.log_metrics({
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "f1": f1
    })

    log_training_plots(history)

    mlflow.keras.log_model(model, "cifar10_cnn")

    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    mlflow.log_dict(report, "classification_report.json")