import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from typing import Tuple

from SafeRLHF.config import (
    CLS_DATA_DIR, CLS_SAFE_DATA_FILE_NAME, CLS_SAFE_DATA_FILE,
    CLS_MODELS_DIR, CLS_MODELS_PLOTS_DIR,
    MAX_FEATURES, SEQUENCE_LENGTH, EMBEDDING_DIM,
    HIDDEN_DIM, DROPOUT,
    EPOCHS, BATCH_SIZE)

FEATURE_NAME: str = 'text'
TARGET_NAME: str = 'is_unsafe'


def get_combined_data(safe_df: pd.DataFrame, unsafe_df: pd.DataFrame) -> pd.DataFrame:
    combined_df = pd.concat([safe_df, unsafe_df], ignore_index=True)
    combined_df[FEATURE_NAME] = combined_df['prompt'] + combined_df['response']
    return combined_df[[FEATURE_NAME, TARGET_NAME]]


def get_train_test_split(combined_df: pd.DataFrame) -> Tuple[list, list,
                                                             list, list]:
    features = combined_df[FEATURE_NAME].values
    targets = combined_df[TARGET_NAME].values
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets,
        test_size=0.2, random_state=42,
        stratify=targets)
    return X_train, X_test, y_train, y_test


def get_vectorizer() -> tf.keras.layers.TextVectorization:
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_FEATURES,
        output_mode='int',
        output_sequence_length=SEQUENCE_LENGTH)
    return vectorize_layer


def get_model(vectorizer: tf.keras.layers.TextVectorization) -> tf.keras.Sequential:
    model = tf.keras.Sequential([
        vectorizer,
        tf.keras.layers.Embedding(MAX_FEATURES, EMBEDDING_DIM),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.LSTM(HIDDEN_DIM),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer='adam',
                  metrics=[tf.metrics.F1Score(threshold=0.5)])
    return model


def plot_loss_history(history: tf.keras.callbacks.History,
                      metrics_name: str,
                      test_metrics_val: float,
                      output_file: str):
    history_dict = history.history
    metrics_val = history_dict[metrics_name]
    val_metrics_val = history_dict[f'val_{metrics_name}']

    epochs = range(1, len(metrics_val) + 1)
    plt.plot(epochs, metrics_val, 'bo',
             label=f'Training {metrics_name}', color='blue')
    plt.plot(epochs, val_metrics_val, 'b', label=f'Validation {metrics_name}')
    plt.plot(epochs[-1], test_metrics_val, 'bo',
             label=f'Test {metrics_name}', color='red')
    plt.title(f'Training, validation and test {metrics_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metrics_name)
    plt.legend()

    plt.savefig(output_file, format='pdf')
    plt.close()


def main():
    safe_data = pd.read_csv(CLS_SAFE_DATA_FILE)
    safe_data[TARGET_NAME] = 0
    unsafe_files = [f for f in os.listdir(CLS_DATA_DIR)
                    if f.endswith('.csv') and f != CLS_SAFE_DATA_FILE_NAME]

    for unsafe_file in unsafe_files:
        unsafe_data = pd.read_csv(os.path.join(CLS_DATA_DIR, unsafe_file))
        unsafe_data[TARGET_NAME] = 1

        combined_data = get_combined_data(safe_data, unsafe_data)
        X_train, X_test, y_train, y_test = get_train_test_split(combined_data)

        vectorizer = get_vectorizer()
        vectorizer.adapt(X_train)

        model = get_model(vectorizer)
        history = model.fit(
            x=X_train, y=tf.expand_dims(y_train, -1),
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            epochs=EPOCHS)

        test_loss, test_f1 = model.evaluate(
            x=X_test, y=tf.expand_dims(y_test, -1),
            batch_size=BATCH_SIZE)

        safety_category = os.path.splitext(unsafe_file)[0]
        for metrics in [('loss', test_loss), ('f1_score', test_f1)]:
            plot_loss_history(history, metrics[0], metrics[1],
                              os.path.join(CLS_MODELS_PLOTS_DIR,
                                           f'{safety_category}_{metrics[0]}.pdf'))

        model.save(os.path.join(CLS_MODELS_DIR,
                                f'{safety_category}.keras'))


if __name__ == '__main__':
    main()
