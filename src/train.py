import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import build_model
from utils import set_seed

def fit_model(X_speech, X_text, y, max_length, embedding_dim, embedding_matrix, vocab_size):
    set_seed(42)

    outer_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_per_fold, uar_per_fold, loss_per_fold = [], [], []

    fold_no = 1
    for train_idx, test_idx in outer_folds.split(X_speech, y):
        print(f'\nTraining for fold {fold_no}...')

        X_train_speech, X_test_speech = X_speech[train_idx], X_speech[test_idx]
        X_train_text, X_test_text = X_text[train_idx], X_text[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_speech = scaler.fit_transform(X_train_speech)
        X_test_speech = scaler.transform(X_test_speech)
        X_train_speech = np.expand_dims(X_train_speech, axis=1)
        X_test_speech = np.expand_dims(X_test_speech, axis=1)

        model = build_model(max_length, embedding_dim, embedding_matrix, X_speech.shape[1], vocab_size)

        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, verbose=1),
            ModelCheckpoint(f'files/best_model_fold{fold_no}.h5', monitor='val_accuracy', save_best_only=True)
        ]

        history = model.fit(
            [X_train_speech, X_train_text], y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        scores = model.evaluate([X_test_speech, X_test_text], y_test, verbose=0)
        y_pred = model.predict([X_test_speech, X_test_text])
        y_pred = np.argmax(y_pred, axis=1)

        acc = scores[1]
        uar = recall_score(y_test, y_pred, average='macro')
        loss = scores[0]

        acc_per_fold.append(acc)
        uar_per_fold.append(uar)
        loss_per_fold.append(loss)

        print(f'> Fold {fold_no} - Accuracy: {acc:.4f} - UAR: {uar:.4f} - Loss: {loss:.4f}')
        fold_no += 1

    print('____________________ RESULTS ____________________')
    print(f'Average Accuracy: {np.mean(acc_per_fold) * 100:.2f}% (+- {np.std(acc_per_fold) * 100:.2f})')
    print(f'Average UAR: {np.mean(uar_per_fold) * 100:.2f}% (+- {np.std(uar_per_fold) * 100:.2f})')
    print(f'Average Loss: {np.mean(loss_per_fold):.4f}')
def main():
    y = np.load('../files/labels.npy')
    X_speech = np.load('../files/selected_features_data.npy')
    X_text = np.load('../files/padded_docs.npy')
    embedding_matrix = np.load('../files/embedding_matrix.npy')

    embedding_dim = embedding_matrix.shape[1]
    vocab_size = embedding_matrix.shape[0]
    max_length = X_text.shape[1]

    N_SAMPLES = X_speech.shape[0]
    perm = np.random.permutation(N_SAMPLES)
    X_speech = X_speech[perm]
    X_text = X_text[perm]
    y = y[perm]

    fit_model(X_speech, X_text, y, max_length, embedding_dim, embedding_matrix, vocab_size)

if __name__ == '__main__':
    main()
