import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Embedding, SpatialDropout1D, Reshape, Conv2D, MaxPool2D,
                                     Concatenate, Flatten, Dropout, Conv1D, MaxPooling1D,
                                     BatchNormalization, Dense)

def emotion_aware_loss(similarity_matrix):
    similarity_matrix = tf.constant(similarity_matrix, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.squeeze(y_true, axis=-1)   # 👈 این خط اضافه بشه
        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        pred_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

        similarities = tf.gather_nd(similarity_matrix, tf.stack([y_true, pred_labels], axis=1))
        weighted_loss = ce_loss * similarities
        return tf.reduce_mean(weighted_loss)

    return loss_fn


def build_model(max_length, embedding_dim, embedding_matrix, n_features, vocab_size):
    # Input for Speech
    input_speech = Input((1, n_features))
    speech = Conv1D(128, kernel_size=5, strides=2, padding='same', activation='relu')(input_speech)
    speech = MaxPooling1D(padding='same')(speech)
    speech = BatchNormalization(axis=-1)(speech)
    speech = Dropout(0.5)(speech)
    speech = Conv1D(128, kernel_size=5, strides=2, padding='same', activation='relu')(speech)
    speech = MaxPooling1D(padding='same')(speech)
    speech = BatchNormalization(axis=-1)(speech)
    speech = Dropout(0.5)(speech)
    speech = Flatten()(speech)

    # Input for Text
    input_text = Input(shape=(max_length,), dtype='int32')
    embedding = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], mask_zero=True, trainable=True)(input_text)
    embedding = SpatialDropout1D(0.5)(embedding)
    reshape = Reshape((max_length, embedding_dim, 1))(embedding)

    num_filters = 512
    filter_sizes = [3, 4, 5]

    conv_blocks = []
    for size in filter_sizes:
        conv = Conv2D(num_filters, (size, embedding_dim), activation='relu')(reshape)
        pool = MaxPool2D(pool_size=(max_length - size + 1, 1))(conv)
        conv_blocks.append(pool)

    text = Concatenate(axis=1)(conv_blocks)
    text = Flatten()(text)
    text = Dropout(0.5)(text)

    # Fusion
    fusion = Concatenate()([speech, text])
    fusion = Dense(256, activation='relu')(fusion)
    fusion = Dense(128, activation='relu')(fusion)
    fusion = Dropout(0.5)(fusion)
    output = Dense(5, activation='softmax')(fusion)

    model = Model([input_speech, input_text], output)

    # Similarity Matrix
    similarity_matrix = [
        [1.0, 0.5, 0.2, 0.6, 0.3],
        [0.5, 1.0, 0.6, 0.4, 0.5],
        [0.2, 0.6, 1.0, 0.2, 0.5],
        [0.6, 0.4, 0.2, 1.0, 0.5],
        [0.3, 0.5, 0.5, 0.5, 1.0],
    ]
    loss_fn = emotion_aware_loss(similarity_matrix)

    model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])
    return model
