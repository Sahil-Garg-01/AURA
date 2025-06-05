import tensorflow as tf
import numpy as np

def build_fusion_model(audio_dim=2048, face_dim=128, text_dim=768, num_classes=7):
    
    audio_input = tf.keras.layers.Input(shape=(audio_dim,), name='audio_input')
    face_input = tf.keras.layers.Input(shape=(face_dim,), name='face_input')
    text_input = tf.keras.layers.Input(shape=(text_dim,), name='text_input')

    concatenated = tf.keras.layers.Concatenate()([audio_input, face_input, text_input])
    x = tf.keras.layers.Dense(128, activation='relu')(concatenated)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    output = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)
    model = tf.keras.Model(inputs=[audio_input, face_input, text_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    model = build_fusion_model()
    model.summary()
    audio_emb = np.random.rand(1, 2048).astype(np.float32)
    face_emb = np.random.rand(1, 128).astype(np.float32)
    text_emb = np.random.rand(1, 768).astype(np.float32)
    prediction = model.predict([audio_emb, face_emb, text_emb])
    print("Prediction shape:", prediction.shape)