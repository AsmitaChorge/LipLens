import os 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten

def load_model() -> Sequential: 
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

    # weight_paths = [
    #     # os.path.join('models','lipnet.h5'),
    #     # os.path.join('models','checkpoint.weights.h5'), 
    #     os.path.join('models','checkpoint'),
    #     # os.path.join('models-checkpoint-96','checkpoint')
    # ]
    # print("Searching for weights in:", weight_paths)
    
    # for path in weight_paths:
    #     try:
    #         if os.path.exists(path):
    #             if path.endswith('.h5') or path.endswith('.weights.h5'):
    #                 model.load_weights(path)
    #             else:
    #                 # Convert legacy checkpoint to .h5 format
    #                 temp_path = path + '.weights.h5'
    #                 model.save_weights(temp_path)
    #                 model.load_weights(temp_path)
    #             print(f"Successfully loaded weights from {path}")
    #             return model
    #     except Exception as e:
    #         print(f"Error loading weights from {path}: {str(e)}")
    #         continue
            
    # print("""
    # ERROR: No valid model weights found. Please:
    # 1. Download pretrained lipnet.h5 from:
    #    https://github.com/rizkiarm/LipNet#pretrained-model
    # 2. Place in models/ directory
    # OR
    # 3. Train model from scratch using LipNet.ipynb
    # """)
    # return None
    model.load_weights(os.path.join('models','checkpoint'))

    return model

