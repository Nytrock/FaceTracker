from training.model_creator import create_model
from tensorflow import keras

img_size = 224
seed = 123
batch_size = 20
epochs = 40

classes_orig = ['BACKGROUND', 'PERSON', 'SKIN', 'LEFT BROW', 'RIGHT_BROW', 'LEFT_EYE', 'RIGHT_EYE', 'LIPS', 'TEETH']
classes_mine = ['BACKGROUND', 'PERSON', 'SKIN', 'BROW', 'EYE', 'LIPS', 'TEETH']

if __name__ == '__main__':
    from dataset_creator import get_datasets

    train_ds, val_ds = get_datasets(img_size, batch_size)
    model = create_model(len(classes_orig), img_size)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    model.save('model.keras')
