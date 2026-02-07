import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 25
NUM_CLASSES = 34  # 0-6 for the 7 gestures

class GestureModelTrainer:
    def __init__(self, data_dir='data2', model_name='gesture_model'):
        self.data_dir = data_dir
        self.model_name = model_name
        self.train_dir = os.path.join(data_dir, 'train')
        self.test_dir = os.path.join(data_dir, 'test')
        self.model = None
        self.history = None
        
    def build_model_v1(self):
        """Original model architecture (improved)"""
        model = Sequential([
            # First conv block
            Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            
            # Second conv block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            
            # Third conv block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            
            # Flatten and dense layers
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.4),
            Dense(NUM_CLASSES, activation='softmax')
        ])
        
        return model
    
    def build_model_v2(self):
        """Improved model with more capacity"""
        model = Sequential([
            # Block 1
            Conv2D(32, (3, 3), padding='same', activation='relu', 
                   input_shape=(IMG_SIZE, IMG_SIZE, 1)),
            BatchNormalization(),
            Conv2D(32, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Block 2
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Block 3
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Dense layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(NUM_CLASSES, activation='softmax')
        ])
        
        return model
    
    def prepare_data_generators(self):
        """Create data generators with augmentation"""
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation data (no augmentation)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            color_mode='grayscale',
            class_mode='categorical',
            shuffle=True
        )
        
        test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            color_mode='grayscale',
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, test_generator
    
    def get_callbacks(self):
        """Create training callbacks"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            # Save best model
            ModelCheckpoint(
                f'{self.model_name}_best.keras',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard
            TensorBoard(
                log_dir=f'logs/{self.model_name}_{timestamp}',
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def train(self, model_version='v2'):
        """Train the model"""
        print("="*60)
        print("TRAINING GESTURE RECOGNITION MODEL")
        print("="*60)
        
        # Build model
        print(f"\nBuilding model (version {model_version})...")
        if model_version == 'v1':
            self.model = self.build_model_v1()
        else:
            self.model = self.build_model_v2()
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        self.model.summary()
        
        # Prepare data
        print("\nPreparing data generators...")
        train_gen, test_gen = self.prepare_data_generators()
        
        print(f"\nTraining samples: {train_gen.samples}")
        print(f"Validation samples: {test_gen.samples}")
        print(f"Class labels: {train_gen.class_indices}")
        
        # Calculate steps
        steps_per_epoch = train_gen.samples // BATCH_SIZE
        validation_steps = test_gen.samples // BATCH_SIZE
        
        print(f"\nSteps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        
        # Train model
        print("\nStarting training...")
        print("="*60)
        
        self.history = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=test_gen,
            validation_steps=validation_steps,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        print("\nTraining complete!")
        
        # Evaluate final model
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        train_loss, train_acc = self.model.evaluate(train_gen, verbose=0)
        test_loss, test_acc = self.model.evaluate(test_gen, verbose=0)
        
        print(f"Training Accuracy: {train_acc*100:.2f}%")
        print(f"Validation Accuracy: {test_acc*100:.2f}%")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {test_loss:.4f}")
        
        return self.history
    
    def save_model(self):
        """Save model in multiple formats"""
        print("\nSaving model...")
        
        # Save in Keras format (recommended)
        self.model.save(f'{self.model_name}.keras')
        print(f"✓ Saved: {self.model_name}.keras")
        
        # Save model architecture as JSON
        model_json = self.model.to_json()
        with open(f'{self.model_name}.json', 'w') as json_file:
            json_file.write(model_json)
        print(f"✓ Saved: {self.model_name}.json")
        
        # Save weights
        self.model.save_weights(f'{self.model_name}.weights.h5')
        print(f"✓ Saved: {self.model_name}.weights.h5")
        
        print("\nModel saved successfully!")
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Train Loss')
        ax2.plot(self.history.history['val_loss'], label='Val Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.model_name}_training_history.png', dpi=300)
        print(f"\n✓ Training history plot saved: {self.model_name}_training_history.png")
        plt.show()


def main():
    # ========================================
    # CONFIGURATION - Change these if needed
    # ========================================
    data_dir = 'data2'           # Directory with preprocessed data
    model_name = 'gesture_model' # Name for saved model
    model_version = 'v2'         # 'v1' (simpler) or 'v2' (better)
    epochs = 25                  # Number of training epochs
    batch_size = 32              # Batch size
    # ========================================
    
    # Update global config
    global EPOCHS, BATCH_SIZE
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    
    # Create trainer
    print(f"\nConfiguration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Model version: {model_version}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}\n")
    
    trainer = GestureModelTrainer(data_dir, model_name)
    
    # Train model
    trainer.train(model_version=model_version)
    
    # Save model
    trainer.save_model()
    
    # Plot history
    trainer.plot_training_history()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
