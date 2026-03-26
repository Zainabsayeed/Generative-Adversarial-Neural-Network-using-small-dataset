import tensorflow as tf
from tensorflow.keras import layers

def build_generator(latent_dim, output_dim):
    model = tf.keras.Sequential(name="Generator")

    Input layer: Latent space (noise)
    model.add(layers.Dense(64, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))

    Hidden layer
    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))

    Output layer: Synthetic Feature Vector
    We use 'tanh' if data is scaled [-1, 1] or 'sigmoid' for [0, 1]
    model.add(layers.Dense(output_dim, activation='sigmoid'))

    return model

# Discriminator Model: Classifies Real vs. Synthetic Data
def build_discriminator(input_dim):
    model = tf.keras.Sequential(name="Discriminator")

    model.add(layers.Dense(128, input_dim=input_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(64))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    Output layer: Probability of being "Real"
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

# Hyperparameters and Model Initialization
import numpy as np
import matplotlib.pyplot as plt

Hyperparameters
latent_dim = 10
feature_dim = 20  # e.g., Top 20 TF-IDF features
optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

Initialize Models
generator = build_generator(latent_dim, feature_dim)
discriminator = build_discriminator(feature_dim)
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

Combined GAN Model (To train the Generator)
discriminator.trainable = False
gan_input = layers.Input(shape=(latent_dim,))
fake_data = generator(gan_input)
gan_output = discriminator(fake_data)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=optimizer)

# Training Loop
def plot_gan_losses(g_losses, d_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss', alpha=0.7)
    plt.plot(d_losses, label='Discriminator Loss', alpha=0.7)
    plt.title("GAN Training Loss (Cyberbullying Augmentation)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


# Visualization of Real vs. Synthetic Data Distributions
def plot_distribution(real_data, synthetic_data):
    plt.figure(figsize=(10, 5))
    plt.hist(real_data.flatten(), bins=30, alpha=0.5, label='Real Small Dataset', color='blue')
    plt.hist(synthetic_data.flatten(), bins=30, alpha=0.5, label='GAN Augmented Data', color='red')
    plt.title("Data Distribution: Real vs. Synthetic Features")
    plt.legend()
    plt.show()