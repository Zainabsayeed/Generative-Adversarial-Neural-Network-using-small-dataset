# Exploring GAN Using Small Dataset

## Project Overview

This project implements a **Generative Adversarial Network (GAN)** to augment small datasets through synthetic data generation. The GAN architecture consists of a Generator and Discriminator network that work together to learn data distributions and create realistic synthetic samples.

### Use Case
The implementation is specifically designed for cyberbullying text classification, where it augments limited training data by generating synthetic feature vectors that mimic the distribution of real data.

## Architecture

### Generator Model
- **Input**: Random noise from latent space (dimension: 10)
- **Hidden Layers**: 
  - Dense(64) → LeakyReLU(0.2) → BatchNormalization
  - Dense(128) → LeakyReLU(0.2) → BatchNormalization
- **Output**: Synthetic feature vector (20 features) with sigmoid activation

### Discriminator Model
- **Input**: Feature vectors (20 dimensions)
- **Hidden Layers**:
  - Dense(128) → LeakyReLU(0.2) → Dropout(0.3)
  - Dense(64) → LeakyReLU(0.2) → Dropout(0.3)
- **Output**: Binary classification (Real vs. Synthetic) with sigmoid activation

## Key Features

- **Latent Dimension**: 10 (noise input dimension)
- **Feature Dimension**: 20 (e.g., TF-IDF features)
- **Optimizer**: Adam with learning rate 0.0002, beta_1 = 0.5
- **Loss Function**: Binary Crossentropy

## Model Components

1. **Generator (`build_generator`)**: Creates synthetic data from random noise
2. **Discriminator (`build_discriminator`)**: Classifies real vs. synthetic data
3. **Combined GAN Model**: Trains the generator to fool the discriminator
4. **Visualization Tools**:
   - `plot_gan_losses()`: Tracks generator and discriminator loss during training
   - `plot_distribution()`: Compares real vs. synthetic data distributions

## Dependencies

- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib

## How to Use

### 1. Train the GAN
```python
# The model is initialized and compiled
# Add your training loop to optimize both networks
```

### 2. Generate Synthetic Data
```python
# Generate synthetic samples using the trained generator
noise = np.random.normal(0, 1, (num_samples, latent_dim))
synthetic_data = generator.predict(noise)
```

### 3. Visualize Results
```python
# Plot training losses
plot_gan_losses(g_losses, d_losses)

# Compare data distributions
plot_distribution(real_data, synthetic_data)
```

## Installation

```bash
pip install tensorflow numpy matplotlib
```

## Workflow

1. **Generator Training**: Learn to generate realistic features
2. **Discriminator Training**: Learn to distinguish real from synthetic data
3. **Adversarial Process**: Both networks improve iteratively
4. **Data Augmentation**: Use trained generator to create synthetic samples for model training

## Notes

- Data scaling should match the activation function (sigmoid for [0,1] or tanh for [-1,1])
- Batch normalization helps stabilize training
- Dropout in discriminator prevents overfitting
- LeakyReLU activation prevents dead neurons in deeper layers

## Author
Created for exploring GAN-based data augmentation techniques on small datasets.

## License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
