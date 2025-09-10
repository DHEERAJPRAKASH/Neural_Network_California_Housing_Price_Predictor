# California Housing Price Predictor

A neural network project that predicts median house values in California districts using machine learning techniques.

## Project Overview

This project implements a deep learning solution to predict housing prices in California using the California Housing dataset from scikit-learn. The model uses a multi-layer perceptron (MLP) neural network to learn patterns from various housing features and predict median house values.

## Dataset Information

### California Housing Dataset
- **Source**: StatLib repository (1990 U.S. census data)
- **Total Instances**: 20,640
- **Features**: 8 numeric attributes
- **Target**: Median house value (in hundreds of thousands of dollars)

### Input Features

| Feature | Description |
|---------|-------------|
| **MedInc** | Median income in block group |
| **HouseAge** | Median house age in block group |
| **AveRooms** | Average number of rooms per household |
| **AveBedrms** | Average number of bedrooms per household |
| **Population** | Block group population |
| **AveOccup** | Average number of household members |
| **Latitude** | Block group latitude |
| **Longitude** | Block group longitude |

### Target Variable
- **Median House Value**: Expressed in hundreds of thousands of dollars ($100,000)

## Technical Implementation

### Libraries Used
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization
- **Pandas**: Data manipulation
- **Scikit-learn**: Data preprocessing and splitting

### Data Preprocessing

1. **Data Splitting**:
   - Training set: 11,610 samples (75% of 15,480)
   - Validation set: 3,870 samples (25% of 15,480)
   - Test set: 5,160 samples (25% of total)

2. **Feature Standardization**:
   - Applied StandardScaler to normalize features
   - Ensures all features have similar scales for better model performance

### Neural Network Architecture

The model uses a sequential architecture with the following layers:

```
Layer 1: Dense(30, activation='relu', input_shape=(8,))
Layer 2: Dense(10, activation='relu')
Layer 3: Dense(1)  # Output layer for regression
```

**Model Parameters**:
- **Total Parameters**: 593
- **Trainable Parameters**: 591
- **Non-trainable Parameters**: 0

### Training Configuration

- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Learning Rate**: 0.001
- **Epochs**: 60
- **Batch Size**: Default (32)

## Results

### Training Performance

The model shows good convergence during training:

- **Initial Loss**: 2.2113
- **Final Training Loss**: 0.3589
- **Final Validation Loss**: 0.4007

### Test Performance

- **Test MSE**: 0.3558

### Sample Predictions

| Sample | Predicted Value | Actual Value |
|--------|----------------|--------------|
| 1 | 0.87 | 0 |
| 2 | 1.90 | 0 |
| 3 | 4.08 | 5 |

## Key Features

### 1. Data Exploration
- Comprehensive dataset description and feature analysis
- Data visualization and statistical summaries

### 2. Preprocessing Pipeline
- Proper train/validation/test split
- Feature standardization for optimal performance
- Data integrity checks

### 3. Model Architecture
- Multi-layer perceptron design
- Appropriate activation functions (ReLU)
- Single output neuron for regression

### 4. Training Process
- Reproducible results with fixed random seeds
- Validation monitoring during training
- Loss visualization

### 5. Evaluation
- Comprehensive model evaluation on test data
- Sample predictions for verification
- Performance metrics analysis

## Usage

### Prerequisites
```bash
pip install tensorflow numpy matplotlib pandas scikit-learn
```

### Running the Project
1. Open the Jupyter notebook: `Neural_Network_California_Housing_Price_Predictor.ipynb`
2. Execute all cells in sequence
3. Observe the training progress and results

### Key Code Snippets

**Data Loading and Preprocessing**:
```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42
)
```

**Model Definition**:
```python
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(1)
])
```

**Training**:
```python
model.compile(
    loss="mean_squared_error",
    optimizer=keras.optimizers.SGD(learning_rate=1e-3)
)
history = model.fit(X_train, y_train, epochs=60, validation_data=(X_valid, y_valid))
```

## Model Performance Analysis

### Strengths
- **Good Convergence**: Model shows steady improvement over 60 epochs
- **Reasonable Accuracy**: Test MSE of 0.3558 indicates decent performance
- **Proper Validation**: Uses validation set to monitor overfitting

### Areas for Improvement
- **Feature Engineering**: Could explore feature interactions
- **Architecture Tuning**: Experiment with different layer sizes and activations
- **Regularization**: Add dropout or L2 regularization to prevent overfitting
- **Hyperparameter Optimization**: Grid search for optimal learning rate and architecture

## Future Enhancements

1. **Advanced Architectures**: Experiment with deeper networks or different architectures
2. **Feature Engineering**: Create new features from existing ones
3. **Cross-Validation**: Implement k-fold cross-validation for robust evaluation
4. **Hyperparameter Tuning**: Use automated hyperparameter optimization
5. **Ensemble Methods**: Combine multiple models for better predictions

## Conclusion

This project successfully demonstrates the application of neural networks for housing price prediction. The model achieves reasonable performance on the California Housing dataset, providing a solid foundation for further improvements and real-world applications in real estate valuation.

The implementation follows machine learning best practices including proper data splitting, feature standardization, and comprehensive evaluation, making it a valuable learning resource for regression problems in deep learning.
