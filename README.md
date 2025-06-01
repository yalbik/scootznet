# ScootzNet - PowerShell Neural Network

A simple feedforward neural network implementation in PowerShell that learns to multiply numbers by 2. This project demonstrates fundamental neural network concepts including forward propagation, backpropagation, and gradient descent training.

## Overview

ScootzNet is designed to learn the doubling function for binary numbers. It takes binary representations of numbers as input and outputs the doubled value. The network supports both 8-bit and 16-bit number processing.

## Features

- **Pure PowerShell Implementation**: No external dependencies required
- **Modular Architecture**: Clean separation of Neuron, Layer, and NeuralNetwork classes
- **Flexible Input Size**: Supports both 8-bit (0-255) and 16-bit (0-65535) numbers
- **Xavier Weight Initialization**: Proper weight initialization for stable training
- **Comprehensive Training**: Options for full dataset or subset training
- **Real-time Progress Tracking**: Training progress and timing information
- **Accuracy Metrics**: Detailed accuracy reporting for test predictions

## Architecture

### Network Structure
- **Input Layer**: 8 or 16 neurons (for 8-bit or 16-bit binary input)
- **Output Layer**: 1 neuron (for the doubled result)
- **No Hidden Layers**: Direct mapping from input to output

### Classes

#### Neuron (`lib/Neuron.ps1`)
- Individual neuron with weights, bias, and activation functions
- Supports both sigmoid and linear activation
- Xavier weight initialization for stable training
- Overflow protection in sigmoid function

#### Layer (`lib/Layer.ps1`)
- Collection of neurons forming a network layer
- Forward pass functionality
- Construction reporting for debugging

#### NeuralNetwork (`lib/NeuralNetwork.ps1`)
- Complete neural network with multiple layers
- Forward and backward propagation
- Training with configurable epochs and learning rate
- Prediction functionality

## Files

### Core Library
- `lib/Neuron.ps1` - Individual neuron implementation
- `lib/Layer.ps1` - Neural network layer implementation
- `lib/NeuralNetwork.ps1` - Complete neural network implementation

### Training Scripts
- `scootznet.ps1` - Full training on all possible inputs
- `scootznet-16bit-subset.ps1` - Faster training on subset of data

## Usage

### 8-bit Numbers (0-255)
```powershell
# Run the original 8-bit training
cd c:\codez\scootznet
powershell -File scootznet.ps1
```

### 16-bit Numbers (0-65535)
```powershell
# Full training (65,536 samples - slow but comprehensive)
cd c:\codez\scootznet
powershell -File scootznet.ps1

# Subset training (1,024 samples - faster for testing)
cd c:\codez\scootznet
powershell -File scootznet-16bit-subset.ps1
```

## Training Process

1. **Data Generation**: Creates binary representations of all numbers in range
2. **Normalization**: Targets are normalized to [0,1] range
3. **Training**: Uses gradient descent with backpropagation
4. **Testing**: Evaluates performance on random test numbers
5. **Accuracy Reporting**: Provides individual and overall accuracy metrics

## Configuration

### Default Parameters
- **Learning Rate**: 0.01
- **Training Epochs**: 5,000-10,000 (varies by script)
- **Architecture**: [Input_Size, 1] (direct mapping)

### Customization
You can modify these parameters in the training scripts:
```powershell
# Change network architecture
$network = [NeuralNetwork]::new(@(16, 1), 0.01)

# Adjust training epochs
$network.Train($trainingInputs, $trainingTargets, 8000)
```

## Sample Output

```
Generating training data for 16-bit numbers (subset for faster training)...
Generating training samples (every 64th number from 0-65535)...
Training data generated: 1024 samples
Each input is 16-bit binary, target is doubled value normalized to [0,1]

Starting training...
Training on subset for faster development and testing.
Epoch 100/8000, Average Loss: 0.2145
Epoch 200/8000, Average Loss: 0.1987
...
Total training time: 15.23 seconds

Testing network with 15 random 16-bit inputs:
Input: 12847, Predicted: 25694.2, Expected: 25694, Accuracy: 99.9%
Input: 3921, Predicted: 7842.1, Expected: 7842, Accuracy: 99.9%
...
Overall Test Accuracy: 98.7%
```

## Technical Details

### Binary Conversion
Numbers are converted to binary arrays where each bit becomes a neuron input:
```powershell
# Example: 5 (decimal) → [0,0,0,0,0,1,0,1] (8-bit binary)
for ($bit = 7; $bit -ge 0; $bit--) {
    if (($number -band [Math]::Pow(2, $bit)) -ne 0) {
        $binaryInput += 1.0
    } else {
        $binaryInput += 0.0
    }
}
```

### Normalization
Output values are normalized to prevent saturation:
```powershell
# 8-bit: divide by 510 (255 * 2)
# 16-bit: divide by 131070 (65535 * 2)
$normalizedTarget = $doubled / $maxPossibleOutput
```

### Weight Initialization
Xavier initialization ensures stable training:
```powershell
$range = [Math]::Sqrt(6.0 / $numInputs)
$weight = ($rand.NextDouble() - 0.5) * 2 * $range
```

## Performance Considerations

### Training Time
- **8-bit full**: ~256 samples, very fast
- **16-bit full**: ~65,536 samples, slow but comprehensive
- **16-bit subset**: ~1,024 samples, balanced speed/coverage

### Memory Usage
- Minimal memory footprint
- All data structures use PowerShell arrays
- Suitable for educational and demonstration purposes

## Limitations

1. **Single Task**: Only learns to multiply by 2
2. **Binary Input Only**: Requires binary representation of numbers
3. **PowerShell Performance**: Not optimized for large-scale training
4. **Simple Architecture**: No hidden layers, limited complexity

## Educational Value

This project demonstrates:
- **Neural Network Fundamentals**: Forward/backward propagation
- **Gradient Descent**: Weight and bias updates
- **Data Preprocessing**: Binary conversion and normalization
- **Training Loops**: Epoch-based learning with progress tracking
- **Model Evaluation**: Accuracy metrics and testing procedures

## Requirements

- Windows PowerShell 5.1 or later
- No external dependencies
- Approximately 50MB RAM for 16-bit training

## Contributing

This is an educational project. Feel free to:
- Add new activation functions
- Implement different architectures
- Add support for other mathematical operations
- Optimize for performance

## License

This project is for educational purposes. Use and modify freely.

---

**ScootzNet** - Learning to double numbers, one bit at a time! 🧠✖️2️⃣
