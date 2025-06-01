# ScootzNet - PowerShell Neural Network

A feedforward neural network implementation in PowerShell that learns to multiply numbers by 2. Demonstrates neural network fundamentals including forward/backward propagation and gradient descent.

## Overview

ScootzNet learns the doubling function for binary numbers. Supports both 8-bit (0-255) and 16-bit (0-65535) number processing with flexible architecture allowing unlimited hidden layers.

## Features

- **Pure PowerShell**: No external dependencies
- **Modular Design**: Neuron, Layer, and NeuralNetwork classes
- **Flexible Architecture**: Supports any number of hidden layers
- **Xavier Initialization**: Stable weight initialization
- **Real-time Progress**: Training progress and timing
- **Accuracy Metrics**: Detailed performance reporting

## Files

- `lib/Neuron.ps1` - Individual neuron with Xavier initialization
- `lib/Layer.ps1` - Neural network layer implementation
- `lib/NeuralNetwork.ps1` - Complete network with training/prediction
- `scootznet-8bit.ps1` - 8-bit training (256 samples, 2,500 epochs)
- `scootznet-16bit-subset.ps1` - 16-bit subset (~1,024 samples, 2,500 epochs)
- `scootznet-16bit-full.ps1` - 16-bit full (65,536 samples, 2,500 epochs)

## Usage

### 8-bit Numbers (0-255) - Recommended for Learning
```powershell
cd scootznet
powershell -File scootznet-8bit.ps1
```

### 16-bit Numbers (0-65535) - Advanced Training
```powershell
# Subset training (development)
powershell -File scootznet-16bit-subset.ps1

# Full training (research/production)
powershell -File scootznet-16bit-full.ps1
```

## Configuration

### Default Parameters
- **Learning Rate**: 0.01 (adjustable)
- **Training Epochs**: 2,500 epochs (all versions)
- **Architecture**: Flexible - supports unlimited hidden layers

### Customization
```powershell
# Current architecture (no hidden layers)
$network = [NeuralNetwork]::new(@(8, 1), 0.01)     # 8-bit: [8 inputs → 1 output]
$network = [NeuralNetwork]::new(@(16, 1), 0.01)    # 16-bit: [16 inputs → 1 output]

# Hidden layer examples
$network = [NeuralNetwork]::new(@(8, 4, 1), 0.01)      # Single hidden layer
$network = [NeuralNetwork]::new(@(16, 12, 8, 4, 1), 0.01)  # Multiple hidden layers

# Learning rate customization
$network = [NeuralNetwork]::new(@(8, 1), 0.001)   # Conservative (0.001-0.005)
$network = [NeuralNetwork]::new(@(8, 1), 0.05)    # Aggressive (0.05-0.1)

# Training epochs
$network.Train($trainingInputs, $trainingTargets, 2500)
```

#### Guidelines
- **Architecture**: First number = input size, last = 1 (output), middle = hidden layers
- **Learning Rate**: Default 0.01, lower for stability, higher for speed
- **Hidden Layers**: More layers = slower training, potentially better learning

## Sample Output

### 8-bit Training Example
```
Generating training data...
Training data generated: 256 samples
Each input is 8-bit binary, target is doubled value normalized to [0,1]

Starting training for 2500 epochs...
Epoch: 0 / 2500 - Starting training...
Epoch: 100 / 2500 - Elapsed: 0.1s, Estimated remaining: 2.4s
Epoch: 1000 / 2500 - Elapsed: 1.0s, Estimated remaining: 1.5s
Epoch: 2000 / 2500 - Elapsed: 2.0s, Estimated remaining: 0.5s
Training completed! Total time: 2.50 seconds

Testing network with 10 random inputs:
Input: 127, Predicted: 254.1, Expected: 254, Accuracy: 99.9%
Input: 89, Predicted: 178.0, Expected: 178, Accuracy: 100.0%
...
Overall Test Accuracy: 99.2%
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

## Architecture Flexibility

### Hidden Layer Support
The neural network supports **unlimited hidden layers** with **any number of neurons per layer**. Current scripts use direct mapping (no hidden layers) for simplicity, but you can easily add complex architectures.

### Architecture Examples
```powershell
# Direct mapping (current scripts)
$network = [NeuralNetwork]::new(@(8, 1), 0.01)    # 8-bit: Input → Output
$network = [NeuralNetwork]::new(@(16, 1), 0.01)   # 16-bit: Input → Output

# Single hidden layer
$network = [NeuralNetwork]::new(@(8, 4, 1), 0.01)     # 8-bit: 8 → 4 → 1
$network = [NeuralNetwork]::new(@(16, 32, 1), 0.01)   # 16-bit: 16 → 32 → 1

# Multiple hidden layers
$network = [NeuralNetwork]::new(@(16, 12, 8, 4, 1), 0.01)  # Progressive reduction
$network = [NeuralNetwork]::new(@(8, 16, 8, 1), 0.01)      # Encoder-decoder style
```

### When to Use Hidden Layers
- **No Hidden Layers**: Fastest training, good for simple patterns like doubling
- **Single Hidden Layer**: Non-linear patterns, moderate training time
- **Multiple Hidden Layers**: Complex patterns, significantly longer training

## Performance
- **8-bit**: 256 samples, fast training (2,500 epochs)
- **16-bit subset**: ~1,024 samples, moderate training
- **16-bit full**: 65,536 samples, slow but comprehensive training

## Requirements
- Windows PowerShell 5.1+ or PowerShell Core 6.0+
- No external dependencies
- 10-200MB RAM (depending on version)

## Quick Start
1. Navigate to project directory: `cd scootznet`
2. Start with 8-bit: `powershell -File scootznet-8bit.ps1`
3. Try 16-bit subset: `powershell -File scootznet-16bit-subset.ps1`
4. Advanced: `powershell -File scootznet-16bit-full.ps1`

## Educational Value
This project demonstrates:
- Neural network fundamentals (forward/backward propagation)
- Architecture flexibility (unlimited hidden layers)
- Gradient descent and weight updates
- Data preprocessing (binary conversion, normalization)
- Model evaluation and accuracy metrics

---
**ScootzNet** - Learning to double numbers, one bit at a time! 🧠✖️2️⃣
