# ScootzNet - PowerShell Neural Network

A feedforward neural network implementation in PowerShell that learns to multiply numbers by 2. Demonstrates neural network fundamentals with 8-bit and 16-bit number processing.

## Quick Start
```powershell
cd scootznet
powershell -File scootznet-8bit.ps1        # Recommended for learning
powershell -File scootznet-16bit-subset.ps1 # Advanced training
powershell -File tests/Run-AllTests.ps1     # Run unit tests
```

## Features

- **Pure PowerShell**: No dependencies, Xavier initialization, real-time progress
- **Modular Design**: Neuron, Layer, and NeuralNetwork classes with full test coverage
- **Smart Architecture**: 8→2→1 and 16→8→1 with easy configuration at script top
- **Enhanced UX**: Rounded predictions, color-coded accuracy (🟢100% 🟡>98% 🟠>90% 🔴<90%)

## Configuration

Scripts include easy-to-modify variables at the top:
```powershell
$LAYERS = @(8, 2, 1)          # Architecture: inputs → hidden → output
$LEARNING_RATE = 0.01         # 0.001 (stable) to 0.05 (fast)
$EPOCHS = 2500               # Training iterations
```

## Files

**Core**: `lib/Neuron.ps1`, `lib/Layer.ps1`, `lib/NeuralNetwork.ps1`  
**Training**: `scootznet-8bit.ps1` (256 samples), `scootznet-16bit-subset.ps1` (~1K), `scootznet-16bit-full.ps1` (65K)  
**Tests**: Complete Pester test suite in `tests/` directory

## Sample Output

```
Training completed! Total time: 2.50 seconds

Testing network with 10 random inputs:
Input: 127, Predicted: 254, Expected: 254, Accuracy: 100.0%
Input: 89, Predicted: 178, Expected: 178, Accuracy: 100.0%
Overall Test Accuracy: 100.0%  [Color: Green]
```

## How It Works

**Binary Conversion**: Numbers → binary arrays (e.g., 5 → [0,0,0,0,0,1,0,1])  
**Neural Processing**: Forward/backward propagation with gradient descent  
**Normalization**: Outputs scaled to [0,1] to prevent saturation

## Architecture Options

```powershell
$LAYERS = @(8, 1)             # Direct (fastest)
$LAYERS = @(8, 4, 1)          # Single hidden layer (recommended)
$LAYERS = @(16, 12, 8, 4, 1)  # Multiple layers (complex patterns)
```

## Requirements
- PowerShell 5.1+ or PowerShell Core 6.0+
- No external dependencies

---
**ScootzNet** - Learning to double numbers, one bit at a time! 🧠✖️2️⃣
