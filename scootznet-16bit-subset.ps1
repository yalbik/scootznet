$Library = @(
    'lib\Neuron.ps1',
    'lib\Layer.ps1',
    'lib\NeuralNetwork.ps1'
)

# Import the library files
foreach ($lib in $Library) {
    . (Join-Path $PSScriptRoot $lib)
}

# Create neural network with architecture [16, 1] - direct input to output
$network = [NeuralNetwork]::new(@(16, 1), 0.01)

# Generate training data for doubling 16-bit numbers (subset for faster training)
Write-Host "Generating training data for 16-bit numbers (subset for faster training)..."
$trainingInputs = @()
$trainingTargets = @()

# Create training samples for a subset of 16-bit numbers (every 64th number for speed)
# This gives us about 1024 samples instead of 65536
$stepSize = 64
Write-Host "Generating training samples (every ${stepSize}th number from 0-65535)..."
for ($i = 0; $i -le 65535; $i += $stepSize) {
    # Convert number to 16-bit binary array
    $binaryInput = @()
    for ($bit = 15; $bit -ge 0; $bit--) {
        if (($i -band [Math]::Pow(2, $bit)) -ne 0) {
            $binaryInput += 1.0
        } else {
            $binaryInput += 0.0
        }
    }
    
    # Target is the number doubled, normalized to [0,1] range
    $doubled = $i * 2
    $normalizedTarget = $doubled / 131070.0  # 131070 = 65535 * 2 (max possible output)
    
    $trainingInputs += ,$binaryInput
    $trainingTargets += ,@($normalizedTarget)
}

Write-Host "Training data generated: $($trainingInputs.Length) samples"
Write-Host "Each input is 16-bit binary, target is doubled value normalized to [0,1]"

# Train the network
Write-Host "`nStarting training..."
Write-Host "Training on subset for faster development and testing."
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$network.Train($trainingInputs, $trainingTargets, 8000)
$stopwatch.Stop()
Write-Host "Total training time: $([Math]::Round($stopwatch.Elapsed.TotalSeconds, 2)) seconds"

# Test the network with random examples
Write-Host "`nTesting network with 15 random 16-bit inputs:"
$rand = New-Object System.Random
$testNumbers = @()
for ($i = 0; $i -lt 15; $i++) {
    $testNumbers += $rand.Next(0, 65536)
}
$totalAccuracy = 0.0

foreach ($testNum in $testNumbers) {
    # Convert to 16-bit binary
    $binaryTest = @()
    for ($bit = 15; $bit -ge 0; $bit--) {
        if (($testNum -band [Math]::Pow(2, $bit)) -ne 0) {
            $binaryTest += 1.0
        } else {
            $binaryTest += 0.0
        }
    }
    
    # Get prediction
    $prediction = $network.Predict($binaryTest)
    $denormalizedPrediction = $prediction[0] * 131070.0  # Convert back to actual number
    $expected = $testNum * 2
      # Calculate accuracy percentage
    $errorAmount = [Math]::Abs([double]$expected - [double]$denormalizedPrediction)
    $accuracy = if ([double]$expected -eq 0.0) { 
        if ([double]$denormalizedPrediction -eq 0.0) { 100.0 } else { 0.0 }
    } else { 
        [Math]::Max(0.0, 100.0 - ([double]$errorAmount / [double]$expected * 100.0))
    }
    $totalAccuracy += [double]$accuracy
    
    Write-Host "Input: $testNum, Predicted: $([Math]::Round([double]$denormalizedPrediction, 1)), Expected: $expected, Accuracy: $([Math]::Round([double]$accuracy, 1))%"
}

$averageAccuracy = $totalAccuracy / $testNumbers.Length
Write-Host "`nOverall Test Accuracy: $([Math]::Round($averageAccuracy, 1))%"
Write-Host "`nNote: This was trained on a subset of data (every ${stepSize}th number)."
Write-Host "For full training on all 65,536 16-bit numbers, use scootznet.ps1"
