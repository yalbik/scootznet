$Library = @(
    'lib\Neuron.ps1',
    'lib\Layer.ps1',
    'lib\NeuralNetwork.ps1'
)

# Import the library files
foreach ($lib in $Library) {
    . (Join-Path $PSScriptRoot $lib)
}

# Create neural network with architecture [8, 1] - direct input to output
$network = [NeuralNetwork]::new(@(8, 1), 0.01)

# Generate training data for doubling 8-bit numbers
Write-Host "Generating training data..."
$trainingInputs = @()
$trainingTargets = @()

# Create training samples for all 8-bit numbers (0-255)
for ($i = 0; $i -le 255; $i++) {
    # Convert number to 8-bit binary array
    $binaryInput = @()
    for ($bit = 7; $bit -ge 0; $bit--) {
        if (($i -band [Math]::Pow(2, $bit)) -ne 0) {
            $binaryInput += 1.0
        } else {
            $binaryInput += 0.0
        }
    }
    
    # Target is the number doubled, normalized to [0,1] range
    $doubled = $i * 2
    $normalizedTarget = $doubled / 510.0  # 510 = 255 * 2 (max possible output)
    
    $trainingInputs += ,$binaryInput
    $trainingTargets += ,@($normalizedTarget)
}

Write-Host "Training data generated: $($trainingInputs.Length) samples"
Write-Host "Each input is 8-bit binary, target is doubled value normalized to [0,1]"

# Train the network
Write-Host "`nStarting training..."
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$network.Train($trainingInputs, $trainingTargets, 10000)
$stopwatch.Stop()
Write-Host "Total training time: $([Math]::Round($stopwatch.Elapsed.TotalSeconds, 2)) seconds"

# Test the network with random examples
Write-Host "`nTesting network with 10 random inputs:"
$rand = New-Object System.Random
$testNumbers = @()
for ($i = 0; $i -lt 10; $i++) {
    $testNumbers += $rand.Next(0, 256)
}
$totalAccuracy = 0.0

foreach ($testNum in $testNumbers) {
    # Convert to binary
    $binaryTest = @()
    for ($bit = 7; $bit -ge 0; $bit--) {
        if (($testNum -band [Math]::Pow(2, $bit)) -ne 0) {
            $binaryTest += 1.0
        } else {
            $binaryTest += 0.0
        }
    }
    
    # Get prediction
    $prediction = $network.Predict($binaryTest)
    $denormalizedPrediction = $prediction[0] * 510.0  # Convert back to actual number
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
