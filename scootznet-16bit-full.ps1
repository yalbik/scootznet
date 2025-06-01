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

# Generate training data for doubling 16-bit numbers (FULL DATASET)
Write-Host "Generating training data for 16-bit numbers (FULL DATASET - this may take a while)..."
$trainingInputs = @()
$trainingTargets = @()

# Create training samples for ALL 16-bit numbers (0-65535)
# This will be 65,536 samples - comprehensive but slow
Write-Host "Generating all 65,536 training samples for complete 16-bit coverage..."
$startTime = Get-Date
for ($i = 0; $i -le 65535; $i++) {
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
    
    # Progress indicator every 5,000 samples
    if (($i % 5000) -eq 0 -and $i -gt 0) {
        $elapsed = (Get-Date) - $startTime
        $remaining = ($elapsed.TotalSeconds / $i) * (65535 - $i)
        Write-Host "Generated $i samples... (Est. $([Math]::Round($remaining, 0)) seconds remaining)"
    }
}

$dataGenTime = (Get-Date) - $startTime
Write-Host "Training data generated: $($trainingInputs.Length) samples"
Write-Host "Data generation time: $([Math]::Round($dataGenTime.TotalSeconds, 2)) seconds"
Write-Host "Each input is 16-bit binary, target is doubled value normalized to [0,1]"

# Train the network
Write-Host "`nStarting training on FULL 16-bit dataset..."
Write-Host "WARNING: This will train on 65,536 samples for 2,500 epochs - expect significant training time!"
Write-Host "Consider using scootznet-16bit-subset.ps1 for faster development/testing."
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$network.Train($trainingInputs, $trainingTargets, 2500)
$stopwatch.Stop()
Write-Host "Total training time: $([Math]::Round($stopwatch.Elapsed.TotalSeconds, 2)) seconds"
Write-Host "Total time (data + training): $([Math]::Round(($dataGenTime.TotalSeconds + $stopwatch.Elapsed.TotalSeconds), 2)) seconds"

# Test the network with random examples
Write-Host "`nTesting network with 20 random 16-bit inputs:"
$rand = New-Object System.Random
$testNumbers = @()
for ($i = 0; $i -lt 20; $i++) {
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
Write-Host "`nFULL 16-bit training completed!"
Write-Host "This network was trained on ALL 65,536 possible 16-bit inputs."
Write-Host "For faster development iterations, consider using:"
Write-Host "  - scootznet-8bit.ps1 (8-bit, 256 samples, ~1-2 seconds)"
Write-Host "  - scootznet-16bit-subset.ps1 (16-bit subset, ~1,024 samples, ~10-15 seconds)"
