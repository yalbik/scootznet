$Library = @(
    'lib\Neuron.ps1',
    'lib\Layer.ps1',
    'lib\NeuralNetwork.ps1'
)

foreach ($File in $Library)
{
    try
    {
        $SourceFile = "$PSScriptRoot\$File"
        Write-Host "Sourcing $($SourceFile)..."
        . "$SourceFile"
    }
    catch
    {
        throw "Failed to load $($File):`n$_"
    }
}

$NETWORK_LAYERS = @(8, 16, 8)
$RANDOM_DATA_SIZE = 16
$TRAINING_PASS_EPOCHS = 1000
$TRAINING_PASSES = 10

Function To-Binary ($Number)
{
    $binary = [Convert]::ToString($Number, 2).PadLeft(8, '0').ToCharArray() | ForEach-Object { [int]::Parse($_) }
    return $binary
}

Function From-Binary ($Binary)
{
    return [Convert]::ToInt32($Binary -join '', 2)
}

Function Random-SevenBitNumber
{
    return Get-Random -Minimum 0 -Maximum 127
}

Write-Host "Creating a neural network with $($NETWORK_LAYERS.Length) layers: $($NETWORK_LAYERS -join '-')..."
$MyNetwork = [NeuralNetwork]::new($NETWORK_LAYERS)

Write-Host "Training the network with $($TRAINING_PASSES) passes of $($TRAINING_PASS_EPOCHS) epochs each, $($RANDOM_DATA_SIZE) inputs per pass..."

$Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
for ($Pass = 0; $Pass -lt $TRAINING_PASSES; $Pass++)
{
    Write-Host "Training pass $($Pass + 1) of $($TRAINING_PASSES) ($($Stopwatch.Elapsed.TotalSeconds)s elapsed)..."
    $inputsBase = @()
    while (([array]$inputsBase).Length -lt $RANDOM_DATA_SIZE)
    { 
        $inputsBase += Random-SevenBitNumber 
    }

    $inputs = @()
    $outputs = @()
    $inputsBase | ForEach-Object { 
        $inputRow = To-Binary $_
        $outputRow = To-Binary ($_ * 2)
        $inputs += ,$inputRow
        $outputs += ,$outputRow
    }

    $MyNetwork.Train($inputs, $outputs, $TRAINING_PASS_EPOCHS)
}
$Stopwatch.Stop()
Write-Host "Training completed in $($Stopwatch.Elapsed.TotalSeconds) seconds"

$testInputsInt = @(
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64
)

$expectedOutputsInt = $testInputsInt | ForEach-Object { [int]($_ * 2) }

$testInputs = $testInputsInt | ForEach-Object { ,(To-Binary $_) }
$expectedOutputs = $expectedOutputsInt | ForEach-Object { ,(To-Binary $_) }

for ($i = 0; $i -lt $testInputs.Length; $i++)
{
    $inputInt = $testInputsInt[$i]
    $input = $testInputs[$i]
    $expectedInt = $expectedOutputsInt[$i]
    $expected = $expectedOutputs[$i]

    $output = $MyNetwork.Predict(@($input))
    
    Write-Host "Input: ($($inputInt)): $($input)"
    Write-Host "`tExpected: ($($expectedInt)): $($expected)"
    Write-Host "`tPredicted: $output"
}
