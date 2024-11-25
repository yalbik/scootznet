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
$RANDOM_DATA_SIZE = 10
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

$Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
for ($Pass = 0; $Pass -lt $TRAINING_PASSES; $Pass++)
{
    Write-Host "Training pass $($Pass + 1) of $($TRAINING_PASSES)..."
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

$testInputs = @(
    @(0, 0, 0, 0, 0, 0, 0, 0),
    @(0, 0, 0, 0, 0, 0, 0, 1),
    @(0, 0, 0, 0, 0, 0, 1, 0),
    @(0, 0, 0, 0, 0, 1, 0, 0),
    @(0, 0, 0, 0, 1, 0, 0, 0),
    @(0, 0, 0, 1, 0, 0, 0, 0),
    @(0, 0, 1, 0, 0, 0, 0, 0),
    @(0, 1, 0, 0, 0, 0, 0, 0)
)

foreach ($input in $testInputs) {
    $output = $MyNetwork.Predict(@($input))
    Write-Host "Input: $input, Output: $output"
}
