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

$MyNetwork = [NeuralNetwork]::new(@(8, 16, 8))

$inputs = @(
    @(0, 0, 0, 0, 0, 0, 0, 0),
    @(0, 0, 0, 0, 0, 0, 0, 1),
    @(0, 0, 0, 0, 0, 0, 1, 0),
    @(0, 0, 0, 0, 0, 1, 0, 0),
    @(0, 0, 0, 0, 1, 0, 0, 0),
    @(0, 0, 0, 1, 0, 0, 0, 0),
    @(0, 0, 1, 0, 0, 0, 0, 0),
    @(0, 1, 0, 0, 0, 0, 0, 0)
)

$outputs = @(
    @(0, 0, 0, 0, 0, 0, 0, 0),
    @(0, 0, 0, 0, 0, 0, 1, 0),
    @(0, 0, 0, 0, 0, 1, 0, 0),
    @(0, 0, 0, 0, 1, 0, 0, 0),
    @(0, 0, 0, 1, 0, 0, 0, 0),
    @(0, 0, 1, 0, 0, 0, 0, 0),
    @(0, 1, 0, 0, 0, 0, 0, 0),
    @(1, 0, 0, 0, 0, 0, 0, 0)
)

$epochs = 10000
Write-Host "Training the neural network with $($epochs) epochs..."
$Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$MyNetwork.Train($inputs, $outputs, $epochs)
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
