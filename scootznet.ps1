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

$NETWORK_LAYERS = @(8, 8)
$LEARNING_RATE = 0.001
$RANDOM_DATA_SIZE = 32
$TRAINING_PASS_EPOCHS = 1000
$TRAINING_PASSES = 100

$RESULTS_FILE = "$($PSScriptRoot)\ScootznetTrainingResults.json"

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

Function Smooth-Output ($RawOutput)
{
    $SmoothedOutput = @()
    $RawOutput | ForEach-Object { $SmoothedOutput += [Math]::Round($_) }
    return $SmoothedOutput
}

Function Actual-Delta ($Expected, $Actual)
{
    $Different = 0
    for ($i = 0; $i -lt $Expected.Length; $i++)
    {
        $Different += [Math]::Abs($Expected[$i] - $Actual[$i])
    }
    return ($Different / $Expected.Length)
}

Write-Host "Creating a neural network with $($NETWORK_LAYERS.Length) layers: $($NETWORK_LAYERS -join '-')..."
$MyNetwork = [NeuralNetwork]::new($NETWORK_LAYERS, $LEARNING_RATE)

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

$ActualDeltas = @()
for ($i = 0; $i -lt $testInputs.Length; $i++)
{
    $inputInt = $testInputsInt[$i]
    $input = $testInputs[$i]
    $expectedInt = $expectedOutputsInt[$i]
    $expected = $expectedOutputs[$i]

    $output = $MyNetwork.Predict(@($input))
    $smoothedOutput = Smooth-Output $output
    $smoothedOutputInt = From-Binary $smoothedOutput
    $actualDelta = Actual-Delta $expected $smoothedOutput

    $ActualDeltas += $actualDelta

    $deltaColor = if ($actualDelta -eq 0) { 'Green' } elseif ($actualDelta -lt 0.2) { 'Yellow' } elseif ($actualDelta -lt 0.6) { 'Magenta' } else { 'Red' }
    Write-Host "Input: $($input) ($($inputInt))"
    Write-Host "`tExpected: $($expected) ($($expectedInt))"
    Write-Host "`tPredicted (raw): $output"
    Write-Host "`tPredicted (smoothed): $smoothedOutput ($($smoothedOutputInt))"
    Write-Host "`tActual delta: " -NoNewLine
    Write-Host "$actualDelta" -ForegroundColor $deltaColor
}

$ResultData = [PSCustomObject]@{
    'NetworkLayers' = $NETWORK_LAYERS -join '-';
    'LearningRate' = $LEARNING_RATE;
    'RandomDataSize' = $RANDOM_DATA_SIZE;
    'TrainingPassEpochs' = $TRAINING_PASS_EPOCHS;
    'TrainingPasses' = $TRAINING_PASSES;
    'TrainingTime' = $Stopwatch.Elapsed.TotalSeconds;
    'AverageDelta' = $(($ActualDeltas | Measure-Object -Average).Average);
}

Write-Host "Average delta: $(($ActualDeltas | Measure-Object -Average).Average)"

Write-Host "Updating results file $($RESULTS_FILE)..."
if (Test-Path -Path "$RESULTS_FILE")
{
    $Results = [array](Get-Content $RESULTS_FILE | ConvertFrom-Json)
    $Results += $ResultData
    $Results | ConvertTo-Json | Set-Content -Path $RESULTS_FILE

    # rewrite the json so members are all in the same order, ConvertTo-Json doesn't guarantee order
    $Results = [array](Get-Content $RESULTS_FILE | ConvertFrom-Json)
    $Results | ConvertTo-Json | Set-Content -Path $RESULTS_FILE
}
else
{
    $ResultData | ConvertTo-Json | Set-Content -Path $RESULTS_FILE
}

