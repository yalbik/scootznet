# Generic Test Runner for ScootzNet Project
# This script loads all necessary dependencies and runs all test scripts in the tests directory

param(
    [string]$TestPattern = "*.Tests.ps1",
    [switch]$Detailed = $false
)

# Get the script directory and construct paths
$TestsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $TestsDir
$LibDir = Join-Path $ProjectRoot "lib"

Write-Host "ScootzNet Test Runner" -ForegroundColor Magenta
Write-Host "=====================" -ForegroundColor Magenta
Write-Host "Project Root: $ProjectRoot" -ForegroundColor Gray
Write-Host "Library Dir:  $LibDir" -ForegroundColor Gray
Write-Host "Tests Dir:    $TestsDir" -ForegroundColor Gray
Write-Host ""

# Pre-load all necessary classes before running tests
Write-Host "Loading Dependencies:" -ForegroundColor Cyan

try {
    Write-Host "Loading Neuron class..." -ForegroundColor Green -NoNewline
    $NeuronPath = Join-Path $LibDir "Neuron.ps1"
    if (-not (Test-Path $NeuronPath)) {
        throw "Neuron.ps1 not found at: $NeuronPath"
    }
    . $NeuronPath
    Write-Host " OK" -ForegroundColor Green
    
    # Test Neuron class instantiation
    Write-Host "Verifying Neuron class..." -ForegroundColor Yellow -NoNewline
    $null = [Neuron]::new(1, $false)
    Write-Host " OK" -ForegroundColor Green
    
} catch {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Error "Failed to load Neuron class: $_"
    exit 1
}

try {
    Write-Host "Loading Layer class..." -ForegroundColor Green -NoNewline
    $LayerPath = Join-Path $LibDir "Layer.ps1"
    if (-not (Test-Path $LayerPath)) {
        throw "Layer.ps1 not found at: $LayerPath"
    }
    . $LayerPath
    Write-Host " OK" -ForegroundColor Green
    
    # Test Layer class instantiation
    Write-Host "Verifying Layer class..." -ForegroundColor Yellow -NoNewline
    $null = [Layer]::new(2, 3, $false)
    Write-Host " OK" -ForegroundColor Green
    
} catch {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Error "Failed to load Layer class: $_"
    exit 1
}

try {
    Write-Host "Loading NeuralNetwork class..." -ForegroundColor Green -NoNewline
    $NeuralNetworkPath = Join-Path $LibDir "NeuralNetwork.ps1"
    if (-not (Test-Path $NeuralNetworkPath)) {
        throw "NeuralNetwork.ps1 not found at: $NeuralNetworkPath"
    }
    . $NeuralNetworkPath
    Write-Host " OK" -ForegroundColor Green
    
    # Test NeuralNetwork class instantiation
    Write-Host "Verifying NeuralNetwork class..." -ForegroundColor Yellow -NoNewline
    $null = [NeuralNetwork]::new(@(2, 1), 0.1)
    Write-Host " OK" -ForegroundColor Green
    
} catch {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Error "Failed to load NeuralNetwork class: $_"
    exit 1
}

Write-Host ""
Write-Host "All dependencies loaded successfully!" -ForegroundColor Green
Write-Host ""

# Import Pester module
Write-Host "Importing Pester module..." -ForegroundColor Cyan
try {
    Import-Module Pester -Force
    Write-Host "Pester module imported successfully!" -ForegroundColor Green
} catch {
    Write-Error "Failed to import Pester module: $_"
    exit 1
}

Write-Host ""

# Find all test files
$TestFiles = Get-ChildItem -Path $TestsDir -Filter $TestPattern | Where-Object { $_.Name -ne "Simple.Tests.ps1" }

if ($TestFiles.Count -eq 0) {
    Write-Warning "No test files found matching pattern: $TestPattern"
    exit 1
}

Write-Host "Found $($TestFiles.Count) test file(s):" -ForegroundColor Cyan
foreach ($file in $TestFiles) {
    Write-Host "  - $($file.Name)" -ForegroundColor Gray
}
Write-Host ""

# Configure Pester
$PesterConfig = New-PesterConfiguration
$PesterConfig.Run.Path = $TestFiles.FullName
$PesterConfig.Output.Verbosity = if ($Detailed) { 'Detailed' } else { 'Normal' }
$PesterConfig.TestResult.Enabled = $true
$PesterConfig.TestResult.OutputPath = Join-Path $TestsDir "TestResults.xml"

# Run all tests
Write-Host "Running Tests..." -ForegroundColor Magenta
Write-Host "=================" -ForegroundColor Magenta

Invoke-Pester -Configuration $PesterConfig