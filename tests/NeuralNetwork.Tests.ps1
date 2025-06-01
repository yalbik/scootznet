
# Load the Neuron class first (dependency)
. "$PSScriptRoot\..\lib\Neuron.ps1"

# Load the Layer class (dependency)
. "$PSScriptRoot\..\lib\Layer.ps1"

# Load the NeuralNetwork class
. "$PSScriptRoot\..\lib\NeuralNetwork.ps1"

Describe "NeuralNetwork Class Tests" {
    
    Context "Constructor Tests" {
        It "Should create a simple network with correct properties" {
            $layerSizes = @(2, 3, 1)
            $learningRate = 0.1
            $network = [NeuralNetwork]::new($layerSizes, $learningRate)
            
            $network.LayerSizes | Should -Be $layerSizes
            $network.LearningRate | Should -Be $learningRate
            $network.Layers.Count | Should -Be 3
        }
        
        It "Should create layers with correct neuron counts" {
            $layerSizes = @(3, 5, 2)
            $network = [NeuralNetwork]::new($layerSizes, 0.01)
            
            $network.Layers[0].NeuronCount | Should -Be 3
            $network.Layers[1].NeuronCount | Should -Be 5
            $network.Layers[2].NeuronCount | Should -Be 2
        }
        
        It "Should create layers with correct input counts" {
            $layerSizes = @(4, 3, 1)
            $network = [NeuralNetwork]::new($layerSizes, 0.05)
            
            # First layer input count equals neuron count (input layer)
            $network.Layers[0].InputCount | Should -Be 4
            # Second layer input count equals previous layer neuron count
            $network.Layers[1].InputCount | Should -Be 4
            # Third layer input count equals previous layer neuron count
            $network.Layers[2].InputCount | Should -Be 3
        }
        
        It "Should create single layer network" {
            $layerSizes = @(2)
            $network = [NeuralNetwork]::new($layerSizes, 0.1)
            
            $network.Layers.Count | Should -Be 1
            $network.Layers[0].NeuronCount | Should -Be 2
            $network.Layers[0].InputCount | Should -Be 2
        }
        
        It "Should create large network" {
            $layerSizes = @(10, 15, 8, 3)
            $network = [NeuralNetwork]::new($layerSizes, 0.001)
            
            $network.Layers.Count | Should -Be 4
            $network.Layers[0].NeuronCount | Should -Be 10
            $network.Layers[1].NeuronCount | Should -Be 15
            $network.Layers[2].NeuronCount | Should -Be 8
            $network.Layers[3].NeuronCount | Should -Be 3
        }
        
        It "Should handle different learning rates" {
            $rates = @(0.01, 0.1, 0.5, 1.0)
            foreach ($rate in $rates) {
                $network = [NeuralNetwork]::new(@(2, 1), $rate)
                $network.LearningRate | Should -Be $rate
            }
        }
        
        It "Should throw error for empty layer sizes" {
            { [NeuralNetwork]::new(@(), 0.1) } | Should -Throw
        }
        
        It "Should throw error for negative learning rate" {
            { [NeuralNetwork]::new(@(2, 1), -0.1) } | Should -Throw
        }
        
        It "Should throw error for zero neuron count in layer" {
            { [NeuralNetwork]::new(@(2, 0, 1), 0.1) } | Should -Throw
        }
    }
    
    Context "Forward Method Tests" {
        BeforeEach {
            $script:network = [NeuralNetwork]::new(@(2, 3, 1), 0.1)
        }
        
        It "Should process inputs and return outputs" {
            $inputs = @(1.0, 0.5)
            $outputs = $script:network.Forward($inputs)
            
            $outputs.Count | Should -Be 1
            $outputs[0] | Should -BeOfType [double]
        }
        
        It "Should store last input" {
            $inputs = @(0.7, -0.3)
            $script:network.Forward($inputs)
            
            $script:network.LastInput.Count | Should -Be 2
            $script:network.LastInput[0] | Should -Be 0.7
            $script:network.LastInput[1] | Should -Be -0.3
        }
        
        It "Should handle zero inputs" {
            $inputs = @(0.0, 0.0)
            $outputs = $script:network.Forward($inputs)
            
            $outputs.Count | Should -Be 1
            $outputs[0] | Should -BeOfType [double]
        }
        
        It "Should handle negative inputs" {
            $inputs = @(-1.0, -2.0)
            $outputs = $script:network.Forward($inputs)
            
            $outputs.Count | Should -Be 1
            $outputs[0] | Should -BeOfType [double]
        }
        
        It "Should produce consistent outputs for same inputs" {
            $inputs = @(0.5, 0.8)
            $outputs1 = $script:network.Forward($inputs)
            $outputs2 = $script:network.Forward($inputs)
            
            $outputs1[0] | Should -Be $outputs2[0]
        }
        
        It "Should throw error for incorrect input size" {
            $inputs = @(1.0)  # Expected 2, provided 1
            { $script:network.Forward($inputs) } | Should -Throw
        }
        
        It "Should throw error for too many inputs" {
            $inputs = @(1.0, 0.5, 0.3)  # Expected 2, provided 3
            { $script:network.Forward($inputs) } | Should -Throw
        }
    }
    
    Context "Backward Method Tests" {
        BeforeEach {
            $script:network = [NeuralNetwork]::new(@(2, 2, 1), 0.1)
            # Perform forward pass first
            $script:network.Forward(@(1.0, 0.5))
        }
        
        It "Should execute backward pass without error" {
            $expected = @(0.8)
            { $script:network.Backward($expected) } | Should -Not -Throw
        }
        
        It "Should update neuron deltas" {
            $expected = @(1.0)
            $script:network.Backward($expected)
            
            # Check that deltas are set (not zero for all neurons)
            $deltaSet = $false
            foreach ($layer in $script:network.Layers) {
                foreach ($neuron in $layer.Neurons) {
                    if ([Math]::Abs($neuron.Delta) -gt 0.001) {
                        $deltaSet = $true
                        break
                    }
                }
            }
            $deltaSet | Should -Be $true
        }
        
        It "Should update weights during backpropagation" {
            # Store original weights
            $originalWeights = @()
            foreach ($layer in $script:network.Layers) {
                foreach ($neuron in $layer.Neurons) {
                    $originalWeights += $neuron.Weights.Clone()
                }
            }
            
            $expected = @(0.5)
            $script:network.Backward($expected)
            
            # Check that at least some weights changed
            $weightsChanged = $false
            $weightIndex = 0
            foreach ($layer in $script:network.Layers) {
                foreach ($neuron in $layer.Neurons) {
                    for ($i = 0; $i -lt $neuron.Weights.Count; $i++) {
                        if ([Math]::Abs($neuron.Weights[$i] - $originalWeights[$weightIndex][$i]) -gt 0.001) {
                            $weightsChanged = $true
                            break
                        }
                    }
                    $weightIndex++
                }
            }
            $weightsChanged | Should -Be $true
        }
        
        It "Should update biases during backpropagation" {
            # Store original biases
            $originalBiases = @()
            foreach ($layer in $script:network.Layers) {
                foreach ($neuron in $layer.Neurons) {
                    $originalBiases += $neuron.Bias
                }
            }
            
            $expected = @(0.3)
            $script:network.Backward($expected)
            
            # Check that at least some biases changed
            $biasesChanged = $false
            $biasIndex = 0
            foreach ($layer in $script:network.Layers) {
                foreach ($neuron in $layer.Neurons) {
                    if ([Math]::Abs($neuron.Bias - $originalBiases[$biasIndex]) -gt 0.001) {
                        $biasesChanged = $true
                        break
                    }
                    $biasIndex++
                }
            }
            $biasesChanged | Should -Be $true
        }
        
        It "Should throw error for incorrect expected output size" {
            $expected = @(0.5, 0.3)  # Expected 1, provided 2
            { $script:network.Backward($expected) } | Should -Throw
        }
        
        It "Should require forward pass before backward pass" {
            $freshNetwork = [NeuralNetwork]::new(@(2, 1), 0.1)
            { $freshNetwork.Backward(@(1.0)) } | Should -Throw
        }
    }
    
    Context "Train Method Tests" {
        BeforeEach {
            $script:network = [NeuralNetwork]::new(@(2, 3, 1), 0.1)
        }
          It "Should train with single sample" {
            $inputs = ,@(1.0, 0.0)
            $targets = ,@(1.0)
            $epochs = 5
            
            { $script:network.Train($inputs, $targets, $epochs) } | Should -Not -Throw
        }
        
        It "Should train with multiple samples" {
            $inputs = @(
                @(1.0, 0.0),
                @(0.0, 1.0),
                @(1.0, 1.0),
                @(0.0, 0.0)
            )
            $targets = @(
                @(1.0),
                @(1.0),
                @(0.0),
                @(0.0)
            )
            $epochs = 10
            
            { $script:network.Train($inputs, $targets, $epochs) } | Should -Not -Throw
        }
          It "Should improve predictions with training" {
            $inputs = ,@(1.0, 0.0)
            $targets = ,@(1.0)
            
            # Get initial prediction
            $initialPrediction = $script:network.Predict($inputs[0])
            $initialError = [Math]::Abs($targets[0][0] - $initialPrediction[0])
            
            # Train
            $script:network.Train($inputs, $targets, 50)
            
            # Get final prediction
            $finalPrediction = $script:network.Predict($inputs[0])
            $finalError = [Math]::Abs($targets[0][0] - $finalPrediction[0])
            
            # Error should decrease
            $finalError | Should -BeLessThan $initialError
        }
          It "Should handle zero epochs" {
            $inputs = ,@(1.0, 0.0)
            $targets = ,@(1.0)
            
            { $script:network.Train($inputs, $targets, 0) } | Should -Not -Throw
        }
        
        It "Should throw error for mismatched input/target counts" {
            $inputs = @(@(1.0, 0.0), @(0.0, 1.0))
            $targets = @(@(1.0))  # Only one target for two inputs
            
            { $script:network.Train($inputs, $targets, 1) } | Should -Throw
        }
    }
    
    Context "Predict Method Tests" {
        BeforeEach {
            $script:network = [NeuralNetwork]::new(@(2, 2, 1), 0.1)
        }
          It "Should return prediction for valid input" {
            $inputData = @(0.5, -0.3)
            $prediction = $script:network.Predict($inputData)
            
            $prediction.Count | Should -Be 1
            $prediction[0] | Should -BeOfType [double]
        }
        
        It "Should be equivalent to Forward method" {
            $inputData = @(0.7, 0.2)
            $forwardResult = $script:network.Forward($inputData)
            $predictResult = $script:network.Predict($inputData)
            
            $predictResult[0] | Should -Be $forwardResult[0]
        }
        
        It "Should handle edge case inputs" {
            $edgeCases = @(
                @(0.0, 0.0),
                @(1.0, 1.0),
                @(-1.0, -1.0),
                @(100.0, -100.0)
            )
            
            foreach ($input in $edgeCases) {
                $prediction = $script:network.Predict($input)
                $prediction.Count | Should -Be 1
                $prediction[0] | Should -BeOfType [double]
            }
        }
    }
    
    Context "GetNetworkInfo Method Tests" {
        It "Should return correct info for simple network" {
            $network = [NeuralNetwork]::new(@(2, 3, 1), 0.1)
            $info = $network.GetNetworkInfo()
            
            $info | Should -Match "Neural Network"
            $info | Should -Match "\[2, 3, 1\]"
            $info | Should -Match "0.1"
        }
        
        It "Should return correct info for complex network" {
            $network = [NeuralNetwork]::new(@(10, 15, 8, 3), 0.001)
            $info = $network.GetNetworkInfo()
            
            $info | Should -Match "\[10, 15, 8, 3\]"
            $info | Should -Match "0.001"
        }
        
        It "Should return correct info for single layer network" {
            $network = [NeuralNetwork]::new(@(5), 0.5)
            $info = $network.GetNetworkInfo()
            
            $info | Should -Match "\[5\]"
            $info | Should -Match "0.5"
        }
    }
    
    Context "Integration Tests" {
        It "Should solve simple XOR-like problem" {
            $network = [NeuralNetwork]::new(@(2, 4, 1), 0.5)
            
            # Simple AND gate data
            $inputs = @(
                @(0.0, 0.0),
                @(0.0, 1.0),
                @(1.0, 0.0),
                @(1.0, 1.0)
            )
            $targets = @(
                @(0.0),
                @(0.0),
                @(0.0),
                @(1.0)
            )
            
            # Train the network
            $network.Train($inputs, $targets, 100)
              # Test predictions
            $predictions = @()
            foreach ($inputSample in $inputs) {
                $predictions += $network.Predict($inputSample)[0]
            }
            
            # Should learn the pattern reasonably well
            $predictions[0] | Should -BeLessThan 0.3  # 0,0 -> 0
            $predictions[1] | Should -BeLessThan 0.3  # 0,1 -> 0
            $predictions[2] | Should -BeLessThan 0.3  # 1,0 -> 0
            $predictions[3] | Should -BeGreaterThan 0.7  # 1,1 -> 1
        }
        
        It "Should handle sequential training sessions" {
            $network = [NeuralNetwork]::new(@(1, 2, 1), 0.1)
            
            # First training session
            $inputs1 = @(@(0.0), @(0.5))
            $targets1 = @(@(0.0), @(0.5))
            $network.Train($inputs1, $targets1, 10)
            
            # Second training session
            $inputs2 = @(@(1.0))
            $targets2 = @(@(1.0))
            $network.Train($inputs2, $targets2, 10)
            
            # Should work without errors
            $prediction = $network.Predict(@(0.8))
            $prediction.Count | Should -Be 1
        }
          It "Should maintain network state across operations" {
            $network = [NeuralNetwork]::new(@(3, 2, 1), 0.1)
            
            # Multiple forward passes
            $network.Forward(@(1.0, 0.0, 0.5))
            $network.Forward(@(0.0, 1.0, 0.3))
            $lastInput1 = $network.LastInput
              # Training
            $trainInputs = ,@(0.5, 0.5, 0.5)
            $trainTargets = ,@(0.8)
            $network.Train($trainInputs, $trainTargets, 5)
            
            # More operations
            $network.Forward(@(-0.2, 0.7, 1.0))
            $lastInput2 = $network.LastInput
            
            # Last input should be updated
            $lastInput2 | Should -Not -Be $lastInput1
            $lastInput2[0] | Should -Be -0.2
            $lastInput2[1] | Should -Be 0.7
            $lastInput2[2] | Should -Be 1.0
        }
    }
    
    Context "Performance and Edge Cases" {        It "Should handle very small learning rates" {
            $network = [NeuralNetwork]::new(@(2, 1), 0.0001)
            $inputs = ,@(1.0, 0.0)
            $targets = ,@(1.0)
            
            { $network.Train($inputs, $targets, 5) } | Should -Not -Throw
        }
        
        It "Should handle large learning rates" {
            $network = [NeuralNetwork]::new(@(2, 1), 10.0)
            # Network expects 2 inputs (same as first layer neuron count)
            $inputs = ,@(0.1, 0.2)
            $targets = ,@(0.5)
            
            { $network.Train($inputs, $targets, 2) } | Should -Not -Throw
        }
        
        It "Should handle extreme input values" {
            $network = [NeuralNetwork]::new(@(2, 1), 0.1)
            $extremeInputs = @(1000.0, -1000.0)
            
            $prediction = $network.Predict($extremeInputs)
            $prediction.Count | Should -Be 1
            $prediction[0] | Should -BeOfType [double]
            # Should not be NaN or Infinity
            [double]::IsNaN($prediction[0]) | Should -Be $false
            [double]::IsInfinity($prediction[0]) | Should -Be $false
        }
          It "Should maintain precision with repeated operations" {
            $network = [NeuralNetwork]::new(@(2, 2, 1), 0.01)
            $inputData = @(0.5, 0.7)
            
            # Repeated predictions should be identical
            $predictions = @()
            for ($i = 0; $i -lt 10; $i++) {
                $predictions += $network.Predict($inputData)[0]
            }
            
            # All predictions should be the same
            for ($i = 1; $i -lt $predictions.Count; $i++) {
                $predictions[$i] | Should -Be $predictions[0]
            }
        }
    }
}
