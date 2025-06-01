# Comprehensive Unit Tests for the Layer Class

Describe "Layer Class Tests" {
    
    Describe "Constructor Tests" {
        It "Should create layer with correct neuron count" {
            $layer = [Layer]::new(3, 2, $false)
            $layer.NeuronCount | Should -Be 3
        }

        It "Should create layer with correct input count" {
            $layer = [Layer]::new(3, 2, $false)
            $layer.InputCount | Should -Be 2
        }

        It "Should initialize neurons array with correct length" {
            $layer = [Layer]::new(5, 3, $false)
            $layer.Neurons.Length | Should -Be 5
        }

        It "Should create all neurons with correct input count" {
            $layer = [Layer]::new(4, 3, $false)
            foreach ($neuron in $layer.Neurons) {
                $neuron.NumInputs | Should -Be 3
            }
        }

        It "Should handle single neuron layer" {
            $layer = [Layer]::new(1, 5, $false)
            $layer.NeuronCount | Should -Be 1
            $layer.Neurons.Length | Should -Be 1
            $layer.Neurons[0].NumInputs | Should -Be 5
        }

        It "Should handle large layer creation" {
            $layer = [Layer]::new(100, 50, $false)
            $layer.NeuronCount | Should -Be 100
            $layer.InputCount | Should -Be 50
            $layer.Neurons.Length | Should -Be 100
        }

        It "Should create output layer correctly" {
            $layer = [Layer]::new(3, 4, $true)
            $layer.NeuronCount | Should -Be 3
            $layer.InputCount | Should -Be 4
            # Note: The isOutputLayer parameter is passed to neurons but not stored in Layer
        }

        It "Should create hidden layer correctly" {
            $layer = [Layer]::new(3, 4, $false)
            $layer.NeuronCount | Should -Be 3
            $layer.InputCount | Should -Be 4
        }
    }

    Describe "Forward Method Tests - Sigmoid Activation" {
        BeforeEach {
            $script:layer = [Layer]::new(2, 3, $false)
            # Set known weights and biases for predictable testing
            $script:layer.Neurons[0].Weights = @(0.5, -0.3, 0.2)
            $script:layer.Neurons[0].Bias = 0.1
            $script:layer.Neurons[1].Weights = @(-0.4, 0.6, -0.1)
            $script:layer.Neurons[1].Bias = -0.2
        }

        It "Should return correct number of outputs" {
            $inputs = @(1.0, 2.0, -1.0)
            $outputs = $script:layer.Forward($inputs, $true)
            $outputs.Length | Should -Be 2
        }

        It "Should calculate correct outputs with known weights" {
            $inputs = @(1.0, 0.0, 0.0)
            $outputs = $script:layer.Forward($inputs, $true)
            
            # First neuron: 0.1 + (1.0 * 0.5) = 0.6, sigmoid(0.6) ≈ 0.646
            # Second neuron: -0.2 + (1.0 * -0.4) = -0.6, sigmoid(-0.6) ≈ 0.354
            $outputs[0] | Should -BeGreaterThan 0.6
            $outputs[0] | Should -BeLessThan 0.7
            $outputs[1] | Should -BeGreaterThan 0.3
            $outputs[1] | Should -BeLessThan 0.4
        }

        It "Should handle zero inputs" {
            $inputs = @(0.0, 0.0, 0.0)
            $outputs = $script:layer.Forward($inputs, $true)
            
            # Should return sigmoid of biases
            $expected1 = [Neuron]::Sigmoid(0.1)
            $expected2 = [Neuron]::Sigmoid(-0.2)
            $outputs[0] | Should -Be $expected1
            $outputs[1] | Should -Be $expected2
        }

        It "Should update neuron outputs after forward pass" {
            $inputs = @(1.0, 2.0, -1.0)
            $outputs = $script:layer.Forward($inputs, $true)
            
            $script:layer.Neurons[0].Output | Should -Be $outputs[0]
            $script:layer.Neurons[1].Output | Should -Be $outputs[1]
        }
    }

    Describe "Forward Method Tests - Linear Activation" {
        BeforeEach {
            $script:layer = [Layer]::new(2, 2, $true)
            $script:layer.Neurons[0].Weights = @(1.0, -1.0)
            $script:layer.Neurons[0].Bias = 0.5
            $script:layer.Neurons[1].Weights = @(2.0, 0.5)
            $script:layer.Neurons[1].Bias = -1.0
        }

        It "Should calculate correct linear outputs" {
            $inputs = @(2.0, 3.0)
            $outputs = $script:layer.Forward($inputs, $false)
            
            # First neuron: 0.5 + (2.0 * 1.0) + (3.0 * -1.0) = -0.5
            # Second neuron: -1.0 + (2.0 * 2.0) + (3.0 * 0.5) = 4.5
            $outputs[0] | Should -Be -0.5
            $outputs[1] | Should -Be 4.5
        }

        It "Should handle negative outputs with linear activation" {
            $inputs = @(-1.0, 1.0)
            $outputs = $script:layer.Forward($inputs, $false)
            
            # First neuron: 0.5 + (-1.0 * 1.0) + (1.0 * -1.0) = -1.5
            # Second neuron: -1.0 + (-1.0 * 2.0) + (1.0 * 0.5) = -2.5
            $outputs[0] | Should -Be -1.5
            $outputs[1] | Should -Be -2.5
        }
    }

    Describe "Input Validation Tests" {
        BeforeEach {
            $script:layer = [Layer]::new(3, 4, $false)
        }

        It "Should throw error when input length is too small" {
            $inputs = @(1.0, 2.0, 3.0)  # Only 3 inputs, expected 4
            { $script:layer.Forward($inputs, $true) } | Should -Throw "*Input length (3) does not match layer's expected input count (4)*"
        }

        It "Should throw error when input length is too large" {
            $inputs = @(1.0, 2.0, 3.0, 4.0, 5.0)  # 5 inputs, expected 4
            { $script:layer.Forward($inputs, $true) } | Should -Throw "*Input length (5) does not match layer's expected input count (4)*"
        }

        It "Should throw error when no inputs provided" {
            $inputs = @()
            { $script:layer.Forward($inputs, $true) } | Should -Throw "*Input length (0) does not match layer's expected input count (4)*"
        }

        It "Should accept exact number of inputs" {
            $inputs = @(1.0, 2.0, 3.0, 4.0)
            { $script:layer.Forward($inputs, $true) } | Should -Not -Throw
        }
    }

    Describe "GetOutputs Method Tests" {
        BeforeEach {
            $script:layer = [Layer]::new(3, 2, $false)
        }

        It "Should return array with correct length" {
            $inputs = @(1.0, 2.0)
            $script:layer.Forward($inputs, $true)
            $outputs = $script:layer.GetOutputs()
            $outputs.Length | Should -Be 3
        }

        It "Should return same values as Forward method" {
            $inputs = @(0.5, -0.5)
            $forwardOutputs = $script:layer.Forward($inputs, $true)
            $getOutputs = $script:layer.GetOutputs()
            
            for ($i = 0; $i -lt $forwardOutputs.Length; $i++) {
                $getOutputs[$i] | Should -Be $forwardOutputs[$i]
            }
        }

        It "Should return updated outputs after multiple forward passes" {
            $inputs1 = @(1.0, 0.0)
            $inputs2 = @(0.0, 1.0)
            
            $script:layer.Forward($inputs1, $true)
            $outputs1 = $script:layer.GetOutputs()
            
            $script:layer.Forward($inputs2, $true)
            $outputs2 = $script:layer.GetOutputs()
            
            # Outputs should be different for different inputs
            $different = $false
            for ($i = 0; $i -lt $outputs1.Length; $i++) {
                if ($outputs1[$i] -ne $outputs2[$i]) {
                    $different = $true
                    break
                }
            }
            $different | Should -Be $true
        }

        It "Should work with single neuron layer" {
            $singleLayer = [Layer]::new(1, 3, $false)
            $inputs = @(1.0, 2.0, 3.0)
            $singleLayer.Forward($inputs, $true)
            $outputs = $singleLayer.GetOutputs()
            $outputs.Length | Should -Be 1
        }
    }

    Describe "GetInfo Method Tests" {
        It "Should return correct information string for small layer" {
            $layer = [Layer]::new(3, 2, $false)
            $info = $layer.GetInfo()
            $info | Should -Be "Layer: 3 neurons, 2 inputs per neuron"
        }

        It "Should return correct information string for large layer" {
            $layer = [Layer]::new(100, 50, $false)
            $info = $layer.GetInfo()
            $info | Should -Be "Layer: 100 neurons, 50 inputs per neuron"
        }

        It "Should return correct information string for single neuron" {
            $layer = [Layer]::new(1, 10, $false)
            $info = $layer.GetInfo()
            $info | Should -Be "Layer: 1 neurons, 10 inputs per neuron"
        }
    }

    Describe "Integration Tests" {
        It "Should work with multiple forward passes" {
            $layer = [Layer]::new(2, 3, $false)
            
            $inputs1 = @(1.0, 0.0, -1.0)
            $inputs2 = @(0.0, 1.0, 0.0)
            $inputs3 = @(-1.0, -1.0, 1.0)
            
            $outputs1 = $layer.Forward($inputs1, $true)
            $outputs2 = $layer.Forward($inputs2, $true)
            $outputs3 = $layer.Forward($inputs3, $true)
            
            # All outputs should be valid (between 0 and 1 for sigmoid)
            foreach ($output in $outputs1 + $outputs2 + $outputs3) {
                $output | Should -BeGreaterOrEqual 0.0
                $output | Should -BeLessOrEqual 1.0
            }
        }

        It "Should maintain consistency between Forward and GetOutputs" {
            $layer = [Layer]::new(4, 5, $false)
            $inputs = @(0.1, 0.2, 0.3, 0.4, 0.5)
            
            for ($i = 0; $i -lt 10; $i++) {
                $forwardResult = $layer.Forward($inputs, $true)
                $getOutputsResult = $layer.GetOutputs()
                
                for ($j = 0; $j -lt $forwardResult.Length; $j++) {
                    $getOutputsResult[$j] | Should -Be $forwardResult[$j]
                }
            }
        }

        It "Should handle alternating sigmoid and linear activations" {
            $layer = [Layer]::new(2, 2, $false)
            $inputs = @(1.0, -1.0)
            
            $sigmoidOutputs = $layer.Forward($inputs, $true)
            $linearOutputs = $layer.Forward($inputs, $false)
            
            # Sigmoid outputs should be between 0 and 1
            foreach ($output in $sigmoidOutputs) {
                $output | Should -BeGreaterOrEqual 0.0
                $output | Should -BeLessOrEqual 1.0
            }
            
            # Linear outputs can be any value
            $linearOutputs.Length | Should -Be 2
        }
    }

    Describe "Performance and Edge Cases" {
        It "Should handle very small input values" {
            $layer = [Layer]::new(2, 3, $false)
            $inputs = @(1e-10, -1e-10, 1e-10)
            
            $outputs = $layer.Forward($inputs, $true)
            $outputs.Length | Should -Be 2
            
            # All outputs should be valid numbers
            foreach ($output in $outputs) {
                $output | Should -BeOfType [double]
                [double]::IsNaN($output) | Should -Be $false
                [double]::IsInfinity($output) | Should -Be $false
            }
        }

        It "Should handle large input values" {
            $layer = [Layer]::new(2, 2, $false)
            $inputs = @(1000.0, -1000.0)
            
            $outputs = $layer.Forward($inputs, $true)
            
            # Should handle overflow protection
            foreach ($output in $outputs) {
                $output | Should -BeGreaterOrEqual 0.0
                $output | Should -BeLessOrEqual 1.0
            }
        }

        It "Should maintain precision with decimal inputs" {
            $layer = [Layer]::new(1, 1, $false)
            $layer.Neurons[0].Weights = @(1.0)
            $layer.Neurons[0].Bias = 0.0
            
            $inputs = @(0.123456789)
            $outputs = $layer.Forward($inputs, $false)
            
            $outputs[0] | Should -Be 0.123456789
        }

        It "Should handle zero neuron weights correctly" {
            $layer = [Layer]::new(2, 3, $false)
            
            # Set all weights to zero
            foreach ($neuron in $layer.Neurons) {
                $neuron.Weights = @(0.0, 0.0, 0.0)
                $neuron.Bias = 0.0
            }
            
            $inputs = @(100.0, -100.0, 50.0)
            $outputs = $layer.Forward($inputs, $true)
            
            # All outputs should be 0.5 (sigmoid of 0)
            foreach ($output in $outputs) {
                $output | Should -Be 0.5
            }
        }
    }
}
