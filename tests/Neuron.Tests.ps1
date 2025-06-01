# Neuron class should be loaded by the test runner

Describe "Neuron Class Tests" {
    
    Describe "Constructor Tests" {
        It "Should create neuron with correct number of inputs" {
            $neuron = [Neuron]::new(3, $false)
            $neuron.NumInputs | Should -Be 3
        }

        It "Should initialize weights array with correct length" {
            $neuron = [Neuron]::new(5, $false)
            $neuron.Weights.Length | Should -Be 5
        }

        It "Should initialize weights with Xavier initialization range" {
            $neuron = [Neuron]::new(4, $false)
            $expectedRange = [Math]::Sqrt(6.0 / 4)
            
            # All weights should be within the Xavier initialization range
            foreach ($weight in $neuron.Weights) {
                [Math]::Abs($weight) | Should -BeLessOrEqual $expectedRange
            }
        }

        It "Should initialize bias with small random value" {
            $neuron = [Neuron]::new(3, $false)
            [Math]::Abs($neuron.Bias) | Should -BeLessOrEqual 0.005
        }

        It "Should initialize Output to zero" {
            $neuron = [Neuron]::new(3, $false)
            $neuron.Output | Should -Be 0.0
        }

        It "Should initialize Delta to zero" {
            $neuron = [Neuron]::new(3, $false)
            $neuron.Delta | Should -Be 0.0
        }

        It "Should handle single input neuron" {
            $neuron = [Neuron]::new(1, $false)
            $neuron.NumInputs | Should -Be 1
            $neuron.Weights.Length | Should -Be 1
        }

        It "Should handle large number of inputs" {
            $neuron = [Neuron]::new(100, $false)
            $neuron.NumInputs | Should -Be 100
            $neuron.Weights.Length | Should -Be 100        }
    }

    Describe "Activate Method Tests - Sigmoid Activation" {
        BeforeEach {
            $script:neuron = [Neuron]::new(3, $false)
            # Set known weights and bias for predictable testing
            $script:neuron.Weights = @(0.5, -0.3, 0.2)
            $script:neuron.Bias = 0.1
        }

        It "Should calculate correct output with sigmoid activation" {
            $inputs = @(1.0, 2.0, -1.0)
            $result = $script:neuron.Activate($inputs, $true)
            
            # Expected calculation: 0.1 + (1.0 * 0.5) + (2.0 * -0.3) + (-1.0 * 0.2) = -0.3
            # Sigmoid(-0.3) ≈ 0.4256
            $result | Should -BeGreaterThan 0.4
            $result | Should -BeLessThan 0.5
        }

        It "Should update Output property when activated" {
            $inputs = @(0.0, 0.0, 0.0)
            $result = $script:neuron.Activate($inputs, $true)
            $script:neuron.Output | Should -Be $result
        }

        It "Should handle zero inputs with sigmoid" {
            $inputs = @(0.0, 0.0, 0.0)
            $result = $script:neuron.Activate($inputs, $true)
            
            # Should return sigmoid of bias (0.1)
            $expected = [Neuron]::Sigmoid(0.1)
            $result | Should -Be $expected
        }

        It "Should handle large positive values without overflow" {
            $script:neuron.Weights = @(100.0, 100.0, 100.0)
            $script:neuron.Bias = 100.0
            $inputs = @(10.0, 10.0, 10.0)
            
            $result = $script:neuron.Activate($inputs, $true)
            $result | Should -BeGreaterThan 0.99
            $result | Should -BeLessOrEqual 1.0
        }

        It "Should handle large negative values without underflow" {
            $script:neuron.Weights = @(-100.0, -100.0, -100.0)
            $script:neuron.Bias = -100.0
            $inputs = @(10.0, 10.0, 10.0)
            
            $result = $script:neuron.Activate($inputs, $true)
            $result | Should -BeLessThan 0.01
            $result | Should -BeGreaterOrEqual 0.0
        }
    }

    Describe "Activate Method Tests - Linear Activation" {
        BeforeEach {
            $script:neuron = [Neuron]::new(2, $true)
            $script:neuron.Weights = @(2.0, -1.5)
            $script:neuron.Bias = 0.5
        }

        It "Should calculate correct output with linear activation" {
            $inputs = @(3.0, 2.0)
            $result = $script:neuron.Activate($inputs, $false)
            
            # Expected: 0.5 + (3.0 * 2.0) + (2.0 * -1.5) = 0.5 + 6.0 - 3.0 = 3.5
            $result | Should -Be 3.5
        }

        It "Should return exact weighted sum for linear activation" {
            $inputs = @(1.0, 1.0)
            $result = $script:neuron.Activate($inputs, $false)
            
            # Expected: 0.5 + (1.0 * 2.0) + (1.0 * -1.5) = 1.0
            $result | Should -Be 1.0
        }

        It "Should handle negative outputs with linear activation" {
            $inputs = @(-1.0, 2.0)
            $result = $script:neuron.Activate($inputs, $false)
            
            # Expected: 0.5 + (-1.0 * 2.0) + (2.0 * -1.5) = 0.5 - 2.0 - 3.0 = -4.5
            $result | Should -Be -4.5
        }
    }

    Describe "Input Validation Tests" {
        BeforeEach {
            $script:neuron = [Neuron]::new(3, $false)
        }

        It "Should throw error when input length is too small" {
            $inputs = @(1.0, 2.0)  # Only 2 inputs, expected 3
            { $script:neuron.Activate($inputs, $true) } | Should -Throw "*Input length (2) does not match expected (3)*"
        }

        It "Should throw error when input length is too large" {
            $inputs = @(1.0, 2.0, 3.0, 4.0)  # 4 inputs, expected 3
            { $script:neuron.Activate($inputs, $true) } | Should -Throw "*Input length (4) does not match expected (3)*"
        }

        It "Should throw error when no inputs provided" {
            $inputs = @()
            { $script:neuron.Activate($inputs, $true) } | Should -Throw "*Input length (0) does not match expected (3)*"
        }

        It "Should accept exact number of inputs" {
            $inputs = @(1.0, 2.0, 3.0)
            { $script:neuron.Activate($inputs, $true) } | Should -Not -Throw
        }
    }

    Describe "Static Sigmoid Method Tests" {
        It "Should return 0.5 for input 0" {
            $result = [Neuron]::Sigmoid(0)
            $result | Should -Be 0.5
        }

        It "Should return value close to 1 for large positive input" {
            $result = [Neuron]::Sigmoid(10)
            $result | Should -BeGreaterThan 0.99
            $result | Should -BeLessOrEqual 1.0
        }

        It "Should return value close to 0 for large negative input" {
            $result = [Neuron]::Sigmoid(-10)
            $result | Should -BeLessThan 0.01
            $result | Should -BeGreaterOrEqual 0.0
        }

        It "Should handle extreme positive values without overflow" {
            $result = [Neuron]::Sigmoid(1000)
            $result | Should -BeGreaterThan 0.99
            $result | Should -BeLessOrEqual 1.0
        }

        It "Should handle extreme negative values without underflow" {
            $result = [Neuron]::Sigmoid(-1000)
            $result | Should -BeLessThan 0.01
            $result | Should -BeGreaterOrEqual 0.0
        }

        It "Should be monotonically increasing" {
            $result1 = [Neuron]::Sigmoid(-1)
            $result2 = [Neuron]::Sigmoid(0)
            $result3 = [Neuron]::Sigmoid(1)
            
            $result1 | Should -BeLessThan $result2
            $result2 | Should -BeLessThan $result3
        }
    }

    Describe "Static SigmoidDerivative Method Tests" {
        It "Should return 0.25 for sigmoid output 0.5" {
            $result = [Neuron]::SigmoidDerivative(0.5)
            $result | Should -Be 0.25
        }

        It "Should return 0 for sigmoid output 0" {
            $result = [Neuron]::SigmoidDerivative(0)
            $result | Should -Be 0
        }

        It "Should return 0 for sigmoid output 1" {
            $result = [Neuron]::SigmoidDerivative(1)
            $result | Should -Be 0
        }

        It "Should return maximum value at 0.5 input" {
            $result1 = [Neuron]::SigmoidDerivative(0.3)
            $result2 = [Neuron]::SigmoidDerivative(0.5)
            $result3 = [Neuron]::SigmoidDerivative(0.7)
            
            $result2 | Should -BeGreaterThan $result1
            $result2 | Should -BeGreaterThan $result3
        }

        It "Should handle edge cases gracefully" {
            # Test very small positive value
            $result1 = [Neuron]::SigmoidDerivative(0.001)
            $result1 | Should -BeGreaterOrEqual 0
            
            # Test very close to 1
            $result2 = [Neuron]::SigmoidDerivative(0.999)
            $result2 | Should -BeGreaterOrEqual 0
        }
    }

    Describe "Integration Tests" {
        It "Should work correctly with sigmoid activation and derivative" {
            $neuron = [Neuron]::new(2, $false)
            $neuron.Weights = @(1.0, -1.0)
            $neuron.Bias = 0.0
            
            $inputs = @(0.5, 0.5)
            $output = $neuron.Activate($inputs, $true)
            $derivative = [Neuron]::SigmoidDerivative($output)
            
            # The derivative should be positive for any valid sigmoid output
            $derivative | Should -BeGreaterThan 0
            $derivative | Should -BeLessOrEqual 0.25  # Maximum derivative is 0.25
        }

        It "Should produce consistent results with same inputs" {
            $neuron = [Neuron]::new(3, $false)
            $inputs = @(1.0, 2.0, 3.0)
            
            $result1 = $neuron.Activate($inputs, $true)
            $result2 = $neuron.Activate($inputs, $true)
            
            $result1 | Should -Be $result2
        }

        It "Should handle mixed positive and negative weights correctly" {
            $neuron = [Neuron]::new(4, $false)
            $neuron.Weights = @(1.0, -1.0, 2.0, -2.0)
            $neuron.Bias = 0.5
            
            $inputs = @(1.0, 1.0, 1.0, 1.0)
            $result = $neuron.Activate($inputs, $false)
            
            # Expected: 0.5 + 1.0 - 1.0 + 2.0 - 2.0 = 0.5
            $result | Should -Be 0.5
        }
    }

    Describe "Performance and Edge Cases" {
        It "Should handle very small weights" {
            $neuron = [Neuron]::new(3, $false)
            $neuron.Weights = @(1e-10, 1e-10, 1e-10)
            $neuron.Bias = 1e-10
            
            $inputs = @(1.0, 1.0, 1.0)
            $result = $neuron.Activate($inputs, $true)
            
            # Should be very close to 0.5 due to tiny weights
            $result | Should -BeGreaterThan 0.49
            $result | Should -BeLessThan 0.51
        }

        It "Should handle zero weights" {
            $neuron = [Neuron]::new(2, $false)
            $neuron.Weights = @(0.0, 0.0)
            $neuron.Bias = 0.0
            
            $inputs = @(100.0, -100.0)
            $result = $neuron.Activate($inputs, $true)
            
            # Should return 0.5 (sigmoid of 0)
            $result | Should -Be 0.5
        }

        It "Should maintain precision with decimal inputs" {
            $neuron = [Neuron]::new(1, $false)
            $neuron.Weights = @(1.0)
            $neuron.Bias = 0.0
            
            $inputs = @(0.123456789)
            $result = $neuron.Activate($inputs, $false)
            
            $result | Should -Be 0.123456789
        }
    }
}
