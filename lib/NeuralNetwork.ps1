class NeuralNetwork {
    [Layer[]] $Layers
    [double] $LearningRate
    [double[]] $LastInput
    [int[]] $LayerSizes

    # Constructor
    NeuralNetwork([int[]] $layerSizes, [double] $learningRate) {
        $this.LayerSizes = $layerSizes
        $this.LearningRate = $learningRate
        $this.Layers = @()

        # Create layers
        for ($i = 0; $i -lt $layerSizes.Length; $i++) {
            $neuronCount = $layerSizes[$i]
            
            # Input size for this layer
            if ($i -eq 0) {
                # First layer (input layer) - neurons count equals input size
                $inputCount = $layerSizes[$i]
            } else {
                # Hidden/output layers - input size is previous layer's neuron count
                $inputCount = $layerSizes[$i - 1]
            }
            
            # Determine if this is the output layer
            $isOutputLayer = ($i -eq $layerSizes.Length - 1)
            
            # Create and add layer
            $this.Layers += [Layer]::new($neuronCount, $inputCount, $isOutputLayer)
        }

        # Report successful creation
        Write-Host ""
        Write-Host "=== Neural Network Successfully Created ==="
        Write-Host "Learning Rate: $learningRate"
        Write-Host "Total Layers: $($this.Layers.Length)"
        Write-Host "Network Architecture: [$($layerSizes -join ', ')]"
        
        for ($i = 0; $i -lt $this.Layers.Length; $i++) {
            $layerType = if ($i -eq 0) { "Input" } 
                        elseif ($i -eq $this.Layers.Length - 1) { "Output" }
                        else { "Hidden" }
            Write-Host "  Layer $i ($layerType): $($this.Layers[$i].NeuronCount) neuron(s)"
        }
        Write-Host "==========================================="
        Write-Host ""
    }

    # Forward pass through the network
    [double[]] Forward([double[]] $inputs) {
        $this.LastInput = $inputs
        $currentOutputs = $inputs

        for ($i = 0; $i -lt $this.Layers.Length; $i++) {
            # Use sigmoid for all layers except output (which uses linear)
            $useSigmoid = ($i -ne $this.Layers.Length - 1)
            $currentOutputs = $this.Layers[$i].Forward($currentOutputs, $useSigmoid)
        }

        return $currentOutputs
    }

    # Backward pass (backpropagation)
    [void] Backward([double[]] $expected) {
        # Calculate output layer delta (linear activation derivative = 1)
        $outputLayer = $this.Layers[-1]
        for ($i = 0; $i -lt $outputLayer.Neurons.Length; $i++) {
            $neuron = $outputLayer.Neurons[$i]
            $error = $expected[$i] - $neuron.Output
            $neuron.Delta = $error  # Linear activation derivative is 1
        }

        # Calculate hidden layer deltas (working backwards)
        for ($layerIndex = $this.Layers.Length - 2; $layerIndex -ge 0; $layerIndex--) {
            $currentLayer = $this.Layers[$layerIndex]
            $nextLayer = $this.Layers[$layerIndex + 1]

            for ($neuronIndex = 0; $neuronIndex -lt $currentLayer.Neurons.Length; $neuronIndex++) {
                $neuron = $currentLayer.Neurons[$neuronIndex]
                
                # Sum weighted deltas from next layer
                $weightedDeltaSum = 0.0
                for ($nextNeuronIndex = 0; $nextNeuronIndex -lt $nextLayer.Neurons.Length; $nextNeuronIndex++) {
                    $nextNeuron = $nextLayer.Neurons[$nextNeuronIndex]
                    $weightedDeltaSum += $nextNeuron.Delta * $nextNeuron.Weights[$neuronIndex]
                }
                
                # Apply sigmoid derivative
                $sigmoidDerivative = $neuron.Output * (1.0 - $neuron.Output)
                $neuron.Delta = $weightedDeltaSum * $sigmoidDerivative
            }
        }

        # Update weights and biases
        for ($layerIndex = 0; $layerIndex -lt $this.Layers.Length; $layerIndex++) {
            $layer = $this.Layers[$layerIndex]
            
            # Get inputs for this layer
            $layerInputs = if ($layerIndex -eq 0) { 
                $this.LastInput 
            } else { 
                $this.Layers[$layerIndex - 1].GetOutputs() 
            }

            # Update each neuron in the layer
            foreach ($neuron in $layer.Neurons) {
                # Update weights
                for ($weightIndex = 0; $weightIndex -lt $neuron.Weights.Length; $weightIndex++) {
                    $weightUpdate = $this.LearningRate * $neuron.Delta * $layerInputs[$weightIndex]
                    $neuron.Weights[$weightIndex] += $weightUpdate
                }
                
                # Update bias
                $biasUpdate = $this.LearningRate * $neuron.Delta
                $neuron.Bias += $biasUpdate
            }
        }
    }    # Train the network
    [void] Train([double[][]] $inputs, [double[][]] $targets, [int] $epochs) {
        Write-Host "Starting training for $epochs epochs..."
        $trainingStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        
        for ($epoch = 0; $epoch -lt $epochs; $epoch++) {
            # Progress reporting every 100 epochs
            if ($epoch % 100 -eq 0) {
                $elapsed = $trainingStopwatch.Elapsed.TotalSeconds
                if ($epoch -gt 0) {
                    $avgTimePerEpoch = $elapsed / $epoch
                    $remainingEpochs = $epochs - $epoch
                    $estimatedRemaining = $avgTimePerEpoch * $remainingEpochs
                    Write-Host "Epoch: $epoch / $epochs - Elapsed: $([Math]::Round($elapsed, 1))s, Estimated remaining: $([Math]::Round($estimatedRemaining, 1))s"
                } else {
                    Write-Host "Epoch: $epoch / $epochs - Starting training..."
                }
            }

            # Train on each sample
            for ($sampleIndex = 0; $sampleIndex -lt $inputs.Length; $sampleIndex++) {
                $this.Forward($inputs[$sampleIndex])
                $this.Backward($targets[$sampleIndex])
            }
        }
        
        $trainingStopwatch.Stop()
        Write-Host "Training completed! Total time: $([Math]::Round($trainingStopwatch.Elapsed.TotalSeconds, 2)) seconds"
    }

    # Make a prediction
    [double[]] Predict([double[]] $inputs) {
        return $this.Forward($inputs)
    }

    # Get network information
    [string] GetNetworkInfo() {
        $info = "Neural Network - Architecture: [$($this.LayerSizes -join ', ')], Learning Rate: $($this.LearningRate)"
        return $info
    }
}