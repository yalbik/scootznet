class NeuralNetwork {
    [Layer[]]$Layers
    [decimal]$LearningRate = 0.01

    NeuralNetwork([int[]]$layerSizes) {
        $this.Layers = @()
        for ($i = 0; $i -lt $layerSizes.Length - 1; $i++) {
            $this.Layers += [Layer]::new($layerSizes[$i + 1], $layerSizes[$i])
        }
    }

    [double[]] Forward([double[]]$inputs) {
        $outputs = $inputs
        foreach ($layer in $this.Layers) {
            $outputs = $layer.Forward($outputs)
        }
        return $outputs
    }

    [void] Backward([double[]]$expected) {
        # Calculate output layer delta
        $lastLayer = $this.Layers[-1]
        for ($i = 0; $i -lt $lastLayer.Neurons.Length; $i++) {
            $neuron = $lastLayer.Neurons[$i]
            $neuron.Delta = ($expected[$i] - $neuron.Output) * $neuron.Output * (1 - $neuron.Output)
        }

        # Calculate hidden layers delta
        for ($l = $this.Layers.Length - 2; $l -ge 0; $l--) {
            $layer = $this.Layers[$l]
            $nextLayer = $this.Layers[$l + 1]
            for ($i = 0; $i -lt $layer.Neurons.Length; $i++) {
                $neuron = $layer.Neurons[$i]
                $sum = 0
                for ($j = 0; $j -lt $nextLayer.Neurons.Length; $j++) {
                    $sum += $nextLayer.Neurons[$j].Weights[$i] * $nextLayer.Neurons[$j].Delta
                }
                $neuron.Delta = $sum * $neuron.Output * (1 - $neuron.Output)
            }
        }

        # Update weights and biases
        foreach ($layer in $this.Layers) {
            foreach ($neuron in $layer.Neurons) {
                for ($i = 0; $i -lt $neuron.Weights.Length; $i++) {
                    $neuron.Weights[$i] += $this.LearningRate * $neuron.Delta * $neuron.Output
                }
                $neuron.Bias += 0.1 * $neuron.Delta
            }
        }
    }

    [void] Train([double[][]]$inputs, [double[][]]$outputs, [int]$epochs) {
        for ($epoch = 0; $epoch -lt $epochs; $epoch++) {
            if ($epoch % 1000 -eq 0 -and $epoch -gt 0) { Write-Host "Epoch: $epoch" }
            for ($i = 0; $i -lt $inputs.Length; $i++) {
                $this.Forward($inputs[$i])
                $this.Backward($outputs[$i])
            }
        }
    }

    [double[]] Predict([double[]]$inputs) {
        return $this.Forward($inputs)
    }
}