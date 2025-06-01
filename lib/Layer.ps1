class Layer {
    [Neuron[]] $Neurons
    [int] $NeuronCount
    [int] $InputCount

    # Constructor
    Layer([int]$neuronCount, [int]$inputCount, [bool]$isOutputLayer = $false) {
        $this.NeuronCount = $neuronCount
        $this.InputCount = $inputCount
        
        # Create array of neurons
        $this.Neurons = @()
        for ($i = 0; $i -lt $neuronCount; $i++) {
            $this.Neurons += [Neuron]::new($inputCount, $isOutputLayer)
        }
        
        # Report successful construction
        $layerType = if ($isOutputLayer) { "output" } else { "hidden" }
        Write-Host "Successfully created $layerType layer with $neuronCount neuron(s), each with $inputCount input(s)."
    }

    # Forward pass through the layer
    [double[]] Forward([double[]] $inputs, [bool] $useSigmoid = $true) {
        # Validate inputs
        if ($inputs.Length -ne $this.InputCount) {
            throw "Input length ($($inputs.Length)) does not match layer's expected input count ($($this.InputCount))"
        }

        # Get output from each neuron
        $outputs = @()
        for ($i = 0; $i -lt $this.Neurons.Length; $i++) {
            $outputs += $this.Neurons[$i].Activate($inputs, $useSigmoid)
        }
        
        return $outputs
    }

    # Get all neuron outputs (for use in backpropagation)
    [double[]] GetOutputs() {
        $outputs = @()
        foreach ($neuron in $this.Neurons) {
            $outputs += $neuron.Output
        }
        return $outputs
    }

    # Get layer information
    [string] GetInfo() {
        return "Layer: $($this.NeuronCount) neurons, $($this.InputCount) inputs per neuron"
    }
}