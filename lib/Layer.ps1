# Define the Layer class
class Layer {
    [Neuron[]]$Neurons

    Layer([int]$neuronCount, [int]$inputCount) {
        $this.Neurons = @(for ($i = 0; $i -lt $neuronCount; $i++) { [Neuron]::new($inputCount) })
    }

    [double[]] Forward([double[]]$inputs) {
        return $this.Neurons | ForEach-Object { $_.Activate($inputs) }
    }
}
