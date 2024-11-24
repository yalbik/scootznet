# Define the Neuron class
class Neuron {
    [double[]]$Weights
    [double]$Bias
    [double]$Output
    [double]$Delta

    Neuron([int]$inputCount) {
        $this.Weights = @(for ($i = 0; $i -lt $inputCount; $i++) { Get-Random -Minimum -1.0 -Maximum 1.0 })
        $this.Bias = Get-Random -Minimum -1.0 -Maximum 1.0
    }

    [double] Activate([double[]]$inputs) {
        $sum = 0
        for ($i = 0; $i -lt $inputs.Length; $i++) {
            $sum += $inputs[$i] * $this.Weights[$i]
        }
        $sum += $this.Bias
        $this.Output = 1 / (1 + [math]::Exp(-$sum))  # sigmoid activation
        return $this.Output
    }
}
