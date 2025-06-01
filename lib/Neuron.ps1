class Neuron {
    [double[]] $Weights
    [double] $Bias
    [double] $Output
    [double] $Delta
    [int] $NumInputs

    # Constructor
    Neuron([int]$numInputs, [bool]$isOutputLayer) {
        $this.NumInputs = $numInputs
        $this.Weights = @(0) * $numInputs
        
        # Initialize weights and bias with small random values
        $rand = New-Object System.Random
        for ($i = 0; $i -lt $numInputs; $i++) {
            # Xavier initialization: range based on number of inputs
            $range = [Math]::Sqrt(6.0 / $numInputs)
            $this.Weights[$i] = ($rand.NextDouble() - 0.5) * 2 * $range
        }
        $this.Bias = ($rand.NextDouble() - 0.5) * 0.01
        
        $this.Output = 0.0
        $this.Delta = 0.0
    }

    # Activation function
    [double] Activate([double[]] $inputs, [bool] $useSigmoid) {
        # Validate inputs
        if ($inputs.Length -ne $this.NumInputs) {
            throw "Input length ($($inputs.Length)) does not match expected ($($this.NumInputs))"
        }

        # Calculate weighted sum
        [double] $sum = $this.Bias
        for ($i = 0; $i -lt $inputs.Length; $i++) {
            $sum += [double]$inputs[$i] * [double]$this.Weights[$i]
        }

        # Apply activation function
        if ($useSigmoid) {
            # Sigmoid activation with overflow protection
            $clampedSum = [Math]::Max(-500.0, [Math]::Min(500.0, $sum))
            $this.Output = 1.0 / (1.0 + [Math]::Exp(-$clampedSum))
        } else {
            # Linear activation for output layer
            $this.Output = $sum
        }

        return $this.Output
    }

    # Static sigmoid function
    static [double] Sigmoid([double] $x) {
        $clampedX = [Math]::Max(-500.0, [Math]::Min(500.0, $x))
        return 1.0 / (1.0 + [Math]::Exp(-$clampedX))
    }

    # Static sigmoid derivative function
    static [double] SigmoidDerivative([double] $sigmoidOutput) {
        return $sigmoidOutput * (1.0 - $sigmoidOutput)
    }
}