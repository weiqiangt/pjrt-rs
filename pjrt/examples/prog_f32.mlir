module {
    func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
        %1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        %2 = stablehlo.add %arg0, %1 : tensor<f32>
        return %2 : tensor<f32>
    }
}   