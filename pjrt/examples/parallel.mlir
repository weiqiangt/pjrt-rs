module @cross_replica {
  func.func @all_reduce(%operand : tensor<4xi64>) -> tensor<4xi64> {
    %result = "stablehlo.all_reduce"(%operand) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      replica_groups = dense<[[0, 1]]> : tensor<1x3xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
    } : (tensor<4xi64>) -> tensor<4xi64>
    return %result : tensor<4xi64>
  }
  func.func @main(%arg0: tensor<4xi64>) {
    %result = call @all_reduce(%arg0) : (tensor<4xi64>) -> tensor<4xi64>
    
    func.return
  }
}