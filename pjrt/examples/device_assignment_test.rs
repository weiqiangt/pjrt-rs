use pjrt::{self, Client, Result};

fn main() -> Result<()> {
    // 初始化插件，类似于 C++ 测试中的 setup
    let api = pjrt::plugin("pjrt_c_api_cpu_plugin.so").load()?;
    
    // 创建客户端
    let client = Client::builder(&api).build()?;
    
    // 测试参数，等价于 C++ 测试中的常量
    const NUM_REPLICAS: usize = 2;
    const NUM_PARTITIONS: usize = 1;
    
    // 调用 default_device_assignment 方法
    // 这等价于 C++ 测试中的 PJRT_Client_DefaultDeviceAssignment 调用
    let assignment = client.default_device_assignment(NUM_REPLICAS, NUM_PARTITIONS)?;
    
    println!("Test: GetDefaultDeviceAssignmentNominal");
    println!("  num_replicas: {}", NUM_REPLICAS);
    println!("  num_partitions: {}", NUM_PARTITIONS);
    println!("  assignment: {:?}", assignment);
    
    // 验证分配结果
    assert_eq!(assignment.num_replicas(), NUM_REPLICAS);
    assert_eq!(assignment.num_partitions(), NUM_PARTITIONS);
    
    println!("  ✅ Test passed: num_replicas = {}, num_partitions = {}", 
             assignment.num_replicas(), assignment.num_partitions());
    
    Ok(())
}
