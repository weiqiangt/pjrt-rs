use pjrt::{plugin, Client};

fn main() -> pjrt::Result<()> {
    // 初始化插件，类似于 C++ 测试中的 setup
    let api = plugin("cpu")?;
    
    // 创建客户端
    let client = Client::cpu()?;
    
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
    
    // 验证分配结果的大小
    let expected_size = NUM_REPLICAS * NUM_PARTITIONS;
    assert_eq!(assignment.assignment().len(), expected_size);
    
    println!("  ✅ Test passed: assignment size = {}", expected_size);
    
    Ok(())
}
