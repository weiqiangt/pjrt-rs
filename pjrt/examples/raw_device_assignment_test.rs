/// 使用高级 API 实现与 C++ 测试等价的代码
/// 
/// 这个示例展示了如何使用 pjrt 的安全 Rust API 来实现
/// 与 C++ 测试中的 PJRT_Client_DefaultDeviceAssignment 调用等价的功能
use pjrt::{self, Client};

fn main() -> pjrt::Result<()> {
    println!("=== PJRT Default Device Assignment Test (Rust API) ===");
    
    // 初始化 API 和客户端
    let api = pjrt::plugin("pjrt_c_api_cpu_plugin.so").load()?;
    let client = Client::builder(&api).build()?;
    
    // 测试参数 - 对应 C++ 测试中的常量
    const NUM_REPLICAS: usize = 2;
    const NUM_PARTITIONS: usize = 1;
    
    println!("Test parameters:");
    println!("  kNumReplicas = {}", NUM_REPLICAS);
    println!("  kNumPartitions = {}", NUM_PARTITIONS);
    
    println!("\nCalling client.default_device_assignment()...");
    println!("(这在底层调用了 PJRT_Client_DefaultDeviceAssignment C API)");
    
    // 调用高级 API - 内部会调用底层的 PJRT_Client_DefaultDeviceAssignment
    let assignment = client.default_device_assignment(NUM_REPLICAS, NUM_PARTITIONS)?;
    
    // 对应 C++ 测试中的 EXPECT_EQ(error, nullptr)
    println!("✅ default_device_assignment returned success (no error)");
    
    // 打印结果
    println!("\nResults:");
    println!("  assignment = {:?}", assignment);
    println!("  num_replicas = {}", assignment.num_replicas());
    println!("  num_partitions = {}", assignment.num_partitions());
    
    // 验证结果 - 对应 C++ 测试中的断言
    assert_eq!(assignment.num_replicas(), NUM_REPLICAS);
    assert_eq!(assignment.num_partitions(), NUM_PARTITIONS);
    
    println!("✅ All assertions passed!");
    
    // 演示额外的功能
    println!("\n=== 额外功能演示 ===");
    let lookup_map = assignment.get_lookup_map();
    println!("Device assignment lookup map: {:?}", lookup_map);
    
    Ok(())
}
