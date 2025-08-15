use pjrt::{self, Client, DeviceAssignment, CompileOptions, LoadedExecutable, Program, ProgramFormat, Result};

fn main() -> Result<()> {
    println!("=== 设备分配绑定到程序编译示例 ===");
    
    // 初始化 API 和客户端
    let api = pjrt::plugin("pjrt_c_api_cpu_plugin.so").load()?;
    let client = Client::builder(&api).build()?;
    
    println!("平台: {}", client.platform_name());
    println!("可用设备数: {}", client.addressable_devices().len());
    
    // 1. 获取默认设备分配
    const NUM_REPLICAS: usize = 2;
    const NUM_PARTITIONS: usize = 1;
    
    let device_assignment = client.default_device_assignment(NUM_REPLICAS, NUM_PARTITIONS)?;
    println!("\n设备分配: {:?}", device_assignment);
    
    // 2. 创建程序
    const CODE: &[u8] = include_bytes!("prog_f32.mlir");
    let program = Program::new(ProgramFormat::MLIR, CODE);
    
    // 3. 目前的解决方案：使用编译选项设置 num_replicas 和 num_partitions
    // 这是间接的方式，因为目前 DeviceAssignmentProto 还没有在 Rust 绑定中实现
    
    // 检查 CompileOptions 是否有相关方法
    let compile_options = demonstrate_current_api_limitations();
    
    println!("\n编译选项:");
    println!("  使用默认选项 (num_replicas=1, num_partitions=1)");
    
    // 4. 编译程序
    let loaded_executable = LoadedExecutable::builder(&client, &program)
        .options(compile_options)
        .build()?;
    
    println!("\n编译成功!");
    println!("可寻址设备: {:?}", loaded_executable.addressable_devices().len());
    
    // 5. 展示高级方案：如何扩展以支持完整的设备分配
    println!("\n=== 完整设备分配支持的设计方案 ===");
    demonstrate_full_device_assignment_design(&device_assignment)?;
    
    Ok(())
}

/// 演示当前 API 的限制
fn demonstrate_current_api_limitations() -> CompileOptions {
    println!("当前 CompileOptions 的限制:");
    println!("- CompileOptions 使用 protobuf 编码");
    println!("- ExecutableBuildOptions 是内部类型，无法直接构造");
    println!("- DeviceAssignmentProto 类型尚未生成到 Rust 绑定中");
    
    // 目前只能使用默认选项
    CompileOptions::new()
}

/// 演示完整设备分配支持的设计方案
/// 这展示了如何扩展当前的 API 以支持完整的设备分配功能
fn demonstrate_full_device_assignment_design(assignment: &DeviceAssignment) -> Result<()> {
    println!("理想的 API 设计应该是:");
    println!("```rust");
    println!("let compile_options = CompileOptions::new()");
    println!("    .executable_build_options(");
    println!("        ExecutableBuildOptions::new()");
    println!("            .device_assignment(assignment.to_proto()) // 需要实现");
    println!("    );");
    println!("```");
    
    println!("\n为了实现这个功能，需要:");
    println!("1. 在 pjrt-sys/build.rs 中添加更多 proto 文件编译:");
    println!("   - xla/xla_data.proto (包含 DeviceAssignmentProto)");
    println!("   - 相关依赖的 proto 文件");
    
    println!("2. 在 DeviceAssignment 中添加 to_proto() 方法:");
    println!("   impl DeviceAssignment {{");
    println!("       pub fn to_proto(&self) -> DeviceAssignmentProto {{");
    println!("           // 转换逻辑");
    println!("       }}");
    println!("   }}");
    
    println!("3. 在 ExecutableBuildOptions 中添加 device_assignment() 方法");
    
    println!("\n当前的解决方案:");
    println!("- 使用 num_replicas 和 num_partitions 参数");
    println!("- 让 PJRT 运行时自动分配设备");
    println!("- 对于大多数用例，这已经足够了");
    
    // 展示当前可用的信息
    println!("\n当前设备分配信息:");
    println!("  副本数: {}", assignment.num_replicas());
    println!("  分区数: {}", assignment.num_partitions());
    
    let lookup_map = assignment.get_lookup_map();
    for (device_id, logical_id) in lookup_map.iter() {
        println!("  设备 {} -> 副本 {}, 分区 {}", 
                device_id, logical_id.replica_id, logical_id.partition_id);
    }
    
    Ok(())
}
