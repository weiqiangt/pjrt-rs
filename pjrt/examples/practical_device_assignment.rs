use pjrt::{self, Client, LoadedExecutable, Program, ProgramFormat, Result, HostBuffer};

fn main() -> Result<()> {
    println!("=== 当前可用的设备分配方案 ===");
    
    // 1. 初始化
    let api = pjrt::plugin("pjrt_c_api_cpu_plugin.so").load()?;
    let client = Client::builder(&api).build()?;
    
    println!("平台: {}", client.platform_name());
    println!("可用设备: {}", client.addressable_devices().len());
    
    // 2. 准备程序
    const CODE: &[u8] = include_bytes!("prog_f32.mlir");
    let program = Program::new(ProgramFormat::MLIR, CODE);
    
    // 3. 方案一：使用默认编译选项（推荐）
    println!("\n--- 方案一: 默认编译选项 ---");
    let loaded_executable1 = LoadedExecutable::builder(&client, &program).build()?;
    println!("编译成功，可寻址设备数: {}", loaded_executable1.addressable_devices().len());
    
    // 执行测试
    let input = HostBuffer::from_scalar(2.5f32);
    let buffer = input.to_sync(&client).copy()?;
    let result = loaded_executable1.execution(vec![buffer]).run_sync()?;
    println!("执行结果: {:?}", result[0][0].to_host_sync().copy()?);
    
    // 4. 方案二：查询和使用设备分配信息
    println!("\n--- 方案二: 查询设备分配信息 ---");
    let assignment = client.default_device_assignment(1, 1)?;
    println!("默认设备分配: {:?}", assignment);
    println!("设备映射: {:?}", assignment.get_lookup_map());
    
    // 5. 方案三：多副本编译（概念演示）
    println!("\n--- 方案三: 多副本场景设计 ---");
    demonstrate_multi_replica_scenario(&client)?;
    
    Ok(())
}

fn demonstrate_multi_replica_scenario(client: &Client) -> Result<()> {
    println!("多副本场景中的设备分配策略:");
    
    // 查询多副本的默认分配
    let assignment = client.default_device_assignment(2, 1)?;
    println!("2副本1分区的设备分配: {:?}", assignment);
    
    // 当前的解决方案：
    // 1. 使用默认编译选项，PJRT会自动处理设备分配
    // 2. 在执行时通过不同的输入数据实现并行
    
    println!("当前解决方案:");
    println!("1. 编译时: 使用默认选项，让PJRT自动分配");
    println!("2. 执行时: 通过多个execution调用实现并行");
    
    // 演示设备查询
    let devices = client.addressable_devices();
    for (i, device) in devices.iter().enumerate() {
        println!("设备 {}: 本地ID={}", i, device.local_hardware_id());
    }
    
    println!("\n未来扩展计划:");
    println!("- 支持 DeviceAssignmentProto 的 protobuf 绑定");
    println!("- 在 CompileOptions 中添加设备分配设置");
    println!("- 提供更细粒度的设备控制API");
    
    Ok(())
}
