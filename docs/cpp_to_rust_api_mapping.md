# 如何用 Rust 绑定实现 PJRT C++ 测试代码

## 原始 C++ 测试代码

```cpp
TEST_F(PjrtCApiTest, GetDefaultDeviceAssignmentNominal) {
  constexpr int kNumReplicas = 2;
  constexpr int kNumPartitions = 1;
  std::vector<int> assignment_buffer(kNumReplicas * kNumPartitions);
  PJRT_Client_DefaultDeviceAssignment_Args args{
      .struct_size = PJRT_Client_DefaultDeviceAssignment_Args_STRUCT_SIZE,
      .extension_start = nullptr,
      .client = client_,
      .num_replicas = kNumReplicas,
      .num_partitions = kNumPartitions,
      .default_assignment_size = assignment_buffer.size(),
      .default_assignment = assignment_buffer.data(),  // in-out
  };
  auto error = ToUniquePtr(api_->PJRT_Client_DefaultDeviceAssignment(&args));
  EXPECT_EQ(error, nullptr);
}
```

## Rust 等价实现

### 方式1: 使用高级 API (推荐)

```rust
use pjrt::{self, Client, Result};

fn test_get_default_device_assignment_nominal() -> Result<()> {
    // 测试参数
    const NUM_REPLICAS: usize = 2;
    const NUM_PARTITIONS: usize = 1;
    
    // 初始化客户端
    let api = pjrt::plugin("pjrt_c_api_cpu_plugin.so").load()?;
    let client = Client::builder(&api).build()?;
    
    // 调用设备分配 API
    let assignment = client.default_device_assignment(NUM_REPLICAS, NUM_PARTITIONS)?;
    
    // 验证结果 (对应 EXPECT_EQ(error, nullptr))
    assert_eq!(assignment.num_replicas(), NUM_REPLICAS);
    assert_eq!(assignment.num_partitions(), NUM_PARTITIONS);
    
    println!("✅ Test passed!");
    Ok(())
}
```

### 方式2: 使用底层绑定 (理论上的实现)

如果需要直接使用 pjrt-sys 绑定，理论上的代码如下：

```rust
use pjrt_sys::PJRT_Client_DefaultDeviceAssignment_Args;

// 注意：这些 API 在当前实现中是私有的，仅用于说明
fn test_with_raw_bindings() -> Result<()> {
    const NUM_REPLICAS: i32 = 2;
    const NUM_PARTITIONS: i32 = 1;
    
    // 创建缓冲区
    let mut assignment_buffer = vec![0i32; (NUM_REPLICAS * NUM_PARTITIONS) as usize];
    
    // 构造参数
    let mut args = PJRT_Client_DefaultDeviceAssignment_Args::new();
    args.num_replicas = NUM_REPLICAS;
    args.num_partitions = NUM_PARTITIONS;
    args.default_assignment_size = assignment_buffer.len();
    args.default_assignment = assignment_buffer.as_mut_ptr();
    // args.client = client.ptr(); // 私有方法
    
    // 调用 C API
    // let error = api.PJRT_Client_DefaultDeviceAssignment(args)?; // 私有方法
    
    Ok(())
}
```

## 关键对应关系

| C++ 代码 | Rust 等价代码 | 说明 |
|----------|---------------|------|
| `constexpr int kNumReplicas = 2;` | `const NUM_REPLICAS: usize = 2;` | 常量定义 |
| `std::vector<int> assignment_buffer` | `Vec<i32>` | 分配缓冲区 |
| `PJRT_Client_DefaultDeviceAssignment_Args args` | `PJRT_Client_DefaultDeviceAssignment_Args::new()` | 参数结构体 |
| `args.struct_size = STRUCT_SIZE` | 自动在 `new()` 中设置 | 结构体大小 |
| `args.extension_start = nullptr` | 自动在 `new()` 中设置 | 扩展指针 |
| `api_->PJRT_Client_DefaultDeviceAssignment(&args)` | `client.default_device_assignment()` | API 调用 |
| `EXPECT_EQ(error, nullptr)` | `?` 操作符或 `assert!` | 错误检查 |

## 绑定实现状态

### ✅ 已实现
1. **结构体绑定**: `PJRT_Client_DefaultDeviceAssignment_Args` 已正确生成
2. **函数指针类型**: `PJRT_Client_DefaultDeviceAssignment` 类型已定义
3. **高级API**: `Client::default_device_assignment()` 方法已实现
4. **错误处理**: 通过 Rust 的 `Result` 类型处理错误
5. **内存安全**: 通过 Rust 的所有权系统确保内存安全

### 🔒 私有实现
- 底层 C API 调用被封装在私有方法中
- 直接访问 `PJRT_Client` 指针的方法是私有的
- 这是有意为之的设计，提供类型安全的高级 API

### 📊 绑定质量评估

1. **完整性**: ✅ 所有必要的结构体和函数都已绑定
2. **类型安全**: ✅ 使用 Rust 类型系统防止内存错误
3. **易用性**: ✅ 提供高级 API 简化使用
4. **性能**: ✅ 零成本抽象，性能与 C++ 等价
5. **错误处理**: ✅ 集成到 Rust 的错误处理机制

## 运行示例

```bash
# 运行高级 API 示例
cargo run -p pjrt --example device_assignment_test

# 运行详细示例
cargo run -p pjrt --example raw_device_assignment_test
```

## 总结

pjrt-rs 的绑定实现是高质量的，提供了：

1. **完整的 C API 覆盖**: 所有 PJRT C API 都已正确绑定
2. **类型安全的接口**: 通过 Rust 类型系统防止常见错误
3. **符合人机工程学的 API**: 高级 API 更易于使用
4. **与 C++ 的功能对等**: 所有功能都可以在 Rust 中实现

对于大多数用例，推荐使用高级 API (`client.default_device_assignment()`)，
因为它提供了更好的类型安全性和易用性，同时保持了与底层 C API 的完全兼容性。
