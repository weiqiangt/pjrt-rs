use pjrt::{self, Client, HostBuffer, LoadedExecutable, Result};
use pjrt::{KeyValueStore, ProgramFormat::MLIR};
use std::thread;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

const CODE: &[u8] = include_bytes!("parallel.mlir");

#[derive(Clone)]
struct KV {
    store: Arc<Mutex<HashMap<String, String>>>,
}

impl pjrt::KeyValueStore for KV {
    fn put(&self, key: &str, value: &str) -> pjrt::Result<()> {
        println!("put {} = {}", key, value);
        let mut store = self.store.lock().unwrap();
        store.insert(key.to_string(), value.to_string());
        Ok(())
    }

    fn get(&self, key: &str, _timeout_in_ms: i32) -> pjrt::Result<String> {
        println!("get {}", key);
        let store = self.store.lock().unwrap();
        match store.get(key) {
            Some(value) => Ok(value.clone()),
            None => Err(pjrt::Error::NullPointer),
        }
    }
}

fn execute(replica_id: u64, kv: KV) -> Result<()> {
    let api = pjrt::plugin("pjrt_c_api_cpu_plugin.so").load()?;
    println!("api_version = {:?}", api.version());

    let kv: Box<dyn KeyValueStore> = Box::new(kv);
    let client = Client::builder(&api).maybe_kv_store(Some(&kv)).build()?;
    println!("process_index = {}", client.process_index());

    println!("platform_name = {}", client.platform_name());

    let program = pjrt::Program::new(MLIR, CODE);

    let loaded_executable = LoadedExecutable::builder(&client, &program).build()?;

    let a = HostBuffer::from_data(vec![1.25f32, 1.25f32, 1.25f32, 1.25f32]).build();
    println!("input = {:?}", a);

    let inputs = a.to_sync(&client).copy()?;

    let result = loaded_executable.execution(inputs).run_sync()?;

    let ouput = &result[0][0];
    let output = ouput.to_host_sync().copy()?;
    println!("output= {:?}", output);

    Ok(())
}

fn main() {
    let kv = KV {
        store: Arc::new(Mutex::new(HashMap::new())),
    };

    let handle_0 = thread::spawn({
        let kv = kv.clone();
        move || execute(0, kv)
    });

    let handle_1 = thread::spawn({
        let kv = kv.clone();
        move || execute(1, kv)
    });

    let _ = handle_0.join();
    let _ = handle_1.join();
}
