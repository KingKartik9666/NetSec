use std::{env,path::PathBuf};
fn main() {
    let args:Vec<String>=env::args().collect();
    if args.iter().any(|v|v=="--predict-worker") {
        if let Err(error)=netsec_runtime::analysis::worker(&args) {eprintln!("{error}");std::process::exit(1)}
        return;
    }
    if let Err(error) = netsec_runtime::process_tree::own_descendants() {
        eprintln!("NetSec startup failed: {error}"); std::process::exit(1);
    }
    let root=args.windows(2).find(|v|v[0]=="--root").map(|v|PathBuf::from(&v[1])).unwrap_or_else(||env::current_dir().unwrap_or_default());
    let port=args.windows(2).find(|v|v[0]=="--port").and_then(|v|v[1].parse().ok()).unwrap_or(0);
    let runtime=tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().expect("Create native app executor");
    if let Err(error)=runtime.block_on(netsec_runtime::transport::serve(root,port,args.iter().any(|v|v=="--parent-stdin"),args.iter().any(|v|v=="--browser"))) {
        eprintln!("NetSec startup failed: {error}");std::process::exit(1)
    }
}
