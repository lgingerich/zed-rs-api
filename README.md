# ZED Rust API

[![Crates.io](https://img.shields.io/crates/v/zed-rs-api.svg)](https://crates.io/crates/zed-rs-api)
[![Documentation](https://docs.rs/zed-rs-api/badge.svg)](https://docs.rs/zed-rs-api)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

Rust bindings for the [Stereolabs ZED SDK](https://www.stereolabs.com/developers/), providing safe and idiomatic Rust interfaces for ZED camera operations.

## Features

- 🦀 **Safe Rust API** - Memory-safe wrappers around the ZED C API
- 📷 **Camera Control** - Initialize, configure, and control ZED cameras
- 🎯 **Depth Sensing** - Access depth data and 3D information
- ⚡ **High Performance** - Zero-cost abstractions over the native SDK
- 🔧 **Builder Pattern** - Fluent API for configuration
- 📖 **Well Documented** - Comprehensive documentation and examples

## Prerequisites

Before using this crate, you need to install the ZED SDK:

1. **Download and install the ZED SDK** from [Stereolabs website](https://www.stereolabs.com/developers/release/)
2. **Ensure the SDK is installed** in the standard location:
   - Linux: `/usr/local/zed/`
   - The library should be at `/usr/local/zed/lib/libsl_zed_c.so`
   - Headers should be at `/usr/local/zed/include/`

3. **Set up library path** (Linux):
   ```bash
   export LD_LIBRARY_PATH=/usr/local/zed/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
zed-rs-api = "0.1.0"
```

## Quick Start

```rust
use zed_sdk::{Camera, InitParameters, Resolution, DepthMode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create and configure camera
    let mut camera = Camera::new(0)?;
    
    let params = InitParameters::default()
        .with_resolution(Resolution::HD1080)
        .with_fps(30)
        .with_depth_mode(DepthMode::Neural);
    
    // Open camera
    camera.open(&params)?;
    
    println!("Camera serial: {}", camera.get_serial_number()?);
    
    // Capture frames
    for i in 0..10 {
        camera.grab()?;
        let timestamp = camera.get_timestamp()?;
        println!("Frame {}: {}", i, timestamp);
    }
    
    Ok(())
}
```

## Running Examples

To run the included example:

```bash
# Set library path and run
LD_LIBRARY_PATH=/usr/local/zed/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH cargo run --bin zed_example
```

## API Overview

### Camera Management

```rust
use zed_sdk::{Camera, InitParameters, Resolution};

// Create camera
let mut camera = Camera::new(0)?;

// Configure parameters
let params = InitParameters::default()
    .with_resolution(Resolution::HD1080)
    .with_fps(30);

// Open and use camera
camera.open(&params)?;
let serial = camera.get_serial_number()?;
```

### Frame Capture

```rust
use zed_sdk::RuntimeParameters;

// Basic frame capture
camera.grab()?;

// With custom parameters
let runtime_params = RuntimeParameters::default();
camera.grab_with_params(&runtime_params)?;

// Get frame timestamp
let timestamp = camera.get_timestamp()?;
```

### Configuration Options

```rust
use zed_sdk::{InitParameters, Resolution, DepthMode};

let params = InitParameters::default()
    .with_resolution(Resolution::HD1080)    // Camera resolution
    .with_fps(30)                           // Frame rate
    .with_depth_mode(DepthMode::Neural)     // Depth computation mode
    .with_depth_maximum_distance(40.0)     // Max depth distance
    .with_image_enhancement(true)           // Enable image enhancement
    .with_verbose(false);                   // SDK logging
```

## Error Handling

The crate provides comprehensive error handling through the `ZedError` enum:

```rust
use zed_sdk::ZedError;

match camera.open(&params) {
    Ok(()) => println!("Camera opened successfully"),
    Err(ZedError::CameraOpenFailed(code)) => {
        eprintln!("Failed to open camera: error code {}", code);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## System Requirements

- **ZED Camera** (ZED, ZED Mini, ZED 2, ZED 2i, ZED X, etc.)
- **CUDA** (for depth processing)
- **ZED SDK** 4.0 or later
- **Rust** 1.70 or later

### Supported Platforms

- ✅ Linux (x86_64, aarch64)
- ⚠️ Windows (planned)

## Building from Source

```bash
git clone https://github.com/yourusername/zed-rs-api.git
cd zed-rs-api

# Build the library
cargo build

# Run tests (requires ZED camera)
LD_LIBRARY_PATH=/usr/local/zed/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH cargo test

# Build documentation
cargo doc --open
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Install the ZED SDK
2. Clone this repository
3. Set up the library path
4. Run `cargo test` to ensure everything works

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Acknowledgments

- [Stereolabs](https://www.stereolabs.com/) for the ZED SDK
- The Rust community for excellent tooling and libraries

## Links

- [ZED SDK Documentation](https://www.stereolabs.com/docs/)
- [Stereolabs Developer Portal](https://www.stereolabs.com/developers/)
- [Crate Documentation](https://docs.rs/zed-sdk-rs)