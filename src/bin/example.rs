use zed_sdk::{
    Camera, InitParameters, Resolution, ViewType, MemoryType, 
    PositionalTrackingParameters, ReferenceFrame,
    ObjectDetectionParameters, ObjectDetectionRuntimeParameters,
    ObjectDetectionModel, ObjectClass, SpatialMappingParameters,
    SpatialMappingState, BodyTrackingParameters, BodyTrackingRuntimeParameters,
    VideoSettings, TimeReference, SvoCompressionMode, MeshFileFormat, Side,
    ZedError,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("ZED SDK Rust Wrapper - Comprehensive Example");
    println!("=============================================");

    // Create camera instance
    let mut camera = Camera::new(0)?;
    println!("✓ Camera created successfully");
    
    // Configure initialization parameters
    let init_params = InitParameters::default()
        .with_resolution(Resolution::HD720)  // Use HD720 for better performance
        .with_fps(30)
        .with_depth_mode(zed_sdk::DepthMode::Neural)
        .with_verbose(true);
    
    // Open the camera
    camera.open(&init_params)?;
    println!("✓ Camera opened successfully");
    
    // Get camera information
    let serial = camera.get_serial_number()?;
    let (width, height) = camera.get_resolution()?;
    let fps = camera.get_camera_fps()?;
    let firmware = camera.get_camera_firmware()?;
    
    println!("📷 Camera Info:");
    println!("   Serial: {}", serial);
    println!("   Resolution: {}x{}", width, height);
    println!("   FPS: {:.1}", fps);
    println!("   Firmware: {}", firmware);
    
    // Set camera settings
    if camera.is_camera_setting_supported(VideoSettings::Exposure)? {
        camera.set_camera_setting(VideoSettings::Exposure, 50)?;
        println!("✓ Exposure set to 50");
    }
    
    if camera.is_camera_setting_supported(VideoSettings::Gain)? {
        camera.set_camera_setting(VideoSettings::Gain, 50)?;
        println!("✓ Gain set to 50");
    }
    
    // Enable positional tracking
    let tracking_params = PositionalTrackingParameters::default()
        .with_area_memory(true)
        .with_imu_fusion(true)
        .with_pose_smoothing(false);
    
    camera.enable_positional_tracking(&tracking_params, None)?;
    println!("✓ Positional tracking enabled");
    
    // Enable object detection
    let detection_params = ObjectDetectionParameters::default()
        .with_detection_model(ObjectDetectionModel::MultiClassBoxMedium)
        .with_tracking(true)
        .with_max_range(15.0);
    
    camera.enable_object_detection(&detection_params)?;
    println!("✓ Object detection enabled");
    
    // Enable spatial mapping
    let mapping_params = SpatialMappingParameters::default()
        .with_resolution_meter(0.08)  // Medium resolution for better performance
        .with_max_memory_usage(1024)  // 1GB memory limit
        .with_max_range_meter(5.0);   // 5 meter range
    
    camera.enable_spatial_mapping(&mapping_params)?;
    println!("✓ Spatial mapping enabled");
    
    // Enable body tracking
    let body_params = BodyTrackingParameters::default()
        .with_tracking(true)
        .with_body_fitting(false)  // Disable for better performance
        .with_max_range(10.0);
    
    match camera.enable_body_tracking(&body_params) {
        Ok(_) => println!("✓ Body tracking enabled"),
        Err(e) => println!("⚠ Body tracking not available: {}", e),
    }
    
    // Enable SVO recording
    let recording_filename = "zed_recording.svo";
    match camera.enable_recording(recording_filename, SvoCompressionMode::H264, 0, 30, false) {
        Ok(_) => println!("✓ SVO recording enabled: {}", recording_filename),
        Err(e) => println!("⚠ Recording not available: {}", e),
    }
    
    println!("\n🎬 Starting capture loop...");
    println!("   Will capture 150 frames (5 seconds at 30 FPS)");
    
    // Main capture loop
    for i in 0..150 {
        // Grab frame
        match camera.grab() {
            Ok(_) => {},
            Err(ZedError::GrabFailed(code)) if code == 1 => {
                // End of SVO file reached, break the loop
                println!("End of SVO file reached");
                break;
            },
            Err(e) => return Err(e.into()),
        }
        
        if i % 30 == 0 {  // Print info every second
            println!("\n📋 Frame {}/150:", i);
            
            // Get image data
            let left_image = camera.retrieve_image(ViewType::Left, MemoryType::Cpu)?;
            println!("   Image: {}x{} ({} channels)", 
                left_image.width, left_image.height, left_image.channels);
            
            // Get depth data
            let depth_data = camera.retrieve_depth(MemoryType::Cpu)?;
            let (min_depth, max_depth) = camera.get_current_min_max_depth()?;
            println!("   Depth: {}x{} (range: {:.2}m - {:.2}m)", 
                depth_data.width, depth_data.height, min_depth, max_depth);
            
            // Get sensor data
            if let Ok(sensors) = camera.get_sensors_data(TimeReference::Image) {
                if sensors.imu.is_available {
                    let accel = &sensors.imu.linear_acceleration;
                    let gyro = &sensors.imu.angular_velocity;
                    println!("   IMU: accel=({:.2}, {:.2}, {:.2}) gyro=({:.2}, {:.2}, {:.2})",
                        accel.x, accel.y, accel.z, gyro.x, gyro.y, gyro.z);
                }
                
                if sensors.barometer.is_available {
                    println!("   Barometer: {:.1} hPa, altitude: {:.1}m",
                        sensors.barometer.pressure, sensors.barometer.relative_altitude);
                }
                
                if sensors.magnetometer.is_available {
                    println!("   Magnetometer: heading {:.1}° (accuracy: {:.2})",
                        sensors.magnetometer.magnetic_heading, 
                        sensors.magnetometer.magnetic_heading_accuracy);
                }
            }
            
            // Get camera pose
            if let Ok(pose_data) = camera.get_pose_data(ReferenceFrame::World) {
                if pose_data.is_valid() {
                    let pos = &pose_data.translation;
                    let (roll, pitch, yaw) = pose_data.get_euler_angles();
                    println!("   Pose: pos=({:.2}, {:.2}, {:.2}) orient=({:.1}°, {:.1}°, {:.1}°) conf={}%",
                        pos.x, pos.y, pos.z, 
                        roll.to_degrees(), pitch.to_degrees(), yaw.to_degrees(),
                        pose_data.confidence_percentage());
                }
            }
            
            // Get detected objects
            let runtime_params = ObjectDetectionRuntimeParameters::default()
                .with_detection_confidence_threshold(50.0);
            if let Ok(objects) = camera.retrieve_objects(&runtime_params, 0) {
                let tracked_objects = objects.get_tracked_objects();
                if !tracked_objects.is_empty() {
                    println!("   Objects: {} detected, {} tracked", objects.len(), tracked_objects.len());
                    for obj in tracked_objects.iter().take(3) {  // Show first 3
                        let pos = &obj.position;
                        println!("     {}: {} at ({:.1}, {:.1}, {:.1}) conf={:.0}%",
                            obj.class_name(), obj.id, pos.x, pos.y, pos.z, obj.confidence);
                    }
                }
            }
            
            // Get detected bodies
            let body_runtime_params = BodyTrackingRuntimeParameters::default()
                .with_detection_confidence_threshold(40.0);
            if let Ok(bodies) = camera.retrieve_bodies(&body_runtime_params, 0) {
                let tracked_bodies = bodies.get_tracked_bodies();
                if !tracked_bodies.is_empty() {
                    println!("   Bodies: {} detected, {} tracked", bodies.len(), tracked_bodies.len());
                    for body in tracked_bodies.iter().take(2) {  // Show first 2
                        let pos = &body.position;
                        println!("     Body {}: at ({:.1}, {:.1}, {:.1}) keypoints={} conf={:.0}%",
                            body.id, pos.x, pos.y, pos.z, body.keypoint_3d.len(), body.confidence);
                    }
                }
            }
            
            // Check spatial mapping state
            if let Ok(state) = camera.get_spatial_mapping_state() {
                match state {
                    SpatialMappingState::Ok => {
                        println!("   Spatial Mapping: ✓ Active");
                        // Request mesh periodically
                        if i % 60 == 0 {
                            camera.request_mesh_async()?;
                            println!("     Mesh generation requested");
                        }
                    },
                    SpatialMappingState::Initializing => println!("   Spatial Mapping: ⏳ Initializing"),
                    SpatialMappingState::FpsLow => println!("   Spatial Mapping: ⚠ FPS too low"),
                    SpatialMappingState::NotEnabled => println!("   Spatial Mapping: ❌ Not enabled"),
                }
            }
            
            // Check recording status
            if let Ok(status) = camera.get_recording_status() {
                if status.is_recording {
                    println!("   Recording: ✓ Active (ratio: {:.1}x, time: {:.1}ms)",
                        status.current_compression_ratio, status.current_compression_time);
                }
            }
        }
        
        // Save snapshots at specific intervals
        if i == 30 {
            match camera.save_current_image(ViewType::Left, "snapshot_left.png") {
                Ok(_) => println!("   📸 Saved snapshot_left.png"),
                Err(e) => println!("   ⚠ Failed to save image: {}", e),
            }
            
            match camera.save_current_depth(Side::Left, "snapshot_depth.png") {
                Ok(_) => println!("   📸 Saved snapshot_depth.png"),
                Err(e) => println!("   ⚠ Failed to save depth: {}", e),
            }
        }
    }
    
    println!("\n🔄 Cleaning up...");
    
    // Check if we can retrieve and save the mesh
    if let Ok(state) = camera.get_spatial_mapping_state() {
        if state == SpatialMappingState::Ok {
            println!("   Generating final mesh...");
            camera.request_mesh_async()?;
            
            // Wait a moment for mesh generation
            std::thread::sleep(std::time::Duration::from_millis(500));
            
            match camera.retrieve_mesh(100) {
                Ok(mesh) => {
                    println!("   ✓ Mesh retrieved: {} vertices, {} triangles", 
                        mesh.num_vertices, mesh.num_triangles);
                    
                    // Save mesh to file
                    match camera.save_mesh("final_mesh.ply", MeshFileFormat::Ply) {
                        Ok(true) => println!("   ✓ Mesh saved to final_mesh.ply"),
                        Ok(false) => println!("   ⚠ Failed to save mesh"),
                        Err(e) => println!("   ⚠ Error saving mesh: {}", e),
                    }
                },
                Err(e) => println!("   ⚠ Failed to retrieve mesh: {}", e),
            }
        }
    }
    
    // Disable all modules
    camera.disable_recording();
    println!("   ✓ Recording disabled");
    
    camera.disable_spatial_mapping();
    println!("   ✓ Spatial mapping disabled");
    
    let _ = camera.disable_body_tracking(0, false);
    println!("   ✓ Body tracking disabled");
    
    camera.disable_object_detection(0, false)?;
    println!("   ✓ Object detection disabled");
    
    camera.disable_positional_tracking(None)?;
    println!("   ✓ Positional tracking disabled");
    
    println!("\n✅ Example completed successfully!");
    println!("📁 Generated files:");
    println!("   - snapshot_left.png (left camera image)");
    println!("   - snapshot_depth.png (depth map)");
    println!("   - final_mesh.ply (3D mesh, if available)");
    println!("   - zed_recording.svo (video recording, if enabled)");
    
    Ok(())
}