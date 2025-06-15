use zed_sdk::{
    Camera, InitParameters, Resolution, DepthMode, ViewType, MemoryType,
    PositionalTrackingParameters, ReferenceFrame,
    ObjectDetectionParameters, ObjectDetectionRuntimeParameters,
    ObjectDetectionModel, ObjectClass
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new camera instance
    let mut camera = Camera::new(0)?;
    
    // Configure initialization parameters
    let init_params = InitParameters::default()
        .with_resolution(Resolution::HD1080)
        .with_fps(30)
        .with_depth_mode(DepthMode::Neural)
        .with_depth_maximum_distance(40.0)
        .with_image_enhancement(true);
    
    // Open the camera
    camera.open(&init_params)?;
    
    // Get camera information
    let serial_number = camera.get_serial_number()?;
    println!("Hello! This is my serial number: {}", serial_number);
    
    let (width, height) = camera.get_resolution()?;
    println!("Camera resolution: {}x{}", width, height);
    
    // Get depth range information
    let min_range = camera.get_depth_min_range_value()?;
    let max_range = camera.get_depth_max_range_value()?;
    println!("Depth range: {:.2}m to {:.2}m", min_range, max_range);
    
    // Enable positional tracking
    println!("Enabling positional tracking...");
    let tracking_params = PositionalTrackingParameters::default()
        .with_area_memory(true)
        .with_imu_fusion(true)
        .with_gravity_as_origin(true);
    
    camera.enable_positional_tracking(&tracking_params, None)?;
    println!("Positional tracking enabled!");
    
    // Enable object detection
    println!("Enabling object detection...");
    let detection_params = ObjectDetectionParameters::default()
        .with_detection_model(ObjectDetectionModel::MultiClassBoxMedium)
        .with_tracking(true)
        .with_max_range(20.0);
    
    camera.enable_object_detection(&detection_params)?;
    println!("Object detection enabled!");
    
    // Capture and process frames
    for i in 0..20 {
        camera.grab()?;
        
        let timestamp = camera.get_timestamp()?;
        let image_timestamp = camera.get_image_timestamp()?;
        println!("Frame {}: timestamp = {}, image_timestamp = {}", i, timestamp, image_timestamp);
        
        // Get positional tracking data
        if i >= 5 { // Give tracking some time to initialize
            match camera.get_pose_data(ReferenceFrame::World) {
                Ok(pose_data) => {
                    if pose_data.is_valid() {
                        println!("  Pose valid! Confidence: {}%", pose_data.confidence_percentage());
                        println!("  Position: ({:.3}, {:.3}, {:.3})", 
                            pose_data.translation.x, pose_data.translation.y, pose_data.translation.z);
                    } else {
                        println!("  Tracking not yet valid...");
                    }
                },
                Err(e) => println!("  Failed to get pose data: {}", e),
            }
        }
        
        // Get object detection results
        if i >= 3 { // Give object detection some time to initialize
            let runtime_params = ObjectDetectionRuntimeParameters::default()
                .with_detection_confidence_threshold(50.0);
            
            match camera.retrieve_objects(&runtime_params, 0) {
                Ok(objects) => {
                    if !objects.is_empty() {
                        println!("  Detected {} objects:", objects.len());
                        
                        // Show all detected objects
                        for obj in &objects.objects {
                            if obj.is_tracked() {
                                let (width_2d, height_2d) = obj.get_2d_size();
                                let center_2d = obj.get_2d_center();
                                
                                println!("    {} #{}: {} at 3D({:.1}, {:.1}, {:.1}m) 2D({:.0}, {:.0}px) size({:.0}x{:.0}px) conf:{:.1}%",
                                    obj.class_name(),
                                    obj.id,
                                    match obj.tracking_state {
                                        zed_sdk::ObjectTrackingState::Ok => "TRACKED",
                                        zed_sdk::ObjectTrackingState::Searching => "SEARCHING",
                                        zed_sdk::ObjectTrackingState::Off => "OFF",
                                        zed_sdk::ObjectTrackingState::Terminate => "TERMINATE",
                                    },
                                    obj.position.x, obj.position.y, obj.position.z,
                                    center_2d.x, center_2d.y,
                                    width_2d, height_2d,
                                    obj.confidence);
                            }
                        }
                        
                        // Show statistics by object class
                        let people = objects.get_objects_by_class(ObjectClass::Person);
                        let vehicles = objects.get_objects_by_class(ObjectClass::Vehicle);
                        let tracked_objects = objects.get_tracked_objects();
                        
                        if !people.is_empty() {
                            println!("    -> {} people detected", people.len());
                        }
                        if !vehicles.is_empty() {
                            println!("    -> {} vehicles detected", vehicles.len());
                        }
                        println!("    -> {} objects actively tracked", tracked_objects.len());
                    } else {
                        println!("  No objects detected");
                    }
                },
                Err(e) => println!("  Failed to retrieve objects: {}", e),
            }
        }
        
        // Retrieve different types of data based on frame number
        match i {
            0 => {
                // Retrieve left camera image
                println!("  Retrieving left camera image...");
                let left_image = camera.retrieve_image(ViewType::Left, MemoryType::Cpu)?;
                println!("  Left image: {}x{} with {} channels, {} bytes", 
                    left_image.width, left_image.height, left_image.channels, left_image.size());
            },
            2 => {
                // Retrieve depth data
                println!("  Retrieving depth data...");
                let depth_data = camera.retrieve_depth(MemoryType::Cpu)?;
                println!("  Depth data: {}x{}, {} values", 
                    depth_data.width, depth_data.height, depth_data.data.len());
                
                // Get current min/max depth values
                let (min_depth, max_depth) = camera.get_current_min_max_depth()?;
                println!("  Current depth range: {:.2}m to {:.2}m", min_depth, max_depth);
                
                // Sample some depth values
                if let Some(center_depth) = depth_data.get_depth(depth_data.width / 2, depth_data.height / 2) {
                    println!("  Center pixel depth: {:.2}m", center_depth);
                }
            },
            4 => {
                // Retrieve point cloud
                println!("  Retrieving point cloud...");
                let point_cloud = camera.retrieve_point_cloud(MemoryType::Cpu)?;
                println!("  Point cloud: {}x{}, {} values", 
                    point_cloud.width, point_cloud.height, point_cloud.data.len());
                
                // Sample 3D point
                if let Some((x, y, z)) = point_cloud.get_point(point_cloud.width / 2, point_cloud.height / 2) {
                    println!("  Center 3D point: ({:.2}, {:.2}, {:.2})", x, y, z);
                }
            },
            15 => {
                // Get transformation matrix
                println!("  Getting transformation matrix...");
                match camera.get_position_array(ReferenceFrame::World) {
                    Ok(transform_matrix) => {
                        println!("  4x4 Transformation matrix:");
                        for row in 0..4 {
                            println!("    [{:.3}, {:.3}, {:.3}, {:.3}]",
                                transform_matrix[row * 4],
                                transform_matrix[row * 4 + 1],
                                transform_matrix[row * 4 + 2],
                                transform_matrix[row * 4 + 3]);
                        }
                    },
                    Err(e) => println!("  Failed to get transformation matrix: {}", e),
                }
            },
            _ => {}
        }
    }
    
    // Disable object detection and positional tracking before closing
    println!("Disabling object detection...");
    camera.disable_object_detection(0, false)?;
    
    println!("Disabling positional tracking...");
    camera.disable_positional_tracking(None)?;
    
    println!("Camera will be automatically closed when dropped");
    Ok(())
}