//! # ZED SDK Rust Bindings
//!
//! This crate provides Rust bindings for the Stereolabs ZED SDK, allowing you to
//! interface with ZED cameras from Rust applications.
//!
//! ## Features
//!
//! - Safe Rust wrappers around the ZED C API
//! - Camera initialization and configuration
//! - Image and depth data capture
//! - Point cloud generation
//! - Confidence map retrieval
//! - Multiple image formats (BGRA, grayscale, unrectified)
//! - Depth data in multiple formats (float, 16-bit millimeters)
//! - Normal vector computation
//! - Disparity map access
//! - **Positional tracking and SLAM**
//! - 6DOF pose estimation (position and orientation)
//! - IMU sensor fusion
//! - Area memory for relocalization
//! - Multiple reference frames (world, camera)
//! - **Object Detection and Tracking**
//! - Multi-class object detection (Person, Vehicle, Bag, Animal, etc.)
//! - Real-time object tracking with unique IDs
//! - 2D and 3D bounding boxes
//! - Object confidence and position estimation
//! - Configurable detection models and filtering
//! - **Spatial Mapping and 3D Reconstruction**
//! - Real-time mesh generation and reconstruction
//! - Configurable resolution and memory usage
//! - Mesh filtering and texture mapping
//! - Point cloud and mesh export (PLY, OBJ formats)
//! - **Sensor Data Access**
//! - IMU data (accelerometer, gyroscope)
//! - Magnetometer readings with heading information
//! - Barometer data with altitude measurements
//! - Temperature sensors monitoring
//! - Synchronized sensor data with camera frames
//! - **Camera Control and Settings**
//! - Brightness, contrast, saturation, sharpness
//! - Exposure and gain control (manual/automatic)
//! - White balance adjustment
//! - LED status control
//! - Region of Interest (ROI) for auto exposure/gain
//! - **SVO Recording and Playback**
//! - Record camera streams to SVO files
//! - Multiple compression modes (Lossless, H264, H265)
//! - Playback control with frame seeking
//! - Recording status monitoring
//! - **Body Tracking and Human Pose Estimation**
//! - Real-time human skeleton detection
//! - Multiple body formats (18, 34, 38, 70 keypoints)
//! - 3D pose estimation with confidence values
//! - Body tracking with unique IDs
//! - Joint position and orientation data
//! - Error handling with Rust Result types
//!
//! ## Example
//!
//! ```no_run
//! use zed_sdk::{Camera, InitParameters, Resolution, ViewType, MemoryType, 
//!               PositionalTrackingParameters, ReferenceFrame,
//!               ObjectDetectionParameters, ObjectDetectionRuntimeParameters,
//!               ObjectDetectionModel, ObjectClass, SpatialMappingParameters,
//!               BodyTrackingParameters, VideoSettings, TimeReference,
//!               SvoCompressionMode, MeshFileFormat, Side, StreamingParameters,
//!               StreamingCodec, PlaneDetectionParameters, Vector2, Vector3,
//!               Quaternion, CoordinateSystem};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Check available devices
//!     let devices = Camera::get_device_list();
//!     println!("Found {} ZED cameras", devices.len());
//!     for device in &devices {
//!         println!("  Camera {}: SN {} (model {})", device.id, device.serial_number, device.camera_model);
//!     }
//!     
//!     let mut camera = Camera::new(0)?;
//!     
//!     let init_params = InitParameters::default()
//!         .with_resolution(Resolution::HD1080)
//!         .with_fps(30);
//!     
//!     camera.open(&init_params)?;
//!     
//!     println!("SDK Version: {}", Camera::get_sdk_version());
//!     
//!     // Enable positional tracking
//!     let tracking_params = PositionalTrackingParameters::default()
//!         .with_area_memory(true)
//!         .with_imu_fusion(true);
//!     camera.enable_positional_tracking(&tracking_params, None)?;
//!     
//!     // Enable object detection
//!     let detection_params = ObjectDetectionParameters::default()
//!         .with_detection_model(ObjectDetectionModel::MultiClassBoxMedium)
//!         .with_tracking(true);
//!     camera.enable_object_detection(&detection_params)?;
//!     
//!     // Enable spatial mapping
//!     let mapping_params = SpatialMappingParameters::default()
//!         .with_resolution_meter(0.05)
//!         .with_max_memory_usage(2048);
//!     camera.enable_spatial_mapping(&mapping_params)?;
//!     
//!     // Enable body tracking
//!     let body_params = BodyTrackingParameters::default()
//!         .with_tracking(true)
//!         .with_body_fitting(true);
//!     camera.enable_body_tracking(&body_params)?;
//!     
//!     // Enable SVO recording
//!     camera.enable_recording("recording.svo", SvoCompressionMode::H264, 0, 30, false)?;
//!     
//!     // Enable streaming
//!     let streaming_params = StreamingParameters::default()
//!         .with_codec(StreamingCodec::H264)
//!         .with_port(30000)
//!         .with_bitrate(8000);
//!     camera.enable_streaming(&streaming_params)?;
//!     
//!     // Set camera settings
//!     camera.set_camera_setting(VideoSettings::Exposure, 50)?;
//!     camera.set_camera_setting(VideoSettings::Gain, 50)?;
//!     
//!     println!("Camera serial: {}", camera.get_serial_number()?);
//!     println!("Camera FPS: {}", camera.get_camera_fps()?);
//!     println!("Input type: {}", camera.get_input_type()?);
//!     
//!     // Capture frames and retrieve data
//!     for i in 0..100 {
//!         camera.grab()?;
//!         
//!         // Get image data
//!         let left_image = camera.retrieve_image(ViewType::Left, MemoryType::Cpu)?;
//!         println!("Frame {}: {}x{} image", i, left_image.width, left_image.height);
//!         
//!         // Get sensor data
//!         if let Ok(sensors) = camera.get_sensors_data(TimeReference::Image) {
//!             if sensors.imu.is_available {
//!                 println!("  IMU: accel=({:.2}, {:.2}, {:.2}), gyro=({:.2}, {:.2}, {:.2})",
//!                     sensors.imu.linear_acceleration.x, sensors.imu.linear_acceleration.y, sensors.imu.linear_acceleration.z,
//!                     sensors.imu.angular_velocity.x, sensors.imu.angular_velocity.y, sensors.imu.angular_velocity.z);
//!             }
//!         }
//!         
//!         // Get detected objects
//!         let runtime_params = ObjectDetectionRuntimeParameters::default()
//!             .with_detection_confidence_threshold(50.0);
//!         if let Ok(objects) = camera.retrieve_objects(&runtime_params, 0) {
//!             println!("  Detected {} objects", objects.len());
//!             for obj in &objects.objects {
//!                 if obj.is_tracked() {
//!                     println!("    {}: {} at ({:.1}, {:.1}, {:.1}) - confidence: {:.1}%",
//!                         obj.class_name(), obj.id, 
//!                         obj.position.x, obj.position.y, obj.position.z,
//!                         obj.confidence);
//!                 }
//!             }
//!         }
//!         
//!         // Get detected bodies
//!         let body_runtime_params = BodyTrackingRuntimeParameters::default()
//!             .with_detection_confidence_threshold(40.0);
//!         if let Ok(bodies) = camera.retrieve_bodies(&body_runtime_params, 0) {
//!             println!("  Detected {} bodies", bodies.len());
//!             for body in &bodies.bodies {
//!                 if body.is_tracked() {
//!                     println!("    Body {}: position ({:.1}, {:.1}, {:.1}) - {} keypoints",
//!                         body.id, body.position.x, body.position.y, body.position.z,
//!                         body.keypoint_3d.len());
//!                 }
//!             }
//!         }
//!         
//!         // Get camera pose
//!         if let Ok(pose_data) = camera.get_pose_data(ReferenceFrame::World) {
//!             if pose_data.is_valid() {
//!                 println!("  Camera position: ({:.2}, {:.2}, {:.2})", 
//!                     pose_data.translation.x, pose_data.translation.y, pose_data.translation.z);
//!             }
//!         }
//!         
//!         // Check spatial mapping state
//!         if let Ok(state) = camera.get_spatial_mapping_state() {
//!             if state == SpatialMappingState::Ok && i % 30 == 0 {
//!                 // Request mesh every 30 frames
//!                 camera.request_mesh_async()?;
//!             }
//!         }
//!         
//!         // Plane detection example (on first frame)
//!         if i == 0 {
//!             // Find floor plane
//!             if let Ok((plane, reset_quat, reset_trans)) = camera.find_floor_plane(None, None) {
//!                 println!("  Found floor plane: type={:?}, center=({:.2}, {:.2}, {:.2})",
//!                     plane.plane_type, plane.plane_center.x, plane.plane_center.y, plane.plane_center.z);
//!             }
//!             
//!             // Find plane at center of image
//!             let center_pixel = Vector2::new(640.0, 360.0);
//!             let plane_params = PlaneDetectionParameters::default();
//!             if let Ok(plane) = camera.find_plane_at_hit(center_pixel, &plane_params, true) {
//!                 println!("  Found plane at center: type={:?}", plane.plane_type);
//!             }
//!         }
//!         
//!         // Save snapshots periodically
//!         if i % 100 == 0 {
//!             camera.save_current_image(ViewType::Left, &format!("image_{}.png", i))?;
//!             camera.save_current_depth(Side::Left, &format!("depth_{}.png", i))?;
//!         }
//!     }
//!     
//!     // Check camera health status
//!     let health = camera.get_health_status()?;
//!     println!("Camera health: {}", health.get_status_description());
//!     if !health.is_healthy() {
//!         eprintln!("Warning: Camera health issues detected!");
//!     }
//!     
//!     // Convert coordinate systems
//!     let mut rotation = Quaternion::identity();
//!     let mut translation = Vector3::new(1.0, 2.0, 3.0);
//!     Camera::convert_coordinate_system(
//!         &mut rotation,
//!         &mut translation,
//!         CoordinateSystem::LeftHandedYUp,
//!         CoordinateSystem::RightHandedZUp
//!     )?;
//!     
//!     // Disable modules and stop recording
//!     camera.disable_streaming();
//!     camera.disable_recording();
//!     camera.disable_spatial_mapping();
//!     camera.disable_body_tracking(0, false)?;
//!     camera.disable_object_detection(0, false)?;
//!     camera.disable_positional_tracking(None)?;
//!     
//!     Ok(())
//! }
//! ```

use std::ffi::CString;
use std::fmt;

// Include the generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Errors that can occur when using the ZED SDK
#[derive(Debug, Clone)]
pub enum ZedError {
    /// Camera creation failed
    CameraCreationFailed,
    /// Camera opening failed with error code
    CameraOpenFailed(i32),
    /// Frame grabbing failed with error code
    GrabFailed(i32),
    /// Invalid camera ID
    InvalidCameraId,
    /// Camera not opened
    CameraNotOpened,
    /// Other error with description
    Other(String),
}

impl fmt::Display for ZedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZedError::CameraCreationFailed => write!(f, "Failed to create camera"),
            ZedError::CameraOpenFailed(code) => write!(f, "Failed to open camera: error code {}", code),
            ZedError::GrabFailed(code) => write!(f, "Failed to grab frame: error code {}", code),
            ZedError::InvalidCameraId => write!(f, "Invalid camera ID"),
            ZedError::CameraNotOpened => write!(f, "Camera not opened"),
            ZedError::Other(msg) => write!(f, "ZED SDK error: {}", msg),
        }
    }
}

impl std::error::Error for ZedError {}

/// Camera resolution options
#[derive(Debug, Clone, Copy)]
pub enum Resolution {
    HD2K,
    HD1080,
    HD720,
    VGA,
}

impl From<Resolution> for u32 {
    fn from(res: Resolution) -> Self {
        match res {
            Resolution::HD2K => SL_RESOLUTION_SL_RESOLUTION_HD2K,
            Resolution::HD1080 => SL_RESOLUTION_SL_RESOLUTION_HD1080,
            Resolution::HD720 => SL_RESOLUTION_SL_RESOLUTION_HD720,
            Resolution::VGA => SL_RESOLUTION_SL_RESOLUTION_VGA,
        }
    }
}

/// Depth mode options
#[derive(Debug, Clone, Copy)]
pub enum DepthMode {
    None,
    Performance,
    Quality,
    Ultra,
    Neural,
}

impl From<DepthMode> for u32 {
    fn from(mode: DepthMode) -> Self {
        match mode {
            DepthMode::None => SL_DEPTH_MODE_SL_DEPTH_MODE_NONE,
            DepthMode::Performance => SL_DEPTH_MODE_SL_DEPTH_MODE_PERFORMANCE,
            DepthMode::Quality => SL_DEPTH_MODE_SL_DEPTH_MODE_QUALITY,
            DepthMode::Ultra => SL_DEPTH_MODE_SL_DEPTH_MODE_ULTRA,
            DepthMode::Neural => SL_DEPTH_MODE_SL_DEPTH_MODE_NEURAL,
        }
    }
}

/// Camera initialization parameters
#[derive(Debug, Clone)]
pub struct InitParameters {
    pub camera_fps: i32,
    pub resolution: Resolution,
    pub depth_mode: DepthMode,
    pub depth_minimum_distance: f32,
    pub depth_maximum_distance: f32,
    pub enable_image_enhancement: bool,
    pub sdk_verbose: bool,
}

impl Default for InitParameters {
    fn default() -> Self {
        Self {
            camera_fps: 30,
            resolution: Resolution::HD1080,
            depth_mode: DepthMode::Neural,
            depth_minimum_distance: -1.0,
            depth_maximum_distance: 40.0,
            enable_image_enhancement: true,
            sdk_verbose: false,
        }
    }
}

impl InitParameters {
    /// Create new initialization parameters with default values
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the camera FPS
    pub fn with_fps(mut self, fps: i32) -> Self {
        self.camera_fps = fps;
        self
    }
    
    /// Set the camera resolution
    pub fn with_resolution(mut self, resolution: Resolution) -> Self {
        self.resolution = resolution;
        self
    }
    
    /// Set the depth mode
    pub fn with_depth_mode(mut self, depth_mode: DepthMode) -> Self {
        self.depth_mode = depth_mode;
        self
    }
    
    /// Set the minimum depth distance
    pub fn with_depth_minimum_distance(mut self, distance: f32) -> Self {
        self.depth_minimum_distance = distance;
        self
    }
    
    /// Set the maximum depth distance
    pub fn with_depth_maximum_distance(mut self, distance: f32) -> Self {
        self.depth_maximum_distance = distance;
        self
    }
    
    /// Enable or disable image enhancement
    pub fn with_image_enhancement(mut self, enable: bool) -> Self {
        self.enable_image_enhancement = enable;
        self
    }
    
    /// Enable or disable SDK verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.sdk_verbose = verbose;
        self
    }
}

/// Runtime parameters for frame grabbing
#[derive(Debug, Clone)]
pub struct RuntimeParameters {
    pub enable_depth: bool,
    pub confidence_threshold: i32,
    pub texture_confidence_threshold: i32,
    pub remove_saturated_areas: bool,
}

impl Default for RuntimeParameters {
    fn default() -> Self {
        Self {
            enable_depth: true,
            confidence_threshold: 95,
            texture_confidence_threshold: 100,
            remove_saturated_areas: true,
        }
    }
}

/// View types for image retrieval
#[derive(Debug, Clone, Copy)]
pub enum ViewType {
    Left,
    Right,
    LeftGray,
    RightGray,
    LeftUnrectified,
    RightUnrectified,
    LeftUnrectifiedGray,
    RightUnrectifiedGray,
    SideBySide,
    Depth,
    Confidence,
    Normals,
    DepthRight,
    NormalsRight,
}

impl From<ViewType> for u32 {
    fn from(view: ViewType) -> Self {
        match view {
            ViewType::Left => SL_VIEW_SL_VIEW_LEFT,
            ViewType::Right => SL_VIEW_SL_VIEW_RIGHT,
            ViewType::LeftGray => SL_VIEW_SL_VIEW_LEFT_GRAY,
            ViewType::RightGray => SL_VIEW_SL_VIEW_RIGHT_GRAY,
            ViewType::LeftUnrectified => SL_VIEW_SL_VIEW_LEFT_UNRECTIFIED,
            ViewType::RightUnrectified => SL_VIEW_SL_VIEW_RIGHT_UNRECTIFIED,
            ViewType::LeftUnrectifiedGray => SL_VIEW_SL_VIEW_LEFT_UNRECTIFIED_GRAY,
            ViewType::RightUnrectifiedGray => SL_VIEW_SL_VIEW_RIGHT_UNRECTIFIED_GRAY,
            ViewType::SideBySide => SL_VIEW_SL_VIEW_SIDE_BY_SIDE,
            ViewType::Depth => SL_VIEW_SL_VIEW_DEPTH,
            ViewType::Confidence => SL_VIEW_SL_VIEW_CONFIDENCE,
            ViewType::Normals => SL_VIEW_SL_VIEW_NORMALS,
            ViewType::DepthRight => SL_VIEW_SL_VIEW_DEPTH_RIGHT,
            ViewType::NormalsRight => SL_VIEW_SL_VIEW_NORMALS_RIGHT,
        }
    }
}

/// Measure types for data retrieval
#[derive(Debug, Clone, Copy)]
pub enum MeasureType {
    Disparity,
    Depth,
    Confidence,
    Xyz,
    XyzRgba,
    XyzBgra,
    XyzArgb,
    XyzAbgr,
    Normals,
    DisparityRight,
    DepthRight,
    XyzRight,
    XyzRgbaRight,
    XyzBgraRight,
    XyzArgbRight,
    XyzAbgrRight,
    NormalsRight,
    DepthU16Mm,
    DepthU16MmRight,
}

impl From<MeasureType> for u32 {
    fn from(measure: MeasureType) -> Self {
        match measure {
            MeasureType::Disparity => SL_MEASURE_SL_MEASURE_DISPARITY,
            MeasureType::Depth => SL_MEASURE_SL_MEASURE_DEPTH,
            MeasureType::Confidence => SL_MEASURE_SL_MEASURE_CONFIDENCE,
            MeasureType::Xyz => SL_MEASURE_SL_MEASURE_XYZ,
            MeasureType::XyzRgba => SL_MEASURE_SL_MEASURE_XYZRGBA,
            MeasureType::XyzBgra => SL_MEASURE_SL_MEASURE_XYZBGRA,
            MeasureType::XyzArgb => SL_MEASURE_SL_MEASURE_XYZARGB,
            MeasureType::XyzAbgr => SL_MEASURE_SL_MEASURE_XYZABGR,
            MeasureType::Normals => SL_MEASURE_SL_MEASURE_NORMALS,
            MeasureType::DisparityRight => SL_MEASURE_SL_MEASURE_DISPARITY_RIGHT,
            MeasureType::DepthRight => SL_MEASURE_SL_MEASURE_DEPTH_RIGHT,
            MeasureType::XyzRight => SL_MEASURE_SL_MEASURE_XYZ_RIGHT,
            MeasureType::XyzRgbaRight => SL_MEASURE_SL_MEASURE_XYZRGBA_RIGHT,
            MeasureType::XyzBgraRight => SL_MEASURE_SL_MEASURE_XYZBGRA_RIGHT,
            MeasureType::XyzArgbRight => SL_MEASURE_SL_MEASURE_XYZARGB_RIGHT,
            MeasureType::XyzAbgrRight => SL_MEASURE_SL_MEASURE_XYZABGR_RIGHT,
            MeasureType::NormalsRight => SL_MEASURE_SL_MEASURE_NORMALS_RIGHT,
            MeasureType::DepthU16Mm => SL_MEASURE_SL_MEASURE_DEPTH_U16_MM,
            MeasureType::DepthU16MmRight => SL_MEASURE_SL_MEASURE_DEPTH_U16_MM_RIGHT,
        }
    }
}

/// Memory type for data storage
#[derive(Debug, Clone, Copy)]
pub enum MemoryType {
    Cpu,
    Gpu,
    Both,
}

impl From<MemoryType> for u32 {
    fn from(mem: MemoryType) -> Self {
        match mem {
            MemoryType::Cpu => SL_MEM_SL_MEM_CPU,
            MemoryType::Gpu => SL_MEM_SL_MEM_GPU,
            MemoryType::Both => SL_MEM_SL_MEM_BOTH,
        }
    }
}

/// Image data container
#[derive(Debug)]
pub struct ImageData {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
}

impl ImageData {
    /// Create a new ImageData with the specified dimensions
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        let size = width * height * channels;
        Self {
            data: vec![0u8; size],
            width,
            height,
            channels,
        }
    }

    /// Get the total size in bytes
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get a slice of the raw data
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get a mutable slice of the raw data
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

/// Depth data container
#[derive(Debug)]
pub struct DepthData {
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

impl DepthData {
    /// Create a new DepthData with the specified dimensions
    pub fn new(width: usize, height: usize) -> Self {
        let size = width * height;
        Self {
            data: vec![0.0f32; size],
            width,
            height,
        }
    }

    /// Get the depth value at the specified pixel coordinates
    pub fn get_depth(&self, x: usize, y: usize) -> Option<f32> {
        if x < self.width && y < self.height {
            Some(self.data[y * self.width + x])
        } else {
            None
        }
    }

    /// Get a slice of the raw data
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// Point cloud data container
#[derive(Debug)]
pub struct PointCloudData {
    pub data: Vec<f32>, // 4 floats per point (X, Y, Z, W)
    pub width: usize,
    pub height: usize,
}

impl PointCloudData {
    /// Create a new PointCloudData with the specified dimensions
    pub fn new(width: usize, height: usize) -> Self {
        let size = width * height * 4; // 4 floats per point
        Self {
            data: vec![0.0f32; size],
            width,
            height,
        }
    }

    /// Get the 3D point at the specified pixel coordinates
    pub fn get_point(&self, x: usize, y: usize) -> Option<(f32, f32, f32)> {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) * 4;
            Some((self.data[idx], self.data[idx + 1], self.data[idx + 2]))
        } else {
            None
        }
    }

    /// Get a slice of the raw data
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// Confidence data container
#[derive(Debug)]
pub struct ConfidenceData {
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

impl ConfidenceData {
    /// Create a new ConfidenceData with the specified dimensions
    pub fn new(width: usize, height: usize) -> Self {
        let size = width * height;
        Self {
            data: vec![0.0f32; size],
            width,
            height,
        }
    }

    /// Get the confidence value at the specified pixel coordinates
    pub fn get_confidence(&self, x: usize, y: usize) -> Option<f32> {
        if x < self.width && y < self.height {
            Some(self.data[y * self.width + x])
        } else {
            None
        }
    }

    /// Get a slice of the raw data
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// Reference frame for positional tracking
#[derive(Debug, Clone, Copy)]
pub enum ReferenceFrame {
    World,
    Camera,
}

impl From<ReferenceFrame> for u32 {
    fn from(frame: ReferenceFrame) -> Self {
        match frame {
            ReferenceFrame::World => SL_REFERENCE_FRAME_SL_REFERENCE_FRAME_WORLD,
            ReferenceFrame::Camera => SL_REFERENCE_FRAME_SL_REFERENCE_FRAME_CAMERA,
        }
    }
}

/// 3D vector representation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vector3 {
    /// Create a new Vector3
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Create a zero vector
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Get the magnitude of the vector
    pub fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalize the vector
    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag > 0.0 {
            Self::new(self.x / mag, self.y / mag, self.z / mag)
        } else {
            *self
        }
    }
}

impl From<Vector3> for SL_Vector3 {
    fn from(v: Vector3) -> Self {
        SL_Vector3 { x: v.x, y: v.y, z: v.z }
    }
}

impl From<SL_Vector3> for Vector3 {
    fn from(v: SL_Vector3) -> Self {
        Vector3 { x: v.x, y: v.y, z: v.z }
    }
}

/// Quaternion representation for rotations
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quaternion {
    /// Create a new Quaternion
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Create an identity quaternion (no rotation)
    pub fn identity() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }

    /// Normalize the quaternion
    pub fn normalize(&self) -> Self {
        let mag = (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt();
        if mag > 0.0 {
            Self::new(self.x / mag, self.y / mag, self.z / mag, self.w / mag)
        } else {
            *self
        }
    }

    /// Convert quaternion to Euler angles (roll, pitch, yaw) in radians
    pub fn to_euler_angles(&self) -> (f32, f32, f32) {
        let roll = (2.0 * (self.w * self.x + self.y * self.z)).atan2(1.0 - 2.0 * (self.x * self.x + self.y * self.y));
        let pitch = (2.0 * (self.w * self.y - self.z * self.x)).asin();
        let yaw = (2.0 * (self.w * self.z + self.x * self.y)).atan2(1.0 - 2.0 * (self.y * self.y + self.z * self.z));
        (roll, pitch, yaw)
    }
}

impl From<Quaternion> for SL_Quaternion {
    fn from(q: Quaternion) -> Self {
        SL_Quaternion { x: q.x, y: q.y, z: q.z, w: q.w }
    }
}

impl From<SL_Quaternion> for Quaternion {
    fn from(q: SL_Quaternion) -> Self {
        Quaternion { x: q.x, y: q.y, z: q.z, w: q.w }
    }
}

/// Pose data containing position and orientation information
#[derive(Debug, Clone)]
pub struct PoseData {
    pub valid: bool,
    pub timestamp: u64,
    pub rotation: Quaternion,
    pub translation: Vector3,
    pub pose_confidence: i32,
    pub pose_covariance: [f32; 36],
}

impl PoseData {
    /// Check if the pose data is valid and tracking is working
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Get the pose confidence as a percentage (0-100)
    pub fn confidence_percentage(&self) -> i32 {
        self.pose_confidence
    }

    /// Get Euler angles from the rotation quaternion
    pub fn get_euler_angles(&self) -> (f32, f32, f32) {
        self.rotation.to_euler_angles()
    }
}

impl From<SL_PoseData> for PoseData {
    fn from(pose: SL_PoseData) -> Self {
        PoseData {
            valid: pose.valid,
            timestamp: pose.timestamp,
            rotation: pose.rotation.into(),
            translation: pose.translation.into(),
            pose_confidence: pose.pose_confidence,
            pose_covariance: pose.pose_covariance,
        }
    }
}

/// Parameters for positional tracking initialization
#[derive(Debug, Clone)]
pub struct PositionalTrackingParameters {
    pub initial_world_rotation: Quaternion,
    pub initial_world_position: Vector3,
    pub enable_area_memory: bool,
    pub enable_pose_smoothing: bool,
    pub set_floor_as_origin: bool,
    pub set_as_static: bool,
    pub enable_imu_fusion: bool,
    pub set_gravity_as_origin: bool,
    pub mode: TrackingMode,
}

/// Tracking mode options
#[derive(Debug, Clone, Copy)]
pub enum TrackingMode {
    Standard,
    Quality,
    Gen1,
}

impl Default for PositionalTrackingParameters {
    fn default() -> Self {
        Self {
            initial_world_rotation: Quaternion::identity(),
            initial_world_position: Vector3::zero(),
            enable_area_memory: true,
            enable_pose_smoothing: false,
            set_floor_as_origin: false,
            set_as_static: false,
            enable_imu_fusion: true,
            set_gravity_as_origin: true,
            mode: TrackingMode::Standard,
        }
    }
}

impl PositionalTrackingParameters {
    /// Create new positional tracking parameters with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the initial world rotation
    pub fn with_initial_world_rotation(mut self, rotation: Quaternion) -> Self {
        self.initial_world_rotation = rotation;
        self
    }

    /// Set the initial world position
    pub fn with_initial_world_position(mut self, position: Vector3) -> Self {
        self.initial_world_position = position;
        self
    }

    /// Enable or disable area memory
    pub fn with_area_memory(mut self, enable: bool) -> Self {
        self.enable_area_memory = enable;
        self
    }

    /// Enable or disable pose smoothing
    pub fn with_pose_smoothing(mut self, enable: bool) -> Self {
        self.enable_pose_smoothing = enable;
        self
    }

    /// Set floor as origin
    pub fn with_floor_as_origin(mut self, enable: bool) -> Self {
        self.set_floor_as_origin = enable;
        self
    }

    /// Set camera as static
    pub fn with_static_camera(mut self, enable: bool) -> Self {
        self.set_as_static = enable;
        self
    }

    /// Enable or disable IMU fusion
    pub fn with_imu_fusion(mut self, enable: bool) -> Self {
        self.enable_imu_fusion = enable;
        self
    }

    /// Set gravity as origin
    pub fn with_gravity_as_origin(mut self, enable: bool) -> Self {
        self.set_gravity_as_origin = enable;
        self
    }

    /// Set tracking mode
    pub fn with_mode(mut self, mode: TrackingMode) -> Self {
        self.mode = mode;
        self
    }
}

/// Device properties
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub camera_state: i32,
    pub id: i32,
    pub path: String,
    pub camera_model: u32,
    pub serial_number: u32,
}

impl From<SL_DeviceProperties> for DeviceProperties {
    fn from(props: SL_DeviceProperties) -> Self {
        let path = unsafe {
            std::ffi::CStr::from_ptr(props.path.as_ptr())
                .to_string_lossy()
                .into_owned()
        };
        
        DeviceProperties {
            camera_state: props.camera_state,
            id: props.id,
            path,
            camera_model: props.camera_model,
            serial_number: props.sn,
        }
    }
}

/// Streaming properties
#[derive(Debug, Clone)]
pub struct StreamingProperties {
    pub ip: String,
    pub port: u16,
    pub serial_number: u32,
    pub current_bitrate: i32,
    pub codec: StreamingCodec,
}

impl From<SL_StreamingProperties> for StreamingProperties {
    fn from(props: SL_StreamingProperties) -> Self {
        let ip = unsafe {
            std::ffi::CStr::from_ptr(props.ip.as_ptr())
                .to_string_lossy()
                .into_owned()
        };
        
        StreamingProperties {
            ip,
            port: props.port,
            serial_number: props.serial_number,
            current_bitrate: props.current_bitrate,
            codec: match props.codec {
                SL_STREAMING_CODEC_SL_STREAMING_CODEC_H265 => StreamingCodec::H265,
                _ => StreamingCodec::H264,
            },
        }
    }
}

/// High-level camera interface
pub struct Camera {
    camera_id: i32,
    is_opened: bool,
}

impl Camera {
    /// Create a new camera instance
    pub fn new(camera_id: i32) -> Result<Self, ZedError> {
        unsafe {
            let success = sl_create_camera(camera_id);
            if !success {
                return Err(ZedError::CameraCreationFailed);
            }
        }
        
        Ok(Camera {
            camera_id,
            is_opened: false,
        })
    }

    /// Get list of connected devices
    pub fn get_device_list() -> Vec<DeviceProperties> {
        unsafe {
            let mut device_list: [SL_DeviceProperties; MAX_CAMERA_PLUGIN as usize] = std::mem::zeroed();
            let mut nb_devices = 0i32;
            
            sl_get_device_list(device_list.as_mut_ptr(), &mut nb_devices);
            
            let mut devices = Vec::new();
            for i in 0..nb_devices as usize {
                if i < device_list.len() {
                    devices.push(device_list[i].into());
                }
            }
            
            devices
        }
    }

    /// Get list of streaming devices
    pub fn get_streaming_device_list() -> Vec<StreamingProperties> {
        unsafe {
            let mut streaming_list: [SL_StreamingProperties; MAX_CAMERA_PLUGIN as usize] = std::mem::zeroed();
            let mut nb_devices = 0i32;
            
            sl_get_streaming_device_list(streaming_list.as_mut_ptr(), &mut nb_devices);
            
            let mut devices = Vec::new();
            for i in 0..nb_devices as usize {
                if i < streaming_list.len() {
                    devices.push(streaming_list[i].into());
                }
            }
            
            devices
        }
    }

    /// Get number of connected ZED cameras
    pub fn get_number_zed_connected() -> i32 {
        unsafe {
            sl_get_number_zed_connected()
        }
    }

    /// Reboot a camera
    pub fn reboot(serial_number: i32, full_reboot: bool) -> Result<(), ZedError> {
        unsafe {
            let result = sl_reboot(serial_number, full_reboot);
            if result != 0 {
                return Err(ZedError::Other(format!("Failed to reboot camera: error code {}", result)));
            }
        }
        Ok(())
    }

    /// Get SDK version
    pub fn get_sdk_version() -> String {
        unsafe {
            let version_ptr = sl_get_sdk_version();
            if version_ptr.is_null() {
                return String::from("Unknown");
            }
            std::ffi::CStr::from_ptr(version_ptr)
                .to_string_lossy()
                .into_owned()
        }
    }

    /// Convert coordinate system of a transform
    pub fn convert_coordinate_system(
        rotation: &mut Quaternion,
        translation: &mut Vector3,
        coord_system_src: CoordinateSystem,
        coord_system_dest: CoordinateSystem,
    ) -> Result<(), ZedError> {
        unsafe {
            let mut sl_rotation: SL_Quaternion = (*rotation).into();
            let mut sl_translation: SL_Vector3 = (*translation).into();
            
            let result = sl_convert_coordinate_system(
                &mut sl_rotation,
                &mut sl_translation,
                coord_system_src.into(),
                coord_system_dest.into(),
            );
            
            if result != 0 {
                return Err(ZedError::Other(format!("Failed to convert coordinate system: error code {}", result)));
            }
            
            *rotation = sl_rotation.into();
            *translation = sl_translation.into();
            
            Ok(())
        }
    }
    
    /// Open the camera with the given parameters
    pub fn open(&mut self, params: &InitParameters) -> Result<(), ZedError> {
        unsafe {
            let mut init_param = SL_InitParameters {
                camera_fps: params.camera_fps,
                resolution: params.resolution.into(),
                input_type: SL_INPUT_TYPE_SL_INPUT_TYPE_USB,
                camera_device_id: self.camera_id,
                camera_image_flip: SL_FLIP_MODE_SL_FLIP_MODE_AUTO,
                camera_disable_self_calib: false,
                enable_image_enhancement: params.enable_image_enhancement,
                svo_real_time_mode: true,
                depth_mode: params.depth_mode.into(),
                depth_stabilization: 30,
                depth_maximum_distance: params.depth_maximum_distance,
                depth_minimum_distance: params.depth_minimum_distance,
                coordinate_unit: SL_UNIT_SL_UNIT_METER,
                coordinate_system: SL_COORDINATE_SYSTEM_SL_COORDINATE_SYSTEM_LEFT_HANDED_Y_UP,
                sdk_gpu_id: -1,
                sdk_verbose: if params.sdk_verbose { 1 } else { 0 },
                sensors_required: false,
                enable_right_side_measure: false,
                open_timeout_sec: 5.0,
                async_grab_camera_recovery: false,
                grab_compute_capping_fps: 0.0,
                enable_image_validity_check: false,
                ..std::mem::zeroed()
            };

            let empty_str = CString::new("").unwrap();
            let state = sl_open_camera(
                self.camera_id,
                &mut init_param,
                0,
                empty_str.as_ptr(),
                empty_str.as_ptr(),
                0,
                empty_str.as_ptr(),
                empty_str.as_ptr(),
                empty_str.as_ptr(),
            );

            if state != 0 {
                return Err(ZedError::CameraOpenFailed(state));
            }
        }
        
        self.is_opened = true;
        Ok(())
    }
    
    /// Get the camera serial number
    pub fn get_serial_number(&self) -> Result<i32, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }
        
        unsafe {
            Ok(sl_get_zed_serial(self.camera_id))
        }
    }
    
    /// Get camera resolution
    pub fn get_resolution(&self) -> Result<(i32, i32), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }
        
        unsafe {
            let width = sl_get_width(self.camera_id);
            let height = sl_get_height(self.camera_id);
            Ok((width, height))
        }
    }
    
    /// Grab a new frame
    pub fn grab(&self) -> Result<(), ZedError> {
        self.grab_with_params(&RuntimeParameters::default())
    }
    
    /// Grab a new frame with custom runtime parameters
    pub fn grab_with_params(&self, params: &RuntimeParameters) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }
        
        unsafe {
            let mut rt_param = SL_RuntimeParameters {
                enable_depth: params.enable_depth,
                confidence_threshold: params.confidence_threshold,
                reference_frame: SL_REFERENCE_FRAME_SL_REFERENCE_FRAME_CAMERA,
                texture_confidence_threshold: params.texture_confidence_threshold,
                remove_saturated_areas: params.remove_saturated_areas,
                ..std::mem::zeroed()
            };
            
            let state = sl_grab(self.camera_id, &mut rt_param);
            if state != 0 {
                return Err(ZedError::GrabFailed(state));
            }
        }
        
        Ok(())
    }
    
    /// Get the timestamp of the current frame
    pub fn get_timestamp(&self) -> Result<u64, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }
        
        unsafe {
            Ok(sl_get_current_timestamp(self.camera_id))
        }
    }
    
    /// Check if the camera is opened
    pub fn is_opened(&self) -> bool {
        self.is_opened
    }
    
    /// Get the camera ID
    pub fn camera_id(&self) -> i32 {
        self.camera_id
    }

    /// Check if camera is opened (using SDK function)
    pub fn is_opened_sdk(&self) -> bool {
        unsafe {
            sl_is_opened(self.camera_id)
        }
    }

    /// Get input type
    pub fn get_input_type(&self) -> Result<i32, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            Ok(sl_get_input_type(self.camera_id))
        }
    }

    /// Get confidence threshold
    pub fn get_confidence_threshold(&self) -> Result<i32, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            Ok(sl_get_confidence_threshold(self.camera_id))
        }
    }

    /// Update self calibration
    pub fn update_self_calibration(&self) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            sl_update_self_calibration(self.camera_id);
        }

        Ok(())
    }

    /// Get frame dropped count
    pub fn get_frame_dropped_count(&self) -> Result<u32, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            Ok(sl_get_frame_dropped_count(self.camera_id))
        }
    }

    /// Get the current health status of the camera
    pub fn get_health_status(&self) -> Result<HealthStatus, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let status_ptr = sl_get_health_status(self.camera_id);
            if status_ptr.is_null() {
                return Err(ZedError::Other("Failed to get health status".to_string()));
            }
            
            let status = (*status_ptr).into();
            Ok(status)
        }
    }

    /// Save area map for positional tracking
    pub fn save_area_map(&self, area_file_path: &str) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        let path_cstr = CString::new(area_file_path)
            .map_err(|_| ZedError::Other("Invalid area file path".to_string()))?;

        unsafe {
            let result = sl_save_area_map(self.camera_id, path_cstr.as_ptr());
            if result != 0 {
                return Err(ZedError::Other(format!("Failed to save area map: error code {}", result)));
            }
        }

        Ok(())
    }

    /// Get area export state
    pub fn get_area_export_state(&self) -> Result<i32, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            Ok(sl_get_area_export_state(self.camera_id))
        }
    }

    /// Check if positional tracking is enabled
    pub fn is_positional_tracking_enabled(&self) -> Result<bool, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            Ok(sl_is_positional_tracking_enabled(self.camera_id))
        }
    }

    /// Set IMU prior orientation
    pub fn set_imu_prior_orientation(&self, rotation: Quaternion) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let result = sl_set_imu_prior_orientation(self.camera_id, rotation.into());
            if result != 0 {
                return Err(ZedError::Other(format!("Failed to set IMU prior orientation: error code {}", result)));
            }
        }

        Ok(())
    }

    /// Retrieve an image from the camera
    pub fn retrieve_image(&self, view_type: ViewType, memory_type: MemoryType) -> Result<ImageData, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        let (width, height) = self.get_resolution()?;
        let channels = match view_type {
            ViewType::LeftGray | ViewType::RightGray | 
            ViewType::LeftUnrectifiedGray | ViewType::RightUnrectifiedGray => 1,
            _ => 4, // BGRA
        };

        let mut image_data = ImageData::new(width as usize, height as usize, channels);

        unsafe {
            let result = sl_retrieve_image(
                self.camera_id,
                image_data.data.as_mut_ptr() as *mut std::ffi::c_void,
                view_type.into(),
                memory_type.into(),
                width,
                height,
                std::ptr::null_mut(),
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to retrieve image: error code {}", result)));
            }
        }

        Ok(image_data)
    }

    /// Retrieve depth data from the camera
    pub fn retrieve_depth(&self, memory_type: MemoryType) -> Result<DepthData, ZedError> {
        self.retrieve_measure_f32(MeasureType::Depth, memory_type)
            .map(|data| DepthData {
                data: data.data,
                width: data.width,
                height: data.height,
            })
    }

    /// Retrieve confidence data from the camera
    pub fn retrieve_confidence(&self, memory_type: MemoryType) -> Result<ConfidenceData, ZedError> {
        self.retrieve_measure_f32(MeasureType::Confidence, memory_type)
            .map(|data| ConfidenceData {
                data: data.data,
                width: data.width,
                height: data.height,
            })
    }

    /// Retrieve point cloud data from the camera
    pub fn retrieve_point_cloud(&self, memory_type: MemoryType) -> Result<PointCloudData, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        let (width, height) = self.get_resolution()?;
        let mut point_cloud_data = PointCloudData::new(width as usize, height as usize);

        unsafe {
            let result = sl_retrieve_measure(
                self.camera_id,
                point_cloud_data.data.as_mut_ptr() as *mut std::ffi::c_void,
                MeasureType::Xyz.into(),
                memory_type.into(),
                width,
                height,
                std::ptr::null_mut(),
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to retrieve point cloud: error code {}", result)));
            }
        }

        Ok(point_cloud_data)
    }

    /// Retrieve colored point cloud data from the camera
    pub fn retrieve_colored_point_cloud(&self, color_format: MeasureType, memory_type: MemoryType) -> Result<PointCloudData, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        // Validate that the measure type is a colored point cloud type
        match color_format {
            MeasureType::XyzRgba | MeasureType::XyzBgra | 
            MeasureType::XyzArgb | MeasureType::XyzAbgr |
            MeasureType::XyzRgbaRight | MeasureType::XyzBgraRight |
            MeasureType::XyzArgbRight | MeasureType::XyzAbgrRight => {},
            _ => return Err(ZedError::Other("Invalid color format for colored point cloud".to_string())),
        }

        let (width, height) = self.get_resolution()?;
        let mut point_cloud_data = PointCloudData::new(width as usize, height as usize);

        unsafe {
            let result = sl_retrieve_measure(
                self.camera_id,
                point_cloud_data.data.as_mut_ptr() as *mut std::ffi::c_void,
                color_format.into(),
                memory_type.into(),
                width,
                height,
                std::ptr::null_mut(),
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to retrieve colored point cloud: error code {}", result)));
            }
        }

        Ok(point_cloud_data)
    }

    /// Retrieve disparity data from the camera
    pub fn retrieve_disparity(&self, memory_type: MemoryType) -> Result<DepthData, ZedError> {
        self.retrieve_measure_f32(MeasureType::Disparity, memory_type)
            .map(|data| DepthData {
                data: data.data,
                width: data.width,
                height: data.height,
            })
    }

    /// Retrieve normal vectors from the camera
    pub fn retrieve_normals(&self, memory_type: MemoryType) -> Result<PointCloudData, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        let (width, height) = self.get_resolution()?;
        let mut normals_data = PointCloudData::new(width as usize, height as usize);

        unsafe {
            let result = sl_retrieve_measure(
                self.camera_id,
                normals_data.data.as_mut_ptr() as *mut std::ffi::c_void,
                MeasureType::Normals.into(),
                memory_type.into(),
                width,
                height,
                std::ptr::null_mut(),
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to retrieve normals: error code {}", result)));
            }
        }

        Ok(normals_data)
    }

    /// Retrieve depth data as 16-bit unsigned integers in millimeters
    pub fn retrieve_depth_u16_mm(&self, memory_type: MemoryType) -> Result<Vec<u16>, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        let (width, height) = self.get_resolution()?;
        let size = (width * height) as usize;
        let mut depth_data = vec![0u16; size];

        unsafe {
            let result = sl_retrieve_measure(
                self.camera_id,
                depth_data.as_mut_ptr() as *mut std::ffi::c_void,
                MeasureType::DepthU16Mm.into(),
                memory_type.into(),
                width,
                height,
                std::ptr::null_mut(),
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to retrieve depth U16: error code {}", result)));
            }
        }

        Ok(depth_data)
    }

    /// Generic method to retrieve f32 measure data
    fn retrieve_measure_f32(&self, measure_type: MeasureType, memory_type: MemoryType) -> Result<DepthData, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        let (width, height) = self.get_resolution()?;
        let mut measure_data = DepthData::new(width as usize, height as usize);

        unsafe {
            let result = sl_retrieve_measure(
                self.camera_id,
                measure_data.data.as_mut_ptr() as *mut std::ffi::c_void,
                measure_type.into(),
                memory_type.into(),
                width,
                height,
                std::ptr::null_mut(),
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to retrieve measure: error code {}", result)));
            }
        }

        Ok(measure_data)
    }

    /// Get the minimum and maximum depth values from the current frame
    pub fn get_current_min_max_depth(&self) -> Result<(f32, f32), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut min_depth = 0.0f32;
            let mut max_depth = 0.0f32;
            
            let result = sl_get_current_min_max_depth(
                self.camera_id,
                &mut min_depth,
                &mut max_depth,
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to get min/max depth: error code {}", result)));
            }

            Ok((min_depth, max_depth))
        }
    }

    /// Get the minimum depth range value
    pub fn get_depth_min_range_value(&self) -> Result<f32, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            Ok(sl_get_depth_min_range_value(self.camera_id))
        }
    }

    /// Get the maximum depth range value
    pub fn get_depth_max_range_value(&self) -> Result<f32, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            Ok(sl_get_depth_max_range_value(self.camera_id))
        }
    }

    /// Get the image timestamp
    pub fn get_image_timestamp(&self) -> Result<u64, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            Ok(sl_get_image_timestamp(self.camera_id))
        }
    }

    /// Enable positional tracking
    pub fn enable_positional_tracking(&self, params: &PositionalTrackingParameters, area_file_path: Option<&str>) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut tracking_param = SL_PositionalTrackingParameters {
                initial_world_rotation: params.initial_world_rotation.into(),
                initial_world_position: params.initial_world_position.into(),
                enable_area_memory: params.enable_area_memory,
                enable_pose_smoothing: params.enable_pose_smoothing,
                set_floor_as_origin: params.set_floor_as_origin,
                set_as_static: params.set_as_static,
                enable_imu_fusion: params.enable_imu_fusion,
                set_gravity_as_origin: params.set_gravity_as_origin,
                mode: match params.mode {
                    TrackingMode::Standard => 0, // SL_POSITIONAL_TRACKING_MODE_STANDARD
                    TrackingMode::Quality => 1,  // SL_POSITIONAL_TRACKING_MODE_QUALITY
                    TrackingMode::Gen1 => 2,     // SL_POSITIONAL_TRACKING_MODE_GEN_1
                },
                ..std::mem::zeroed()
            };

            let area_file_cstr = if let Some(path) = area_file_path {
                CString::new(path).map_err(|_| ZedError::Other("Invalid area file path".to_string()))?
            } else {
                CString::new("").unwrap()
            };

            let result = sl_enable_positional_tracking(
                self.camera_id,
                &mut tracking_param,
                area_file_cstr.as_ptr(),
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to enable positional tracking: error code {}", result)));
            }
        }

        Ok(())
    }

    /// Disable positional tracking
    pub fn disable_positional_tracking(&self, area_file_path: Option<&str>) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let area_file_cstr = if let Some(path) = area_file_path {
                CString::new(path).map_err(|_| ZedError::Other("Invalid area file path".to_string()))?
            } else {
                CString::new("").unwrap()
            };

            sl_disable_positional_tracking(self.camera_id, area_file_cstr.as_ptr());
        }

        Ok(())
    }

    /// Get the current camera pose
    pub fn get_position(&self, reference_frame: ReferenceFrame) -> Result<(Quaternion, Vector3), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut rotation = SL_Quaternion { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };
            let mut position = SL_Vector3 { x: 0.0, y: 0.0, z: 0.0 };

            let result = sl_get_position(
                self.camera_id,
                &mut rotation,
                &mut position,
                reference_frame.into(),
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to get position: error code {}", result)));
            }

            Ok((rotation.into(), position.into()))
        }
    }

    /// Get detailed pose data
    pub fn get_pose_data(&self, reference_frame: ReferenceFrame) -> Result<PoseData, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut pose_data: SL_PoseData = std::mem::zeroed();

            let result = sl_get_position_data(
                self.camera_id,
                &mut pose_data,
                reference_frame.into(),
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to get pose data: error code {}", result)));
            }

            Ok(pose_data.into())
        }
    }

    /// Get position as a 4x4 transformation matrix (column-major order)
    pub fn get_position_array(&self, reference_frame: ReferenceFrame) -> Result<[f32; 16], ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut pose_array = [0.0f32; 16];

            let result = sl_get_position_array(
                self.camera_id,
                pose_array.as_mut_ptr(),
                reference_frame.into(),
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to get position array: error code {}", result)));
            }

            Ok(pose_array)
        }
    }

    /// Reset positional tracking with optional initial pose
    pub fn reset_positional_tracking(&self, rotation: Option<Quaternion>, translation: Option<Vector3>) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let rot = rotation.unwrap_or(Quaternion::identity()).into();
            let trans = translation.unwrap_or(Vector3::zero()).into();

            let result = sl_reset_positional_tracking(self.camera_id, rot, trans);

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to reset positional tracking: error code {}", result)));
            }
        }

        Ok(())
    }

    /// Reset positional tracking with offset
    pub fn reset_positional_tracking_with_offset(
        &self,
        rotation: Quaternion,
        translation: Vector3,
        target_rotation: Quaternion,
        target_translation: Vector3,
    ) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let result = sl_reset_positional_tracking_with_offset(
                self.camera_id,
                rotation.into(),
                translation.into(),
                target_rotation.into(),
                target_translation.into(),
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to reset positional tracking with offset: error code {}", result)));
            }
        }

        Ok(())
    }

    /// Get position at a specific target frame
    pub fn get_position_at_target_frame(
        &self,
        target_rotation: Quaternion,
        target_translation: Vector3,
        reference_frame: ReferenceFrame,
    ) -> Result<(Quaternion, Vector3), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut rotation = SL_Quaternion { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };
            let mut position = SL_Vector3 { x: 0.0, y: 0.0, z: 0.0 };
            let mut target_rot = target_rotation.into();
            let mut target_trans = target_translation.into();

            let result = sl_get_position_at_target_frame(
                self.camera_id,
                &mut rotation,
                &mut position,
                &mut target_rot,
                &mut target_trans,
                reference_frame.into(),
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to get position at target frame: error code {}", result)));
            }

            Ok((rotation.into(), position.into()))
        }
    }

    /// Enable object detection
    pub fn enable_object_detection(&self, params: &ObjectDetectionParameters) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut detection_param = SL_ObjectDetectionParameters {
                instance_module_id: params.instance_module_id,
                enable_tracking: params.enable_tracking,
                enable_segmentation: params.enable_segmentation,
                detection_model: params.detection_model.into(),
                max_range: params.max_range,
                filtering_mode: params.filtering_mode.into(),
                prediction_timeout_s: params.prediction_timeout_s,
                allow_reduced_precision_inference: params.allow_reduced_precision_inference,
                ..std::mem::zeroed()
            };

            let result = sl_enable_object_detection(self.camera_id, &mut detection_param);

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to enable object detection: error code {}", result)));
            }
        }

        Ok(())
    }

    /// Disable object detection
    pub fn disable_object_detection(&self, instance_id: u32, force_disable_all: bool) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            sl_disable_object_detection(self.camera_id, instance_id, force_disable_all);
        }

        Ok(())
    }

    /// Retrieve detected objects
    pub fn retrieve_objects(&self, runtime_params: &ObjectDetectionRuntimeParameters, instance_id: u32) -> Result<Objects, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut objects_data: SL_Objects = std::mem::zeroed();
            
            // Prepare runtime parameters
            let mut rt_param = SL_ObjectDetectionRuntimeParameters {
                detection_confidence_threshold: runtime_params.detection_confidence_threshold,
                ..std::mem::zeroed()
            };

            // Set object class filter if provided
            if !runtime_params.object_class_filter.is_empty() {
                for (i, &class) in runtime_params.object_class_filter.iter().enumerate() {
                    if i < rt_param.object_class_filter.len() {
                        rt_param.object_class_filter[i] = u32::from(class) as i32;
                    }
                }
            }

            // Set object confidence thresholds if provided
            if !runtime_params.object_confidence_threshold.is_empty() {
                for (i, &threshold) in runtime_params.object_confidence_threshold.iter().enumerate() {
                    if i < rt_param.object_confidence_threshold.len() {
                        rt_param.object_confidence_threshold[i] = threshold;
                    }
                }
            }

            let result = sl_retrieve_objects(
                self.camera_id,
                &mut rt_param,
                &mut objects_data,
                instance_id,
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to retrieve objects: error code {}", result)));
            }

            // Convert C data to Rust structures
            let mut objects = Vec::new();
            for i in 0..objects_data.nb_objects as usize {
                if i >= objects_data.object_list.len() {
                    break;
                }

                let obj = &objects_data.object_list[i];
                
                // Convert bounding box 2D
                let mut bbox_2d = [Vector2::zero(); 4];
                for j in 0..4 {
                    bbox_2d[j] = obj.bounding_box_2d[j].into();
                }

                // Convert bounding box 3D
                let mut bbox_3d = [Vector3::zero(); 8];
                for j in 0..8 {
                    bbox_3d[j] = obj.bounding_box[j].into();
                }

                // Convert head bounding box 2D
                let mut head_bbox_2d = [Vector2::zero(); 4];
                for j in 0..4 {
                    head_bbox_2d[j] = obj.head_bounding_box_2d[j].into();
                }

                // Convert head bounding box 3D
                let mut head_bbox_3d = [Vector3::zero(); 8];
                for j in 0..8 {
                    head_bbox_3d[j] = obj.head_bounding_box[j].into();
                }

                let object_data = ObjectData {
                    id: obj.id,
                    unique_object_id: format!("obj_{}", obj.id), // Simplified unique ID
                    raw_label: obj.raw_label,
                    label: obj.label.into(),
                    sublabel: obj.sublabel as i32,
                    confidence: obj.confidence,
                    tracking_state: obj.tracking_state.into(),
                    position: obj.position.into(),
                    velocity: obj.velocity.into(),
                    dimensions: obj.dimensions.into(),
                    bounding_box_2d: bbox_2d,
                    bounding_box: bbox_3d,
                    head_bounding_box_2d: head_bbox_2d,
                    head_bounding_box: head_bbox_3d,
                    head_position: obj.head_position.into(),
                    position_covariance: obj.position_covariance,
                };

                objects.push(object_data);
            }

            Ok(Objects {
                timestamp: objects_data.timestamp,
                is_new: objects_data.is_new != 0,
                is_tracked: objects_data.is_tracked != 0,
                objects,
            })
        }
    }

    /// Enable spatial mapping
    pub fn enable_spatial_mapping(&self, params: &SpatialMappingParameters) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut mapping_param = SL_SpatialMappingParameters {
                resolution_meter: params.resolution_meter,
                max_memory_usage: params.max_memory_usage,
                save_texture: params.save_texture,
                use_chunk_only: params.use_chunk_only,
                max_range_meter: params.max_range_meter,
                ..std::mem::zeroed()
            };

            let result = sl_enable_spatial_mapping(self.camera_id, &mut mapping_param);

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to enable spatial mapping: error code {}", result)));
            }
        }

        Ok(())
    }

    /// Disable spatial mapping
    pub fn disable_spatial_mapping(&self) {
        if self.is_opened {
            unsafe {
                sl_disable_spatial_mapping(self.camera_id);
            }
        }
    }

    /// Get the current spatial mapping state
    pub fn get_spatial_mapping_state(&self) -> Result<SpatialMappingState, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let state = sl_get_spatial_mapping_state(self.camera_id);
            Ok(state.into())
        }
    }

    /// Pause or resume spatial mapping
    pub fn pause_spatial_mapping(&self, pause: bool) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            sl_pause_spatial_mapping(self.camera_id, pause);
        }

        Ok(())
    }

    /// Request mesh generation (asynchronous)
    pub fn request_mesh_async(&self) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            sl_request_mesh_async(self.camera_id);
        }

        Ok(())
    }

    /// Get mesh request status
    pub fn get_mesh_request_status(&self) -> Result<i32, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            Ok(sl_get_mesh_request_status_async(self.camera_id))
        }
    }

    /// Update and retrieve the mesh
    pub fn update_mesh(&self) -> Result<(Vec<i32>, Vec<i32>, i32, Vec<i32>, i32, i32), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let max_submesh = 1000;
            let mut nb_vertices_per_submesh = vec![0i32; max_submesh];
            let mut nb_triangles_per_submesh = vec![0i32; max_submesh];
            let mut nb_submeshes = 0i32;
            let mut updated_indices = vec![0i32; max_submesh];
            let mut nb_vertices_tot = 0i32;
            let mut nb_triangles_tot = 0i32;

            let result = sl_update_mesh(
                self.camera_id,
                nb_vertices_per_submesh.as_mut_ptr(),
                nb_triangles_per_submesh.as_mut_ptr(),
                &mut nb_submeshes,
                updated_indices.as_mut_ptr(),
                &mut nb_vertices_tot,
                &mut nb_triangles_tot,
                max_submesh as i32,
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to update mesh: error code {}", result)));
            }

            Ok((
                nb_vertices_per_submesh,
                nb_triangles_per_submesh,
                nb_submeshes,
                updated_indices,
                nb_vertices_tot,
                nb_triangles_tot,
            ))
        }
    }

    /// Retrieve mesh data
    pub fn retrieve_mesh(&self, max_submeshes: usize) -> Result<MeshData, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        // First update the mesh to get the latest size information
        let (_, _, nb_submeshes, _, nb_vertices_tot, nb_triangles_tot) = self.update_mesh()?;

        let mut mesh_data = MeshData::new();
        mesh_data.vertices = vec![0.0f32; (nb_vertices_tot * 3) as usize];
        mesh_data.triangles = vec![0i32; (nb_triangles_tot * 3) as usize];
        mesh_data.colors = vec![0u8; (nb_vertices_tot * 4) as usize];
        mesh_data.uvs = vec![0.0f32; (nb_vertices_tot * 2) as usize];
        mesh_data.texture = vec![0u8; 1024 * 1024 * 3]; // Default texture size

        unsafe {
            let result = sl_retrieve_mesh(
                self.camera_id,
                mesh_data.vertices.as_mut_ptr(),
                mesh_data.triangles.as_mut_ptr(),
                mesh_data.colors.as_mut_ptr(),
                mesh_data.uvs.as_mut_ptr(),
                mesh_data.texture.as_mut_ptr(),
                max_submeshes as i32,
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to retrieve mesh: error code {}", result)));
            }
        }

        mesh_data.num_vertices = nb_vertices_tot as usize;
        mesh_data.num_triangles = nb_triangles_tot as usize;

        Ok(mesh_data)
    }

    /// Save mesh to file
    pub fn save_mesh(&self, filename: &str, format: MeshFileFormat) -> Result<bool, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        let filename_cstr = CString::new(filename)
            .map_err(|_| ZedError::Other("Invalid filename".to_string()))?;

        unsafe {
            let success = sl_save_mesh(self.camera_id, filename_cstr.as_ptr(), format.into());
            Ok(success)
        }
    }

    /// Save point cloud to file
    pub fn save_point_cloud_file(&self, filename: &str, format: MeshFileFormat) -> Result<bool, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        let filename_cstr = CString::new(filename)
            .map_err(|_| ZedError::Other("Invalid filename".to_string()))?;

        unsafe {
            let success = sl_save_point_cloud(self.camera_id, filename_cstr.as_ptr(), format.into());
            Ok(success)
        }
    }

    /// Filter mesh using specified filter level
    pub fn filter_mesh(&self, filter: MeshFilter, max_submeshes: usize) -> Result<(Vec<i32>, Vec<i32>, i32, Vec<i32>, i32, i32), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut nb_vertices_per_submesh = vec![0i32; max_submeshes];
            let mut nb_triangles_per_submesh = vec![0i32; max_submeshes];
            let mut nb_updated_submeshes = 0i32;
            let mut updated_indices = vec![0i32; max_submeshes];
            let mut nb_vertices_tot = 0i32;
            let mut nb_triangles_tot = 0i32;

            let success = sl_filter_mesh(
                self.camera_id,
                filter.into(),
                nb_vertices_per_submesh.as_mut_ptr(),
                nb_triangles_per_submesh.as_mut_ptr(),
                &mut nb_updated_submeshes,
                updated_indices.as_mut_ptr(),
                &mut nb_vertices_tot,
                &mut nb_triangles_tot,
                max_submeshes as i32,
            );

            if !success {
                return Err(ZedError::Other("Failed to filter mesh".to_string()));
            }

            Ok((
                nb_vertices_per_submesh,
                nb_triangles_per_submesh,
                nb_updated_submeshes,
                updated_indices,
                nb_vertices_tot,
                nb_triangles_tot,
            ))
        }
    }

    /// Get gravity estimation from spatial mapping
    pub fn spatial_mapping_get_gravity_estimation(&self) -> Result<Vector3, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut gravity = SL_Vector3 { x: 0.0, y: 0.0, z: 0.0 };
            sl_spatial_mapping_get_gravity_estimation(self.camera_id, &mut gravity);
            Ok(gravity.into())
        }
    }

    /// Extract whole spatial map
    pub fn extract_whole_spatial_map(&self) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let result = sl_extract_whole_spatial_map(self.camera_id);
            if result != 0 {
                return Err(ZedError::Other(format!("Failed to extract whole spatial map: error code {}", result)));
            }
        }

        Ok(())
    }

    /// Update fused point cloud
    pub fn update_fused_point_cloud(&self) -> Result<i32, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut nb_vertices_tot = 0i32;
            let result = sl_update_fused_point_cloud(self.camera_id, &mut nb_vertices_tot);
            
            if result != 0 {
                return Err(ZedError::Other(format!("Failed to update fused point cloud: error code {}", result)));
            }

            Ok(nb_vertices_tot)
        }
    }

    /// Retrieve fused point cloud
    pub fn retrieve_fused_point_cloud(&self) -> Result<Vec<f32>, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        // First update to get the size
        let nb_vertices = self.update_fused_point_cloud()?;
        
        unsafe {
            let mut vertices = vec![0.0f32; (nb_vertices * 4) as usize]; // 4 floats per vertex (XYZW)
            
            let result = sl_retrieve_fused_point_cloud(self.camera_id, vertices.as_mut_ptr());
            
            if result != 0 {
                return Err(ZedError::Other(format!("Failed to retrieve fused point cloud: error code {}", result)));
            }

            Ok(vertices)
        }
    }

    /// Load mesh from file
    pub fn load_mesh(&self, filename: &str, max_submeshes: usize) -> Result<(Vec<i32>, Vec<i32>, i32, Vec<i32>, i32, i32, Vec<i32>), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        let filename_cstr = CString::new(filename)
            .map_err(|_| ZedError::Other("Invalid filename".to_string()))?;

        unsafe {
            let mut nb_vertices_per_submesh = vec![0i32; max_submeshes];
            let mut nb_triangles_per_submesh = vec![0i32; max_submeshes];
            let mut nb_submeshes = 0i32;
            let mut updated_indices = vec![0i32; max_submeshes];
            let mut nb_vertices_tot = 0i32;
            let mut nb_triangles_tot = 0i32;
            let mut texture_sizes = vec![0i32; max_submeshes * 2]; // width, height pairs

            let success = sl_load_mesh(
                self.camera_id,
                filename_cstr.as_ptr(),
                nb_vertices_per_submesh.as_mut_ptr(),
                nb_triangles_per_submesh.as_mut_ptr(),
                &mut nb_submeshes,
                updated_indices.as_mut_ptr(),
                &mut nb_vertices_tot,
                &mut nb_triangles_tot,
                texture_sizes.as_mut_ptr(),
                max_submeshes as i32,
            );

            if !success {
                return Err(ZedError::Other("Failed to load mesh".to_string()));
            }

            Ok((
                nb_vertices_per_submesh,
                nb_triangles_per_submesh,
                nb_submeshes,
                updated_indices,
                nb_vertices_tot,
                nb_triangles_tot,
                texture_sizes,
            ))
        }
    }

    /// Apply texture to mesh
    pub fn apply_texture(&self, max_submeshes: usize) -> Result<(Vec<i32>, Vec<i32>, i32, Vec<i32>, i32, i32, Vec<i32>), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut nb_vertices_per_submesh = vec![0i32; max_submeshes];
            let mut nb_triangles_per_submesh = vec![0i32; max_submeshes];
            let mut nb_updated_submeshes = 0i32;
            let mut updated_indices = vec![0i32; max_submeshes];
            let mut nb_vertices_tot = 0i32;
            let mut nb_triangles_tot = 0i32;
            let mut texture_sizes = vec![0i32; max_submeshes * 2]; // width, height pairs

            let success = sl_apply_texture(
                self.camera_id,
                nb_vertices_per_submesh.as_mut_ptr(),
                nb_triangles_per_submesh.as_mut_ptr(),
                &mut nb_updated_submeshes,
                updated_indices.as_mut_ptr(),
                &mut nb_vertices_tot,
                &mut nb_triangles_tot,
                texture_sizes.as_mut_ptr(),
                max_submeshes as i32,
            );

            if !success {
                return Err(ZedError::Other("Failed to apply texture".to_string()));
            }

            Ok((
                nb_vertices_per_submesh,
                nb_triangles_per_submesh,
                nb_updated_submeshes,
                updated_indices,
                nb_vertices_tot,
                nb_triangles_tot,
                texture_sizes,
            ))
        }
    }

    /// Get sensor data (IMU, magnetometer, barometer, temperature)
    pub fn get_sensors_data(&self, time_reference: TimeReference) -> Result<SensorsData, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut sensors_data: SL_SensorsData = std::mem::zeroed();

            let result = sl_get_sensors_data(
                self.camera_id,
                &mut sensors_data,
                time_reference.into(),
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to get sensors data: error code {}", result)));
            }

            Ok(sensors_data.into())
        }
    }

    /// Get IMU orientation
    pub fn get_imu_orientation(&self, time_reference: TimeReference) -> Result<Quaternion, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut quaternion = SL_Quaternion { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };

            let result = sl_get_imu_orientation(
                self.camera_id,
                &mut quaternion,
                time_reference.into(),
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to get IMU orientation: error code {}", result)));
            }

            Ok(quaternion.into())
        }
    }

    /// Check if a camera setting is supported
    pub fn is_camera_setting_supported(&self, setting: VideoSettings) -> Result<bool, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            Ok(sl_is_camera_setting_supported(self.camera_id, setting.into()))
        }
    }

    /// Set camera setting value
    pub fn set_camera_setting(&self, setting: VideoSettings, value: i32) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let result = sl_set_camera_settings(self.camera_id, setting.into(), value);

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to set camera setting: error code {}", result)));
            }
        }

        Ok(())
    }

    /// Get camera setting value
    pub fn get_camera_setting(&self, setting: VideoSettings) -> Result<i32, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut value = 0i32;
            let result = sl_get_camera_settings(self.camera_id, setting.into(), &mut value);

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to get camera setting: error code {}", result)));
            }

            Ok(value)
        }
    }

    /// Get camera setting min/max values
    pub fn get_camera_setting_range(&self, setting: VideoSettings) -> Result<(i32, i32), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut min_value = 0i32;
            let mut max_value = 0i32;
            let result = sl_get_camera_settings_min_max(
                self.camera_id,
                setting.into(),
                &mut min_value,
                &mut max_value,
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to get camera setting range: error code {}", result)));
            }

            Ok((min_value, max_value))
        }
    }

    /// Set region of interest for auto exposure/gain
    pub fn set_roi_for_aec_agc(&self, side: Side, roi: &Rect, reset: bool) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut sl_roi: SL_Rect = (*roi).into();
            let result = sl_set_roi_for_aec_agc(
                self.camera_id,
                side.into(),
                &mut sl_roi,
                reset,
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to set ROI for AEC/AGC: error code {}", result)));
            }
        }

        Ok(())
    }

    /// Get region of interest for auto exposure/gain
    pub fn get_roi_for_aec_agc(&self, side: Side) -> Result<Rect, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut sl_roi: SL_Rect = std::mem::zeroed();
            let result = sl_get_roi_for_aec_agc(self.camera_id, side.into(), &mut sl_roi);

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to get ROI for AEC/AGC: error code {}", result)));
            }

            Ok(sl_roi.into())
        }
    }

    /// Get camera FPS
    pub fn get_camera_fps(&self) -> Result<f32, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            Ok(sl_get_camera_fps(self.camera_id))
        }
    }

    /// Get current FPS
    pub fn get_current_fps(&self) -> Result<f32, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            Ok(sl_get_current_fps(self.camera_id))
        }
    }

    /// Get camera firmware version
    pub fn get_camera_firmware(&self) -> Result<i32, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            Ok(sl_get_camera_firmware(self.camera_id))
        }
    }

    /// Get sensors firmware version
    pub fn get_sensors_firmware(&self) -> Result<i32, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            Ok(sl_get_sensors_firmware(self.camera_id))
        }
    }

    /// Get camera model
    pub fn get_camera_model(&self) -> Result<u32, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            Ok(sl_get_camera_model(self.camera_id) as u32)
        }
    }

    /// Save current image to file
    pub fn save_current_image(&self, view: ViewType, filename: &str) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        let filename_cstr = CString::new(filename)
            .map_err(|_| ZedError::Other("Invalid filename".to_string()))?;

        unsafe {
            let result = sl_save_current_image(self.camera_id, view.into(), filename_cstr.as_ptr());

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to save current image: error code {}", result)));
            }
        }

        Ok(())
    }

    /// Save current depth map to file
    pub fn save_current_depth(&self, side: Side, filename: &str) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        let filename_cstr = CString::new(filename)
            .map_err(|_| ZedError::Other("Invalid filename".to_string()))?;

        unsafe {
            let result = sl_save_current_depth(self.camera_id, side.into(), filename_cstr.as_ptr());

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to save current depth: error code {}", result)));
            }
        }

        Ok(())
    }

    /// Save current point cloud to file  
    pub fn save_current_point_cloud(&self, side: Side, filename: &str) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        let filename_cstr = CString::new(filename)
            .map_err(|_| ZedError::Other("Invalid filename".to_string()))?;

        unsafe {
            let result = sl_save_current_point_cloud(self.camera_id, side.into(), filename_cstr.as_ptr());

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to save current point cloud: error code {}", result)));
            }
        }

        Ok(())
    }

    /// Enable SVO recording
    pub fn enable_recording(&self, filename: &str, compression_mode: SvoCompressionMode, bitrate: u32, target_fps: i32, transcode: bool) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        let filename_cstr = CString::new(filename)
            .map_err(|_| ZedError::Other("Invalid filename".to_string()))?;

        unsafe {
            let result = sl_enable_recording(
                self.camera_id,
                filename_cstr.as_ptr(),
                compression_mode.into(),
                bitrate,
                target_fps,
                transcode,
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to enable recording: error code {}", result)));
            }
        }

        Ok(())
    }

    /// Disable SVO recording
    pub fn disable_recording(&self) {
        if self.is_opened {
            unsafe {
                sl_disable_recording(self.camera_id);
            }
        }
    }

    /// Get recording status
    pub fn get_recording_status(&self) -> Result<RecordingStatus, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let status_ptr = sl_get_recording_status(self.camera_id);
            if status_ptr.is_null() {
                return Err(ZedError::Other("Failed to get recording status".to_string()));
            }
            
            let status = *status_ptr;
            Ok(status.into())
        }
    }

    /// Pause or resume recording
    pub fn pause_recording(&self, pause: bool) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            sl_pause_recording(self.camera_id, pause);
        }

        Ok(())
    }

    /// Set SVO position for playback
    pub fn set_svo_position(&self, frame_number: i32) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            sl_set_svo_position(self.camera_id, frame_number);
        }

        Ok(())
    }

    /// Get current SVO position
    pub fn get_svo_position(&self) -> Result<i32, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            Ok(sl_get_svo_position(self.camera_id))
        }
    }

    /// Get SVO number of frames
    pub fn get_svo_number_of_frames(&self) -> Result<i32, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            Ok(sl_get_svo_number_of_frames(self.camera_id))
        }
    }

    /// Pause SVO reading
    pub fn pause_svo_reading(&self, pause: bool) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            sl_pause_svo_reading(self.camera_id, pause);
        }

        Ok(())
    }

    /// Enable body tracking
    pub fn enable_body_tracking(&self, params: &BodyTrackingParameters) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut tracking_param = SL_BodyTrackingParameters {
                instance_module_id: params.instance_module_id,
                enable_tracking: params.enable_tracking,
                enable_body_fitting: params.enable_body_fitting,
                body_format: params.body_format.into(),
                body_selection: params.body_selection.into(),
                max_range: params.max_range,
                prediction_timeout_s: params.prediction_timeout_s,
                allow_reduced_precision_inference: params.allow_reduced_precision_inference,
                ..std::mem::zeroed()
            };

            let result = sl_enable_body_tracking(self.camera_id, &mut tracking_param);

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to enable body tracking: error code {}", result)));
            }
        }

        Ok(())
    }

    /// Disable body tracking
    pub fn disable_body_tracking(&self, instance_id: u32, force_disable_all: bool) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            sl_disable_body_tracking(self.camera_id, instance_id, force_disable_all);
        }

        Ok(())
    }

    /// Retrieve detected bodies
    pub fn retrieve_bodies(&self, runtime_params: &BodyTrackingRuntimeParameters, instance_id: u32) -> Result<Bodies, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut bodies_data: SL_Bodies = std::mem::zeroed();
            
            let mut rt_param = SL_BodyTrackingRuntimeParameters {
                detection_confidence_threshold: runtime_params.detection_confidence_threshold,
                minimum_keypoints_threshold: runtime_params.minimum_keypoints_threshold,
                skeleton_smoothing: runtime_params.skeleton_smoothing,
                ..std::mem::zeroed()
            };

            let result = sl_retrieve_bodies(
                self.camera_id,
                &mut rt_param,
                &mut bodies_data,
                instance_id,
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to retrieve bodies: error code {}", result)));
            }

            // Convert C data to Rust structures
            let mut bodies = Vec::new();
            for i in 0..bodies_data.nb_bodies as usize {
                if i >= bodies_data.body_list.len() {
                    break;
                }

                let body = &bodies_data.body_list[i];
                
                // Convert bounding box 2D
                let mut bbox_2d = [Vector2::zero(); 4];
                for j in 0..4 {
                    bbox_2d[j] = body.bounding_box_2d[j].into();
                }

                // Convert bounding box 3D
                let mut bbox_3d = [Vector3::zero(); 8];
                for j in 0..8 {
                    bbox_3d[j] = body.bounding_box[j].into();
                }

                // Convert head bounding box 2D
                let mut head_bbox_2d = [Vector2::zero(); 4];
                for j in 0..4 {
                    head_bbox_2d[j] = body.head_bounding_box_2d[j].into();
                }

                // Convert head bounding box 3D
                let mut head_bbox_3d = [Vector3::zero(); 8];
                for j in 0..8 {
                    head_bbox_3d[j] = body.head_bounding_box[j].into();
                }

                // Convert keypoints (assuming max 70 keypoints as per Body70 format)
                let mut keypoint_3d = Vec::new();
                let mut keypoint_2d = Vec::new();
                let mut keypoint_confidence = Vec::new();
                for j in 0..70.min(body.keypoint.len()) {
                    keypoint_3d.push(body.keypoint[j].into());
                    keypoint_2d.push(body.keypoint_2d[j].into());
                    keypoint_confidence.push(body.keypoint_confidence[j]);
                }

                // Convert local position and orientation arrays
                let mut local_position_per_joint = Vec::new();
                let mut local_orientation_per_joint = Vec::new();
                for j in 0..70.min(body.local_position_per_joint.len()) {
                    local_position_per_joint.push(body.local_position_per_joint[j].into());
                    local_orientation_per_joint.push(body.local_orientation_per_joint[j].into());
                }

                let body_data = BodyData {
                    id: body.id,
                    unique_object_id: format!("body_{}", body.id),
                    tracking_state: body.tracking_state.into(),
                    action_state: body.action_state,
                    position: body.position.into(),
                    velocity: body.velocity.into(),
                    dimensions: body.dimensions.into(),
                    confidence: body.confidence,
                    keypoint_3d,
                    keypoint_2d,
                    keypoint_confidence,
                    local_position_per_joint,
                    local_orientation_per_joint,
                    global_root_orientation: body.global_root_orientation.into(),
                    bounding_box_2d: bbox_2d,
                    bounding_box: bbox_3d,
                    head_bounding_box_2d: head_bbox_2d,
                    head_bounding_box: head_bbox_3d,
                    head_position: body.head_position.into(),
                };

                bodies.push(body_data);
            }

            Ok(Bodies {
                timestamp: bodies_data.timestamp,
                is_new: bodies_data.is_new != 0,
                is_tracked: bodies_data.is_tracked != 0,
                bodies,
            })
        }
    }

    /// Enable streaming
    pub fn enable_streaming(&self, params: &StreamingParameters) -> Result<(), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let result = sl_enable_streaming(
                self.camera_id,
                params.codec.into(),
                params.bitrate,
                params.port,
                params.gop_size,
                if params.adaptative_bitrate { 1 } else { 0 },
                params.chunk_size,
                params.target_framerate,
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to enable streaming: error code {}", result)));
            }
        }

        Ok(())
    }

    /// Disable streaming
    pub fn disable_streaming(&self) {
        if self.is_opened {
            unsafe {
                sl_disable_streaming(self.camera_id);
            }
        }
    }

    /// Check if streaming is enabled
    pub fn is_streaming_enabled(&self) -> Result<bool, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            Ok(sl_is_streaming_enabled(self.camera_id) != 0)
        }
    }

    /// Get streaming parameters
    pub fn get_streaming_parameters(&self) -> Result<StreamingParameters, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let params_ptr = sl_get_streaming_parameters(self.camera_id);
            if params_ptr.is_null() {
                return Err(ZedError::Other("Failed to get streaming parameters".to_string()));
            }

            let params = *params_ptr;
            Ok(StreamingParameters {
                codec: match params.codec {
                    SL_STREAMING_CODEC_SL_STREAMING_CODEC_H265 => StreamingCodec::H265,
                    _ => StreamingCodec::H264,
                },
                port: params.port,
                bitrate: params.bitrate,
                gop_size: params.gop_size,
                adaptative_bitrate: params.adaptative_bitrate != 0,
                chunk_size: params.chunk_size,
                target_framerate: params.target_framerate,
            })
        }
    }

    /// Find floor plane
    pub fn find_floor_plane(&self, prior_rotation: Option<Quaternion>, prior_translation: Option<Vector3>) -> Result<(PlaneData, Quaternion, Vector3), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut reset_quaternion = SL_Quaternion { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };
            let mut reset_translation = SL_Vector3 { x: 0.0, y: 0.0, z: 0.0 };
            let prior_rot = prior_rotation.unwrap_or(Quaternion::identity()).into();
            let prior_trans = prior_translation.unwrap_or(Vector3::zero()).into();

            let plane_ptr = sl_find_floor_plane(
                self.camera_id,
                &mut reset_quaternion,
                &mut reset_translation,
                prior_rot,
                prior_trans,
            );

            if plane_ptr.is_null() {
                return Err(ZedError::Other("Failed to find floor plane".to_string()));
            }

            let plane_data = (*plane_ptr).into();
            Ok((plane_data, reset_quaternion.into(), reset_translation.into()))
        }
    }

    /// Find plane at hit point
    pub fn find_plane_at_hit(&self, pixel: Vector2, params: &PlaneDetectionParameters, check_area_threshold: bool) -> Result<PlaneData, ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut detection_params = SL_PlaneDetectionParameters {
                max_distance_threshold: params.max_distance_threshold,
                normal_similarity_threshold: params.normal_similarity_threshold,
            };

            let pixel_sl = SL_Vector2 { x: pixel.x, y: pixel.y };

            let plane_ptr = sl_find_plane_at_hit(
                self.camera_id,
                pixel_sl,
                &mut detection_params,
                check_area_threshold,
            );

            if plane_ptr.is_null() {
                return Err(ZedError::Other("Failed to find plane at hit point".to_string()));
            }

            Ok((*plane_ptr).into())
        }
    }

    /// Convert floor plane to mesh
    pub fn convert_floorplane_to_mesh(&self) -> Result<(Vec<f32>, Vec<i32>), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut nb_vertices_tot = 0i32;
            let mut nb_triangles_tot = 0i32;

            // First call to get sizes
            let result = sl_convert_floorplane_to_mesh(
                self.camera_id,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut nb_vertices_tot,
                &mut nb_triangles_tot,
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to get floor plane mesh size: error code {}", result)));
            }

            // Allocate buffers
            let mut vertices = vec![0.0f32; (nb_vertices_tot * 3) as usize];
            let mut triangles = vec![0i32; nb_triangles_tot as usize];

            // Second call to get data
            let result = sl_convert_floorplane_to_mesh(
                self.camera_id,
                vertices.as_mut_ptr(),
                triangles.as_mut_ptr(),
                &mut nb_vertices_tot,
                &mut nb_triangles_tot,
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to convert floor plane to mesh: error code {}", result)));
            }

            Ok((vertices, triangles))
        }
    }

    /// Convert hit plane to mesh
    pub fn convert_hitplane_to_mesh(&self) -> Result<(Vec<f32>, Vec<i32>), ZedError> {
        if !self.is_opened {
            return Err(ZedError::CameraNotOpened);
        }

        unsafe {
            let mut nb_vertices_tot = 0i32;
            let mut nb_triangles_tot = 0i32;

            // First call to get sizes
            let result = sl_convert_hitplane_to_mesh(
                self.camera_id,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut nb_vertices_tot,
                &mut nb_triangles_tot,
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to get hit plane mesh size: error code {}", result)));
            }

            // Allocate buffers
            let mut vertices = vec![0.0f32; (nb_vertices_tot * 3) as usize];
            let mut triangles = vec![0i32; nb_triangles_tot as usize];

            // Second call to get data
            let result = sl_convert_hitplane_to_mesh(
                self.camera_id,
                vertices.as_mut_ptr(),
                triangles.as_mut_ptr(),
                &mut nb_vertices_tot,
                &mut nb_triangles_tot,
            );

            if result != 0 {
                return Err(ZedError::Other(format!("Failed to convert hit plane to mesh: error code {}", result)));
            }

            Ok((vertices, triangles))
        }
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        if self.is_opened {
            unsafe {
                sl_close_camera(self.camera_id);
            }
        }
    }
}

/// Object detection model options
#[derive(Debug, Clone, Copy)]
pub enum ObjectDetectionModel {
    MultiClassBoxFast,
    MultiClassBoxMedium,
    MultiClassBoxAccurate,
    PersonHeadBoxFast,
    PersonHeadBoxAccurate,
    CustomBoxObjects,
    CustomYoloLikeBoxObjects,
}

impl From<ObjectDetectionModel> for u32 {
    fn from(model: ObjectDetectionModel) -> Self {
        match model {
            ObjectDetectionModel::MultiClassBoxFast => SL_OBJECT_DETECTION_MODEL_SL_OBJECT_DETECTION_MODEL_MULTI_CLASS_BOX_FAST,
            ObjectDetectionModel::MultiClassBoxMedium => SL_OBJECT_DETECTION_MODEL_SL_OBJECT_DETECTION_MODEL_MULTI_CLASS_BOX_MEDIUM,
            ObjectDetectionModel::MultiClassBoxAccurate => SL_OBJECT_DETECTION_MODEL_SL_OBJECT_DETECTION_MODEL_MULTI_CLASS_BOX_ACCURATE,
            ObjectDetectionModel::PersonHeadBoxFast => SL_OBJECT_DETECTION_MODEL_SL_OBJECT_DETECTION_MODEL_PERSON_HEAD_BOX_FAST,
            ObjectDetectionModel::PersonHeadBoxAccurate => SL_OBJECT_DETECTION_MODEL_SL_OBJECT_DETECTION_MODEL_PERSON_HEAD_BOX_ACCURATE,
            ObjectDetectionModel::CustomBoxObjects => SL_OBJECT_DETECTION_MODEL_SL_OBJECT_DETECTION_MODEL_CUSTOM_BOX_OBJECTS,
            ObjectDetectionModel::CustomYoloLikeBoxObjects => SL_OBJECT_DETECTION_MODEL_SL_OBJECT_DETECTION_MODEL_CUSTOM_YOLOLIKE_BOX_OBJECTS,
        }
    }
}

/// Object class types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ObjectClass {
    Person,
    Vehicle,
    Bag,
    Animal,
    Electronics,
    FruitVegetable,
    Sport,
}

impl From<u32> for ObjectClass {
    fn from(class: u32) -> Self {
        match class {
            SL_OBJECT_CLASS_SL_OBJECT_CLASS_PERSON => ObjectClass::Person,
            SL_OBJECT_CLASS_SL_OBJECT_CLASS_VEHICLE => ObjectClass::Vehicle,
            SL_OBJECT_CLASS_SL_OBJECT_CLASS_BAG => ObjectClass::Bag,
            SL_OBJECT_CLASS_SL_OBJECT_CLASS_ANIMAL => ObjectClass::Animal,
            SL_OBJECT_CLASS_SL_OBJECT_CLASS_ELECTRONICS => ObjectClass::Electronics,
            SL_OBJECT_CLASS_SL_OBJECT_CLASS_FRUIT_VEGETABLE => ObjectClass::FruitVegetable,
            SL_OBJECT_CLASS_SL_OBJECT_CLASS_SPORT => ObjectClass::Sport,
            _ => ObjectClass::Person, // Default fallback
        }
    }
}

impl From<ObjectClass> for u32 {
    fn from(class: ObjectClass) -> Self {
        match class {
            ObjectClass::Person => SL_OBJECT_CLASS_SL_OBJECT_CLASS_PERSON,
            ObjectClass::Vehicle => SL_OBJECT_CLASS_SL_OBJECT_CLASS_VEHICLE,
            ObjectClass::Bag => SL_OBJECT_CLASS_SL_OBJECT_CLASS_BAG,
            ObjectClass::Animal => SL_OBJECT_CLASS_SL_OBJECT_CLASS_ANIMAL,
            ObjectClass::Electronics => SL_OBJECT_CLASS_SL_OBJECT_CLASS_ELECTRONICS,
            ObjectClass::FruitVegetable => SL_OBJECT_CLASS_SL_OBJECT_CLASS_FRUIT_VEGETABLE,
            ObjectClass::Sport => SL_OBJECT_CLASS_SL_OBJECT_CLASS_SPORT,
        }
    }
}

/// Object tracking state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ObjectTrackingState {
    Off,
    Ok,
    Searching,
    Terminate,
}

impl From<u32> for ObjectTrackingState {
    fn from(state: u32) -> Self {
        match state {
            SL_OBJECT_TRACKING_STATE_SL_OBJECT_TRACKING_STATE_OFF => ObjectTrackingState::Off,
            SL_OBJECT_TRACKING_STATE_SL_OBJECT_TRACKING_STATE_OK => ObjectTrackingState::Ok,
            SL_OBJECT_TRACKING_STATE_SL_OBJECT_TRACKING_STATE_SEARCHING => ObjectTrackingState::Searching,
            SL_OBJECT_TRACKING_STATE_SL_OBJECT_TRACKING_STATE_TERMINATE => ObjectTrackingState::Terminate,
            _ => ObjectTrackingState::Off, // Default fallback
        }
    }
}

/// 2D vector representation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector2 {
    pub x: f32,
    pub y: f32,
}

impl Vector2 {
    /// Create a new Vector2
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Create a zero vector
    pub fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
}

impl From<SL_Vector2> for Vector2 {
    fn from(v: SL_Vector2) -> Self {
        Vector2 { x: v.x, y: v.y }
    }
}

/// Detected object data
#[derive(Debug, Clone)]
pub struct ObjectData {
    pub id: i32,
    pub unique_object_id: String,
    pub raw_label: i32,
    pub label: ObjectClass,
    pub sublabel: i32,
    pub confidence: f32,
    pub tracking_state: ObjectTrackingState,
    pub position: Vector3,
    pub velocity: Vector3,
    pub dimensions: Vector3,
    pub bounding_box_2d: [Vector2; 4],
    pub bounding_box: [Vector3; 8],
    pub head_bounding_box_2d: [Vector2; 4],
    pub head_bounding_box: [Vector3; 8],
    pub head_position: Vector3,
    pub position_covariance: [f32; 6],
}

impl ObjectData {
    /// Check if the object is currently being tracked
    pub fn is_tracked(&self) -> bool {
        self.tracking_state == ObjectTrackingState::Ok
    }

    /// Check if the object is valid (not terminated)
    pub fn is_valid(&self) -> bool {
        self.tracking_state != ObjectTrackingState::Terminate
    }

    /// Get the object class as a string
    pub fn class_name(&self) -> &'static str {
        match self.label {
            ObjectClass::Person => "Person",
            ObjectClass::Vehicle => "Vehicle",
            ObjectClass::Bag => "Bag",
            ObjectClass::Animal => "Animal",
            ObjectClass::Electronics => "Electronics",
            ObjectClass::FruitVegetable => "Fruit/Vegetable",
            ObjectClass::Sport => "Sport",
        }
    }

    /// Get the 2D bounding box center point
    pub fn get_2d_center(&self) -> Vector2 {
        let sum_x = self.bounding_box_2d.iter().map(|p| p.x).sum::<f32>();
        let sum_y = self.bounding_box_2d.iter().map(|p| p.y).sum::<f32>();
        Vector2::new(sum_x / 4.0, sum_y / 4.0)
    }

    /// Get the 2D bounding box width and height
    pub fn get_2d_size(&self) -> (f32, f32) {
        let min_x = self.bounding_box_2d.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
        let max_x = self.bounding_box_2d.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
        let min_y = self.bounding_box_2d.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
        let max_y = self.bounding_box_2d.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);
        (max_x - min_x, max_y - min_y)
    }
}

/// Collection of detected objects
#[derive(Debug, Clone)]
pub struct Objects {
    pub timestamp: u64,
    pub is_new: bool,
    pub is_tracked: bool,
    pub objects: Vec<ObjectData>,
}

impl Objects {
    /// Get the number of detected objects
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Check if any objects were detected
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// Get objects of a specific class
    pub fn get_objects_by_class(&self, class: ObjectClass) -> Vec<&ObjectData> {
        self.objects.iter().filter(|obj| obj.label == class).collect()
    }

    /// Get only tracked objects
    pub fn get_tracked_objects(&self) -> Vec<&ObjectData> {
        self.objects.iter().filter(|obj| obj.is_tracked()).collect()
    }

    /// Get object by ID
    pub fn get_object_by_id(&self, id: i32) -> Option<&ObjectData> {
        self.objects.iter().find(|obj| obj.id == id)
    }
}

/// Object detection parameters
#[derive(Debug, Clone)]
pub struct ObjectDetectionParameters {
    pub instance_module_id: u32,
    pub enable_tracking: bool,
    pub enable_segmentation: bool,
    pub detection_model: ObjectDetectionModel,
    pub max_range: f32,
    pub filtering_mode: ObjectFilteringMode,
    pub prediction_timeout_s: f32,
    pub allow_reduced_precision_inference: bool,
}

/// Object filtering mode
#[derive(Debug, Clone, Copy)]
pub enum ObjectFilteringMode {
    None,
    Nms3D,
    Nms3DPerClass,
}

impl From<ObjectFilteringMode> for u32 {
    fn from(mode: ObjectFilteringMode) -> Self {
        match mode {
            ObjectFilteringMode::None => SL_OBJECT_FILTERING_MODE_SL_OBJECT_FILTERING_MODE_NONE,
            ObjectFilteringMode::Nms3D => SL_OBJECT_FILTERING_MODE_SL_OBJECT_FILTERING_MODE_NMS_3D,
            ObjectFilteringMode::Nms3DPerClass => SL_OBJECT_FILTERING_MODE_SL_OBJECT_FILTERING_MODE_NMS_3D_PER_CLASS,
        }
    }
}

impl Default for ObjectDetectionParameters {
    fn default() -> Self {
        Self {
            instance_module_id: 0,
            enable_tracking: true,
            enable_segmentation: false,
            detection_model: ObjectDetectionModel::MultiClassBoxMedium,
            max_range: 40.0,
            filtering_mode: ObjectFilteringMode::Nms3D,
            prediction_timeout_s: 0.2,
            allow_reduced_precision_inference: false,
        }
    }
}

impl ObjectDetectionParameters {
    /// Create new object detection parameters with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the instance module ID
    pub fn with_instance_id(mut self, id: u32) -> Self {
        self.instance_module_id = id;
        self
    }

    /// Enable or disable object tracking
    pub fn with_tracking(mut self, enable: bool) -> Self {
        self.enable_tracking = enable;
        self
    }

    /// Enable or disable segmentation
    pub fn with_segmentation(mut self, enable: bool) -> Self {
        self.enable_segmentation = enable;
        self
    }

    /// Set the detection model
    pub fn with_detection_model(mut self, model: ObjectDetectionModel) -> Self {
        self.detection_model = model;
        self
    }

    /// Set the maximum detection range
    pub fn with_max_range(mut self, range: f32) -> Self {
        self.max_range = range;
        self
    }

    /// Set the filtering mode
    pub fn with_filtering_mode(mut self, mode: ObjectFilteringMode) -> Self {
        self.filtering_mode = mode;
        self
    }

    /// Set the prediction timeout
    pub fn with_prediction_timeout(mut self, timeout: f32) -> Self {
        self.prediction_timeout_s = timeout;
        self
    }
}

/// Runtime parameters for object detection
#[derive(Debug, Clone)]
pub struct ObjectDetectionRuntimeParameters {
    pub detection_confidence_threshold: f32,
    pub object_class_filter: Vec<ObjectClass>,
    pub object_confidence_threshold: Vec<i32>,
}

impl Default for ObjectDetectionRuntimeParameters {
    fn default() -> Self {
        Self {
            detection_confidence_threshold: 20.0,
            object_class_filter: vec![],
            object_confidence_threshold: vec![],
        }
    }
}

impl ObjectDetectionRuntimeParameters {
    /// Create new runtime parameters with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the detection confidence threshold
    pub fn with_detection_confidence_threshold(mut self, threshold: f32) -> Self {
        self.detection_confidence_threshold = threshold;
        self
    }

    /// Set object class filter
    pub fn with_object_class_filter(mut self, classes: Vec<ObjectClass>) -> Self {
        self.object_class_filter = classes;
        self
    }

    /// Set object confidence thresholds
    pub fn with_object_confidence_threshold(mut self, thresholds: Vec<i32>) -> Self {
        self.object_confidence_threshold = thresholds;
        self
    }
}

/// Spatial mapping resolution presets
#[derive(Debug, Clone, Copy)]
pub enum SpatialMappingResolution {
    High,   // 0.05m
    Medium, // 0.08m  
    Low,    // 0.15m
}

impl From<SpatialMappingResolution> for f32 {
    fn from(res: SpatialMappingResolution) -> Self {
        match res {
            SpatialMappingResolution::High => 0.05,
            SpatialMappingResolution::Medium => 0.08,
            SpatialMappingResolution::Low => 0.15,
        }
    }
}

/// Spatial mapping filtering modes
#[derive(Debug, Clone, Copy)]
pub enum MeshFilter {
    Low,
    Medium, 
    High,
}

impl From<MeshFilter> for u32 {
    fn from(filter: MeshFilter) -> Self {
        match filter {
            MeshFilter::Low => SL_MESH_FILTER_SL_MESH_FILTER_LOW,
            MeshFilter::Medium => SL_MESH_FILTER_SL_MESH_FILTER_MEDIUM,
            MeshFilter::High => SL_MESH_FILTER_SL_MESH_FILTER_HIGH,
        }
    }
}

/// Spatial mapping state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpatialMappingState {
    Initializing,
    Ok,
    NotEnabled,
    FpsLow,
}

impl From<u32> for SpatialMappingState {
    fn from(state: u32) -> Self {
        match state {
            SL_SPATIAL_MAPPING_STATE_SL_SPATIAL_MAPPING_STATE_INITIALIZING => SpatialMappingState::Initializing,
            SL_SPATIAL_MAPPING_STATE_SL_SPATIAL_MAPPING_STATE_OK => SpatialMappingState::Ok,
            SL_SPATIAL_MAPPING_STATE_SL_SPATIAL_MAPPING_STATE_NOT_ENABLED => SpatialMappingState::NotEnabled,
            SL_SPATIAL_MAPPING_STATE_SL_SPATIAL_MAPPING_STATE_FPS_LOW => SpatialMappingState::FpsLow,
            _ => SpatialMappingState::NotEnabled,
        }
    }
}

/// Parameters for spatial mapping
#[derive(Debug, Clone)]
pub struct SpatialMappingParameters {
    pub resolution_meter: f32,
    pub max_memory_usage: i32,
    pub save_texture: bool,
    pub use_chunk_only: bool,
    pub max_range_meter: f32,
}

impl Default for SpatialMappingParameters {
    fn default() -> Self {
        Self {
            resolution_meter: 0.05,
            max_memory_usage: 2048,
            save_texture: false,
            use_chunk_only: false,
            max_range_meter: 3.5,
        }
    }
}

impl SpatialMappingParameters {
    /// Create new spatial mapping parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the resolution in meters
    pub fn with_resolution_meter(mut self, resolution: f32) -> Self {
        self.resolution_meter = resolution;
        self
    }

    /// Set the resolution using preset
    pub fn with_resolution_preset(mut self, preset: SpatialMappingResolution) -> Self {
        self.resolution_meter = preset.into();
        self
    }

    /// Set maximum memory usage in MB
    pub fn with_max_memory_usage(mut self, memory_mb: i32) -> Self {
        self.max_memory_usage = memory_mb;
        self
    }

    /// Enable or disable texture saving
    pub fn with_save_texture(mut self, save: bool) -> Self {
        self.save_texture = save;
        self
    }

    /// Use chunk-only mode
    pub fn with_chunk_only(mut self, chunk_only: bool) -> Self {
        self.use_chunk_only = chunk_only;
        self
    }

    /// Set maximum range in meters
    pub fn with_max_range_meter(mut self, range: f32) -> Self {
        self.max_range_meter = range;
        self
    }
}

/// Mesh data container
#[derive(Debug)]
pub struct MeshData {
    pub vertices: Vec<f32>,
    pub triangles: Vec<i32>,
    pub colors: Vec<u8>,
    pub uvs: Vec<f32>,
    pub texture: Vec<u8>,
    pub num_vertices: usize,
    pub num_triangles: usize,
}

impl MeshData {
    /// Create a new empty mesh
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            triangles: Vec::new(),
            colors: Vec::new(),
            uvs: Vec::new(),
            texture: Vec::new(),
            num_vertices: 0,
            num_triangles: 0,
        }
    }
}

/// Camera video settings
#[derive(Debug, Clone, Copy)]
pub enum VideoSettings {
    Brightness,
    Contrast,
    Hue,
    Saturation,
    Sharpness,
    Gamma,
    Gain,
    Exposure,
    AecAgc,
    WhitebalanceTemperature,
    WhitebalanceAuto,
    LedStatus,
    ExposureTime,
    AnalogGain,
    DigitalGain,
}

impl From<VideoSettings> for u32 {
    fn from(setting: VideoSettings) -> Self {
        match setting {
            VideoSettings::Brightness => SL_VIDEO_SETTINGS_SL_VIDEO_SETTINGS_BRIGHTNESS,
            VideoSettings::Contrast => SL_VIDEO_SETTINGS_SL_VIDEO_SETTINGS_CONTRAST,
            VideoSettings::Hue => SL_VIDEO_SETTINGS_SL_VIDEO_SETTINGS_HUE,
            VideoSettings::Saturation => SL_VIDEO_SETTINGS_SL_VIDEO_SETTINGS_SATURATION,
            VideoSettings::Sharpness => SL_VIDEO_SETTINGS_SL_VIDEO_SETTINGS_SHARPNESS,
            VideoSettings::Gamma => SL_VIDEO_SETTINGS_SL_VIDEO_SETTINGS_GAMMA,
            VideoSettings::Gain => SL_VIDEO_SETTINGS_SL_VIDEO_SETTINGS_GAIN,
            VideoSettings::Exposure => SL_VIDEO_SETTINGS_SL_VIDEO_SETTINGS_EXPOSURE,
            VideoSettings::AecAgc => SL_VIDEO_SETTINGS_SL_VIDEO_SETTINGS_AEC_AGC,
            VideoSettings::WhitebalanceTemperature => SL_VIDEO_SETTINGS_SL_VIDEO_SETTINGS_WHITEBALANCE_TEMPERATURE,
            VideoSettings::WhitebalanceAuto => SL_VIDEO_SETTINGS_SL_VIDEO_SETTINGS_WHITEBALANCE_AUTO,
            VideoSettings::LedStatus => SL_VIDEO_SETTINGS_SL_VIDEO_SETTINGS_LED_STATUS,
            VideoSettings::ExposureTime => SL_VIDEO_SETTINGS_SL_VIDEO_SETTINGS_EXPOSURE_TIME,
            VideoSettings::AnalogGain => SL_VIDEO_SETTINGS_SL_VIDEO_SETTINGS_ANALOG_GAIN,
            VideoSettings::DigitalGain => SL_VIDEO_SETTINGS_SL_VIDEO_SETTINGS_DIGITAL_GAIN,
        }
    }
}

/// Side enumeration for stereo cameras
#[derive(Debug, Clone, Copy)]
pub enum Side {
    Left,
    Right,
    Both,
}

impl From<Side> for u32 {
    fn from(side: Side) -> Self {
        match side {
            Side::Left => SL_SIDE_SL_SIDE_LEFT,
            Side::Right => SL_SIDE_SL_SIDE_RIGHT, 
            Side::Both => SL_SIDE_SL_SIDE_BOTH,
        }
    }
}

/// Time reference for sensor data
#[derive(Debug, Clone, Copy)]
pub enum TimeReference {
    Image,
    Current,
}

impl From<TimeReference> for u32 {
    fn from(time_ref: TimeReference) -> Self {
        match time_ref {
            TimeReference::Image => SL_TIME_REFERENCE_SL_TIME_REFERENCE_IMAGE,
            TimeReference::Current => SL_TIME_REFERENCE_SL_TIME_REFERENCE_CURRENT,
        }
    }
}

/// IMU sensor data
#[derive(Debug, Clone)]
pub struct ImuData {
    pub is_available: bool,
    pub timestamp_ns: u64,
    pub angular_velocity: Vector3,
    pub linear_acceleration: Vector3,
    pub angular_velocity_uncalibrated: Vector3,
    pub linear_acceleration_uncalibrated: Vector3,
    pub orientation: Quaternion,
}

impl From<SL_IMUData> for ImuData {
    fn from(imu: SL_IMUData) -> Self {
        ImuData {
            is_available: imu.is_available,
            timestamp_ns: imu.timestamp_ns,
            angular_velocity: imu.angular_velocity.into(),
            linear_acceleration: imu.linear_acceleration.into(),
            angular_velocity_uncalibrated: imu.angular_velocity_unc.into(),
            linear_acceleration_uncalibrated: imu.linear_acceleration_unc.into(),
            orientation: imu.orientation.into(),
        }
    }
}

/// Barometer sensor data
#[derive(Debug, Clone)]
pub struct BarometerData {
    pub is_available: bool,
    pub timestamp_ns: u64,
    pub pressure: f32,
    pub relative_altitude: f32,
}

impl From<SL_BarometerData> for BarometerData {
    fn from(baro: SL_BarometerData) -> Self {
        BarometerData {
            is_available: baro.is_available,
            timestamp_ns: baro.timestamp_ns,
            pressure: baro.pressure,
            relative_altitude: baro.relative_altitude,
        }
    }
}

/// Magnetometer sensor data  
#[derive(Debug, Clone)]
pub struct MagnetometerData {
    pub is_available: bool,
    pub timestamp_ns: u64,
    pub magnetic_field_calibrated: Vector3,
    pub magnetic_field_uncalibrated: Vector3,
    pub magnetic_heading: f32,
    pub magnetic_heading_accuracy: f32,
}

impl From<SL_MagnetometerData> for MagnetometerData {
    fn from(mag: SL_MagnetometerData) -> Self {
        MagnetometerData {
            is_available: mag.is_available,
            timestamp_ns: mag.timestamp_ns,
            magnetic_field_calibrated: mag.magnetic_field_c.into(),
            magnetic_field_uncalibrated: mag.magnetic_field_unc.into(),
            magnetic_heading: mag.magnetic_heading,
            magnetic_heading_accuracy: mag.magnetic_heading_accuracy,
        }
    }
}

/// Temperature sensor data
#[derive(Debug, Clone)]
pub struct TemperatureData {
    pub imu_temp: f32,
    pub barometer_temp: f32,
    pub onboard_left_temp: f32,
    pub onboard_right_temp: f32,
}

impl From<SL_TemperatureData> for TemperatureData {
    fn from(temp: SL_TemperatureData) -> Self {
        TemperatureData {
            imu_temp: temp.imu_temp,
            barometer_temp: temp.barometer_temp,
            onboard_left_temp: temp.onboard_left_temp,
            onboard_right_temp: temp.onboard_right_temp,
        }
    }
}

/// Combined sensor data
#[derive(Debug, Clone)]
pub struct SensorsData {
    pub imu: ImuData,
    pub barometer: BarometerData,
    pub magnetometer: MagnetometerData,
    pub temperature: TemperatureData,
    pub camera_moving_state: i32,
    pub image_sync_trigger: i32,
}

impl From<SL_SensorsData> for SensorsData {
    fn from(sensors: SL_SensorsData) -> Self {
        SensorsData {
            imu: sensors.imu.into(),
            barometer: sensors.barometer.into(),
            magnetometer: sensors.magnetometer.into(),
            temperature: sensors.temperature.into(),
            camera_moving_state: sensors.camera_moving_state,
            image_sync_trigger: sensors.image_sync_trigger,
        }
    }
}

/// Rectangle structure for ROI
#[derive(Debug, Clone)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl From<Rect> for SL_Rect {
    fn from(rect: Rect) -> Self {
        SL_Rect {
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
        }
    }
}

impl From<SL_Rect> for Rect {
    fn from(rect: SL_Rect) -> Self {
        Rect {
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
        }
    }
}

/// Mesh file format options
#[derive(Debug, Clone, Copy)]
pub enum MeshFileFormat {
    Ply,
    PlyBinary,
    Obj,
}

impl From<MeshFileFormat> for u32 {
    fn from(format: MeshFileFormat) -> Self {
        match format {
            MeshFileFormat::Ply => SL_MESH_FILE_FORMAT_SL_MESH_FILE_FORMAT_PLY,
            MeshFileFormat::PlyBinary => SL_MESH_FILE_FORMAT_SL_MESH_FILE_FORMAT_PLY_BIN,
            MeshFileFormat::Obj => SL_MESH_FILE_FORMAT_SL_MESH_FILE_FORMAT_OBJ,
        }
    }
}

/// SVO compression modes
#[derive(Debug, Clone, Copy)]
pub enum SvoCompressionMode {
    Lossless,
    H264,
    H265,
}

impl From<SvoCompressionMode> for u32 {
    fn from(mode: SvoCompressionMode) -> Self {
        match mode {
            SvoCompressionMode::Lossless => SL_SVO_COMPRESSION_MODE_SL_SVO_COMPRESSION_MODE_LOSSLESS,
            SvoCompressionMode::H264 => SL_SVO_COMPRESSION_MODE_SL_SVO_COMPRESSION_MODE_H264,
            SvoCompressionMode::H265 => SL_SVO_COMPRESSION_MODE_SL_SVO_COMPRESSION_MODE_H265,
        }
    }
}

/// Streaming codec options
#[derive(Debug, Clone, Copy)]
pub enum StreamingCodec {
    H264,
    H265,
}

impl From<StreamingCodec> for u32 {
    fn from(codec: StreamingCodec) -> Self {
        match codec {
            StreamingCodec::H264 => SL_STREAMING_CODEC_SL_STREAMING_CODEC_H264,
            StreamingCodec::H265 => SL_STREAMING_CODEC_SL_STREAMING_CODEC_H265,
        }
    }
}

/// Streaming parameters
#[derive(Debug, Clone)]
pub struct StreamingParameters {
    pub codec: StreamingCodec,
    pub port: u16,
    pub bitrate: u32,
    pub gop_size: i32,
    pub adaptative_bitrate: bool,
    pub chunk_size: i32,
    pub target_framerate: i32,
}

impl Default for StreamingParameters {
    fn default() -> Self {
        Self {
            codec: StreamingCodec::H264,
            port: 30000,
            bitrate: 8000,
            gop_size: -1,
            adaptative_bitrate: false,
            chunk_size: 32768,
            target_framerate: 0,
        }
    }
}

impl StreamingParameters {
    /// Create new streaming parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the codec
    pub fn with_codec(mut self, codec: StreamingCodec) -> Self {
        self.codec = codec;
        self
    }

    /// Set the port
    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Set the bitrate in Kbits/s
    pub fn with_bitrate(mut self, bitrate: u32) -> Self {
        self.bitrate = bitrate;
        self
    }

    /// Set the GOP size
    pub fn with_gop_size(mut self, gop_size: i32) -> Self {
        self.gop_size = gop_size;
        self
    }

    /// Enable or disable adaptive bitrate
    pub fn with_adaptive_bitrate(mut self, enable: bool) -> Self {
        self.adaptative_bitrate = enable;
        self
    }

    /// Set the chunk size
    pub fn with_chunk_size(mut self, chunk_size: i32) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Set the target framerate
    pub fn with_target_framerate(mut self, framerate: i32) -> Self {
        self.target_framerate = framerate;
        self
    }
}

/// Recording status information
#[derive(Debug, Clone)]
pub struct RecordingStatus {
    pub is_recording: bool,
    pub is_paused: bool,
    pub status: bool,
    pub current_compression_time: f64,
    pub current_compression_ratio: f64,
    pub average_compression_time: f64,
    pub average_compression_ratio: f64,
}

impl From<SL_RecordingStatus> for RecordingStatus {
    fn from(status: SL_RecordingStatus) -> Self {
        RecordingStatus {
            is_recording: status.is_recording,
            is_paused: status.is_paused,
            status: status.status,
            current_compression_time: status.current_compression_time,
            current_compression_ratio: status.current_compression_ratio,
            average_compression_time: status.average_compression_time,
            average_compression_ratio: status.average_compression_ratio,
        }
    }
}

/// Coordinate system options
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoordinateSystem {
    Image,
    LeftHandedYUp,
    RightHandedYUp,
    RightHandedZUp,
    LeftHandedZUp,
    RightHandedZUpXFwd,
}

impl From<CoordinateSystem> for u32 {
    fn from(coord: CoordinateSystem) -> Self {
        match coord {
            CoordinateSystem::Image => SL_COORDINATE_SYSTEM_SL_COORDINATE_SYSTEM_IMAGE,
            CoordinateSystem::LeftHandedYUp => SL_COORDINATE_SYSTEM_SL_COORDINATE_SYSTEM_LEFT_HANDED_Y_UP,
            CoordinateSystem::RightHandedYUp => SL_COORDINATE_SYSTEM_SL_COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP,
            CoordinateSystem::RightHandedZUp => SL_COORDINATE_SYSTEM_SL_COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP,
            CoordinateSystem::LeftHandedZUp => SL_COORDINATE_SYSTEM_SL_COORDINATE_SYSTEM_LEFT_HANDED_Z_UP,
            CoordinateSystem::RightHandedZUpXFwd => SL_COORDINATE_SYSTEM_SL_COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP_X_FWD,
        }
    }
}

impl From<u32> for CoordinateSystem {
    fn from(coord: u32) -> Self {
        match coord {
            SL_COORDINATE_SYSTEM_SL_COORDINATE_SYSTEM_IMAGE => CoordinateSystem::Image,
            SL_COORDINATE_SYSTEM_SL_COORDINATE_SYSTEM_LEFT_HANDED_Y_UP => CoordinateSystem::LeftHandedYUp,
            SL_COORDINATE_SYSTEM_SL_COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP => CoordinateSystem::RightHandedYUp,
            SL_COORDINATE_SYSTEM_SL_COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP => CoordinateSystem::RightHandedZUp,
            SL_COORDINATE_SYSTEM_SL_COORDINATE_SYSTEM_LEFT_HANDED_Z_UP => CoordinateSystem::LeftHandedZUp,
            SL_COORDINATE_SYSTEM_SL_COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP_X_FWD => CoordinateSystem::RightHandedZUpXFwd,
            _ => CoordinateSystem::LeftHandedYUp, // Default
        }
    }
}

/// Health status of the camera
#[derive(Debug, Clone)]
pub struct HealthStatus {
    /// Status of the camera's image module
    pub image_status: i32,
    /// Status of the camera's depth module
    pub depth_status: i32,
    /// Status of the camera's sensors (IMU, etc.)
    pub sensors_status: i32,
}

impl From<SL_HealthStatus> for HealthStatus {
    fn from(status: SL_HealthStatus) -> Self {
        HealthStatus {
            image_status: status.image_status,
            depth_status: status.depth_status,
            sensors_status: status.sensors_status,
        }
    }
}

impl HealthStatus {
    /// Check if all systems are healthy
    pub fn is_healthy(&self) -> bool {
        self.image_status == 0 && self.depth_status == 0 && self.sensors_status == 0
    }

    /// Get a human-readable status description
    pub fn get_status_description(&self) -> String {
        let mut status = Vec::new();
        
        if self.image_status != 0 {
            status.push(format!("Image module error: {}", self.image_status));
        }
        if self.depth_status != 0 {
            status.push(format!("Depth module error: {}", self.depth_status));
        }
        if self.sensors_status != 0 {
            status.push(format!("Sensors module error: {}", self.sensors_status));
        }
        
        if status.is_empty() {
            "All systems healthy".to_string()
        } else {
            status.join(", ")
        }
    }
}

/// Body tracking detection models
#[derive(Debug, Clone, Copy)]
pub enum BodyTrackingModel {
    HumanBodyFast,
    HumanBodyMedium,
    HumanBodyAccurate,
}

impl From<BodyTrackingModel> for u32 {
    fn from(model: BodyTrackingModel) -> Self {
        match model {
            BodyTrackingModel::HumanBodyFast => SL_BODY_TRACKING_MODEL_SL_BODY_TRACKING_MODEL_HUMAN_BODY_FAST,
            BodyTrackingModel::HumanBodyMedium => SL_BODY_TRACKING_MODEL_SL_BODY_TRACKING_MODEL_HUMAN_BODY_MEDIUM,
            BodyTrackingModel::HumanBodyAccurate => SL_BODY_TRACKING_MODEL_SL_BODY_TRACKING_MODEL_HUMAN_BODY_ACCURATE,
        }
    }
}

/// Body format types
#[derive(Debug, Clone, Copy)]
pub enum BodyFormat {
    Body18,
    Body34,
    Body38,
    Body70,
}

impl From<u32> for BodyFormat {
    fn from(format: u32) -> Self {
        match format {
            SL_BODY_FORMAT_SL_BODY_FORMAT_BODY_18 => BodyFormat::Body18,
            SL_BODY_FORMAT_SL_BODY_FORMAT_BODY_34 => BodyFormat::Body34,
            SL_BODY_FORMAT_SL_BODY_FORMAT_BODY_38 => BodyFormat::Body38,
            SL_BODY_FORMAT_SL_BODY_FORMAT_BODY_70 => BodyFormat::Body70,
            _ => BodyFormat::Body18,
        }
    }
}

impl From<BodyFormat> for u32 {
    fn from(format: BodyFormat) -> Self {
        match format {
            BodyFormat::Body18 => SL_BODY_FORMAT_SL_BODY_FORMAT_BODY_18,
            BodyFormat::Body34 => SL_BODY_FORMAT_SL_BODY_FORMAT_BODY_34,
            BodyFormat::Body38 => SL_BODY_FORMAT_SL_BODY_FORMAT_BODY_38,
            BodyFormat::Body70 => SL_BODY_FORMAT_SL_BODY_FORMAT_BODY_70,
        }
    }
}

/// Body tracking state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BodyTrackingState {
    Off,
    Ok,
    Searching,
    Terminate,
}

impl From<u32> for BodyTrackingState {
    fn from(state: u32) -> Self {
        match state {
            SL_OBJECT_TRACKING_STATE_SL_OBJECT_TRACKING_STATE_OFF => BodyTrackingState::Off,
            SL_OBJECT_TRACKING_STATE_SL_OBJECT_TRACKING_STATE_OK => BodyTrackingState::Ok,
            SL_OBJECT_TRACKING_STATE_SL_OBJECT_TRACKING_STATE_SEARCHING => BodyTrackingState::Searching,
            SL_OBJECT_TRACKING_STATE_SL_OBJECT_TRACKING_STATE_TERMINATE => BodyTrackingState::Terminate,
            _ => BodyTrackingState::Off,
        }
    }
}

/// Body tracking parameters
#[derive(Debug, Clone)]
pub struct BodyTrackingParameters {
    pub instance_module_id: u32,
    pub enable_tracking: bool,
    pub enable_body_fitting: bool,
    pub body_format: BodyFormat,
    pub body_selection: BodyTrackingSelection,
    pub max_range: f32,
    pub prediction_timeout_s: f32,
    pub allow_reduced_precision_inference: bool,
}

/// Body tracking selection mode
#[derive(Debug, Clone, Copy)]
pub enum BodyTrackingSelection {
    Full,
    UpperBody,
}

impl From<BodyTrackingSelection> for u32 {
    fn from(selection: BodyTrackingSelection) -> Self {
        match selection {
            BodyTrackingSelection::Full => SL_BODY_TRACKING_SELECTION_SL_BODY_TRACKING_SELECTION_FULL,
            BodyTrackingSelection::UpperBody => SL_BODY_TRACKING_SELECTION_SL_BODY_TRACKING_SELECTION_UPPER_BODY,
        }
    }
}

impl Default for BodyTrackingParameters {
    fn default() -> Self {
        Self {
            instance_module_id: 0,
            enable_tracking: true,
            enable_body_fitting: false,
            body_format: BodyFormat::Body18,
            body_selection: BodyTrackingSelection::Full,
            max_range: 20.0,
            prediction_timeout_s: 0.2,
            allow_reduced_precision_inference: false,
        }
    }
}

impl BodyTrackingParameters {
    /// Create new body tracking parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the instance module ID
    pub fn with_instance_id(mut self, id: u32) -> Self {
        self.instance_module_id = id;
        self
    }

    /// Enable or disable body tracking
    pub fn with_tracking(mut self, enable: bool) -> Self {
        self.enable_tracking = enable;
        self
    }

    /// Enable or disable body fitting
    pub fn with_body_fitting(mut self, enable: bool) -> Self {
        self.enable_body_fitting = enable;
        self
    }

    /// Set the body format
    pub fn with_body_format(mut self, format: BodyFormat) -> Self {
        self.body_format = format;
        self
    }

    /// Set the body selection mode
    pub fn with_body_selection(mut self, selection: BodyTrackingSelection) -> Self {
        self.body_selection = selection;
        self
    }

    /// Set the maximum detection range
    pub fn with_max_range(mut self, range: f32) -> Self {
        self.max_range = range;
        self
    }

    /// Set the prediction timeout
    pub fn with_prediction_timeout(mut self, timeout: f32) -> Self {
        self.prediction_timeout_s = timeout;
        self
    }
}

/// Runtime parameters for body tracking
#[derive(Debug, Clone)]
pub struct BodyTrackingRuntimeParameters {
    pub detection_confidence_threshold: f32,
    pub minimum_keypoints_threshold: i32,
    pub skeleton_smoothing: f32,
}

impl Default for BodyTrackingRuntimeParameters {
    fn default() -> Self {
        Self {
            detection_confidence_threshold: 20.0,
            minimum_keypoints_threshold: -1,
            skeleton_smoothing: 0.0,
        }
    }
}

impl BodyTrackingRuntimeParameters {
    /// Create new runtime parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Set detection confidence threshold
    pub fn with_detection_confidence_threshold(mut self, threshold: f32) -> Self {
        self.detection_confidence_threshold = threshold;
        self
    }

    /// Set minimum keypoints threshold
    pub fn with_minimum_keypoints_threshold(mut self, threshold: i32) -> Self {
        self.minimum_keypoints_threshold = threshold;
        self
    }

    /// Set skeleton smoothing
    pub fn with_skeleton_smoothing(mut self, smoothing: f32) -> Self {
        self.skeleton_smoothing = smoothing;
        self
    }
}

/// Keypoint data
#[derive(Debug, Clone)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl From<SL_Vector3> for Keypoint {
    fn from(v: SL_Vector3) -> Self {
        Keypoint { x: v.x, y: v.y, z: v.z }
    }
}

/// Body data containing skeleton information
#[derive(Debug, Clone)]
pub struct BodyData {
    pub id: i32,
    pub unique_object_id: String,
    pub tracking_state: BodyTrackingState,
    pub action_state: u32,
    pub position: Vector3,
    pub velocity: Vector3,
    pub dimensions: Vector3,
    pub confidence: f32,
    pub keypoint_3d: Vec<Keypoint>,
    pub keypoint_2d: Vec<Vector2>,
    pub keypoint_confidence: Vec<f32>,
    pub local_position_per_joint: Vec<Vector3>,
    pub local_orientation_per_joint: Vec<Quaternion>,
    pub global_root_orientation: Quaternion,
    pub bounding_box_2d: [Vector2; 4],
    pub bounding_box: [Vector3; 8],
    pub head_bounding_box_2d: [Vector2; 4],
    pub head_bounding_box: [Vector3; 8],
    pub head_position: Vector3,
}

impl BodyData {
    /// Check if the body is currently being tracked
    pub fn is_tracked(&self) -> bool {
        self.tracking_state == BodyTrackingState::Ok
    }

    /// Check if the body is valid (not terminated)
    pub fn is_valid(&self) -> bool {
        self.tracking_state != BodyTrackingState::Terminate
    }

    /// Get the 2D bounding box center point
    pub fn get_2d_center(&self) -> Vector2 {
        let sum_x = self.bounding_box_2d.iter().map(|p| p.x).sum::<f32>();
        let sum_y = self.bounding_box_2d.iter().map(|p| p.y).sum::<f32>();
        Vector2::new(sum_x / 4.0, sum_y / 4.0)
    }

    /// Get the 2D bounding box width and height
    pub fn get_2d_size(&self) -> (f32, f32) {
        let min_x = self.bounding_box_2d.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
        let max_x = self.bounding_box_2d.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
        let min_y = self.bounding_box_2d.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
        let max_y = self.bounding_box_2d.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);
        (max_x - min_x, max_y - min_y)
    }
}

/// Collection of detected bodies
#[derive(Debug, Clone)]
pub struct Bodies {
    pub timestamp: u64,
    pub is_new: bool,
    pub is_tracked: bool,
    pub bodies: Vec<BodyData>,
}

impl Bodies {
    /// Get the number of detected bodies
    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    /// Check if any bodies were detected
    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    /// Get only tracked bodies
    pub fn get_tracked_bodies(&self) -> Vec<&BodyData> {
        self.bodies.iter().filter(|body| body.is_tracked()).collect()
    }

    /// Get body by ID
    pub fn get_body_by_id(&self, id: i32) -> Option<&BodyData> {
        self.bodies.iter().find(|body| body.id == id)
    }
}

/// Plane type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlaneType {
    Horizontal,
    Vertical,
    Unknown,
}

impl From<u32> for PlaneType {
    fn from(plane_type: u32) -> Self {
        match plane_type {
            SL_PLANE_TYPE_SL_PLANE_TYPE_HORIZONTAL => PlaneType::Horizontal,
            SL_PLANE_TYPE_SL_PLANE_TYPE_VERTICAL => PlaneType::Vertical,
            _ => PlaneType::Unknown,
        }
    }
}

/// Plane data
#[derive(Debug, Clone)]
pub struct PlaneData {
    pub error_code: i32,
    pub plane_type: PlaneType,
    pub plane_normal: Vector3,
    pub plane_center: Vector3,
    pub plane_transform_position: Vector3,
    pub plane_transform_orientation: Quaternion,
    pub plane_equation: [f32; 4],
    pub extents: Vector2,
    pub bounds: Vec<Vector3>,
}

impl From<SL_PlaneData> for PlaneData {
    fn from(plane: SL_PlaneData) -> Self {
        let mut bounds = Vec::new();
        for i in 0..plane.bounds_size as usize {
            if i < 256 { // MAX_PLANE_VERTEX_COUNT
                bounds.push(plane.bounds[i].into());
            }
        }

        PlaneData {
            error_code: plane.error_code,
            plane_type: plane.r#type.into(),
            plane_normal: plane.plane_normal.into(),
            plane_center: plane.plane_center.into(),
            plane_transform_position: plane.plane_transform_position.into(),
            plane_transform_orientation: plane.plane_transform_orientation.into(),
            plane_equation: plane.plane_equation,
            extents: Vector2::new(plane.extents.x, plane.extents.y),
            bounds,
        }
    }
}

/// Plane detection parameters
#[derive(Debug, Clone)]
pub struct PlaneDetectionParameters {
    pub max_distance_threshold: f32,
    pub normal_similarity_threshold: f32,
}

impl Default for PlaneDetectionParameters {
    fn default() -> Self {
        Self {
            max_distance_threshold: 0.15,
            normal_similarity_threshold: 15.0,
        }
    }
}

impl PlaneDetectionParameters {
    /// Create new plane detection parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max distance threshold
    pub fn with_max_distance_threshold(mut self, threshold: f32) -> Self {
        self.max_distance_threshold = threshold;
        self
    }

    /// Set normal similarity threshold
    pub fn with_normal_similarity_threshold(mut self, threshold: f32) -> Self {
        self.normal_similarity_threshold = threshold;
        self
    }
}

