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
//! - Error handling with Rust Result types
//!
//! ## Example
//!
//! ```no_run
//! use zed_sdk::{Camera, InitParameters, Resolution, ViewType, MemoryType, 
//!               PositionalTrackingParameters, ReferenceFrame,
//!               ObjectDetectionParameters, ObjectDetectionRuntimeParameters,
//!               ObjectDetectionModel, ObjectClass};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut camera = Camera::new(0)?;
//!     
//!     let init_params = InitParameters::default()
//!         .with_resolution(Resolution::HD1080)
//!         .with_fps(30);
//!     
//!     camera.open(&init_params)?;
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
//!     println!("Camera serial: {}", camera.get_serial_number()?);
//!     
//!     // Capture frames and retrieve data
//!     for i in 0..10 {
//!         camera.grab()?;
//!         
//!         // Get image data
//!         let left_image = camera.retrieve_image(ViewType::Left, MemoryType::Cpu)?;
//!         println!("Frame {}: {}x{} image", i, left_image.width, left_image.height);
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
//!         // Get camera pose
//!         if let Ok(pose_data) = camera.get_pose_data(ReferenceFrame::World) {
//!             if pose_data.is_valid() {
//!                 println!("  Camera position: ({:.2}, {:.2}, {:.2})", 
//!                     pose_data.translation.x, pose_data.translation.y, pose_data.translation.z);
//!             }
//!         }
//!     }
//!     
//!     camera.disable_object_detection(0, false)?;
//!     camera.disable_positional_tracking(None)?;
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