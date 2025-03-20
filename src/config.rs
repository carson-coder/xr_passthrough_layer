use serde::{Deserialize, Serialize};

/// Because your eye and the camera is at different physical locations, it is impossible
/// to project camera view into VR space perfectly. There are trade offs approximating
/// this projection. (viewing range means things too close to you will give you double vision).
#[derive(Eq, PartialEq, Debug, Serialize, Deserialize, Clone, Copy, PartialOrd, Ord)]
pub enum ProjectionMode {
    /// in this mode, we assume your eyes are at the cameras' physical location. this mode
    /// has larger viewing range, but everything will smaller to you.
    FromCamera,
    /// in this mode, we assume your cameras are at your eyes' physical location. everything will
    /// have the right scale in this mode, but the viewing range is smaller.
    FromEye,
}

impl Default for ProjectionMode {
    fn default() -> Self {
        Self::FromCamera
    }
}
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Clone, Copy, PartialOrd, Ord)]
pub enum Eye {
    Left,
    Right,
}

pub const fn default_display_eye() -> Eye {
    Eye::Left
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
#[serde(tag = "mode")]
pub enum DisplayMode {
    #[default]
    Direct,
    /// display a stereo image on the overlay. conceptually the overlay becomes a portal from VR
    /// space to real world. you will be able to see more of the real world if the overlay occupys
    /// more of your field of view.
    Stereo {
        /// how is the camera's image projected onto the overlay
        #[serde(default)]
        projection_mode: ProjectionMode,
    },
    /// display one of the camera's image on the overlay
    Flat {
        /// which camera's image to display
        #[serde(default = "default_display_eye")]
        eye: Eye,
    },
}

impl DisplayMode {
    pub(crate) fn projection_mode(&self) -> Option<ProjectionMode> {
        match self {
            DisplayMode::Stereo { projection_mode } => Some(*projection_mode),
            _ => None,
        }
    }
    pub(crate) fn is_stereo(&self) -> bool {
        matches!(self, DisplayMode::Stereo { .. } | DisplayMode::Direct)
    }
}

pub const fn default_z_order() -> u32 {
    u32::MAX
}

/// Index camera passthrough
#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    /// camera device to use. auto detect if not set
    #[serde(default)]
    pub camera_device: String,
    /// how is the camera view displayed on the overlay
    #[serde(default)]
    pub display_mode: DisplayMode,
    /// enable debug option, including:
    ///   - use trigger button to do renderdoc capture
    #[serde(default)]
    pub debug: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            camera_device: "".to_owned(),
            display_mode: Default::default(),
            debug: false,
        }
    }
}

use anyhow::Result;
use xdg::BaseDirectories;
pub fn load_config(xdg: &BaseDirectories) -> Result<Config> {
    if let Some(f) = xdg.find_config_file("index_camera_passthrough.toml") {
        let cfg = std::fs::read_to_string(f)?;
        Ok(toml::from_str(&cfg)?)
    } else {
        Ok(Default::default())
    }
}
