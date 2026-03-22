//! Shared weather-domain types for the wxtrain workspace.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectionKind {
    LatLon,
    LambertConformal,
    PolarStereographic,
    Mercator,
    RotatedLatLon,
    RadarPolar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataFormat {
    Grib1,
    Grib2,
    Netcdf,
    Zarr,
    NexradLevel2,
    NexradLevel3,
    Bufkit,
    Csv,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GridShape {
    pub ny: usize,
    pub nx: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GridSpec {
    pub projection: ProjectionKind,
    pub shape: GridShape,
    pub dx_m: Option<f64>,
    pub dy_m: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldKey {
    pub dataset: String,
    pub variable: String,
    pub level: String,
    pub valid_time: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldMeta {
    pub key: FieldKey,
    pub units: String,
    pub source_format: DataFormat,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Field2D {
    pub meta: FieldMeta,
    pub grid: GridSpec,
    pub values_len: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoundingLevel {
    pub pressure_hpa: f64,
    pub temperature_c: f64,
    pub dewpoint_c: f64,
    pub height_m: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoundingProfile {
    pub station_id: String,
    pub valid_time: String,
    pub levels: Vec<SoundingLevel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RadarProduct {
    Reflectivity,
    Velocity,
    SpectrumWidth,
    DifferentialReflectivity,
    CorrelationCoefficient,
    DifferentialPhase,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RadarVolumeRef {
    pub site: String,
    pub scan_time: String,
    pub sweeps: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingChannel {
    pub name: String,
    pub units: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingSpec {
    pub dataset_name: String,
    pub channels: Vec<TrainingChannel>,
    pub labels: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Grid2D {
    pub nx: usize,
    pub ny: usize,
    pub values: Vec<f64>,
}

impl Grid2D {
    pub fn new(nx: usize, ny: usize, values: Vec<f64>) -> Self {
        assert_eq!(
            nx * ny,
            values.len(),
            "grid shape and data length must match"
        );
        Self { nx, ny, values }
    }

    pub fn zeros(nx: usize, ny: usize) -> Self {
        Self {
            nx,
            ny,
            values: vec![0.0; nx * ny],
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn index(&self, x: usize, y: usize) -> usize {
        y * self.nx + x
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        self.values[self.index(x, y)]
    }

    pub fn set(&mut self, x: usize, y: usize, value: f64) {
        let idx = self.index(x, y);
        self.values[idx] = value;
    }
}
