//! Export targets and manifest contracts.

use std::fs;
use std::path::Path;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use wx_types::{Grid2D, TrainingSpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    Arrow,
    Parquet,
    WebDataset,
    Zarr,
    Jsonl,
    Npz,
    NpyDirectory,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportPlan {
    pub spec: TrainingSpec,
    pub format: ExportFormat,
    pub shard_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub dataset_name: String,
    pub generated_at: String,
    pub format: ExportFormat,
    pub shard_count: usize,
    pub channels: Vec<String>,
    pub labels: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelStats {
    pub min: f64,
    pub mean: f64,
    pub max: f64,
    #[serde(default)]
    pub std: f64,
    #[serde(default)]
    pub count: usize,
    #[serde(default)]
    pub nan_count: usize,
}

/// Per-channel normalization statistics for `channel_stats.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelNormEntry {
    pub name: String,
    pub units: String,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
    pub count: usize,
    pub nan_count: usize,
}

/// Top-level structure written to `channel_stats.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelNormStats {
    pub channels: Vec<ChannelNormEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SampleChannelArtifact {
    pub message_no: u64,
    pub name: String,
    pub level: String,
    pub units: String,
    pub width: usize,
    pub height: usize,
    pub missing_count: usize,
    pub data_file: String,
    pub preview_file: Option<String>,
    pub stats: Option<ChannelStats>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SampleBundleManifest {
    pub dataset_name: String,
    pub sample_id: String,
    pub generated_at: String,
    pub format: ExportFormat,
    pub source: String,
    pub channel_count: usize,
    pub channels: Vec<SampleChannelArtifact>,
}

#[derive(Debug, Default)]
pub struct ExportEngine;

impl ExportEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn recommended_formats(&self) -> [ExportFormat; 4] {
        [
            ExportFormat::Arrow,
            ExportFormat::Parquet,
            ExportFormat::WebDataset,
            ExportFormat::Zarr,
        ]
    }

    pub fn manifest_from_plan(&self, plan: &ExportPlan) -> DatasetManifest {
        DatasetManifest {
            dataset_name: plan.spec.dataset_name.clone(),
            generated_at: Utc::now().to_rfc3339(),
            format: plan.format,
            shard_count: plan.shard_count,
            channels: plan
                .spec
                .channels
                .iter()
                .map(|channel| channel.name.clone())
                .collect(),
            labels: plan.spec.labels.clone(),
        }
    }

    pub fn to_json_pretty(&self, manifest: &DatasetManifest) -> Result<String, String> {
        serde_json::to_string_pretty(manifest)
            .map_err(|err| format!("failed to serialize manifest: {err}"))
    }

    pub fn write_manifest(
        &self,
        path: impl AsRef<Path>,
        manifest: &DatasetManifest,
    ) -> Result<(), String> {
        let json = self.to_json_pretty(manifest)?;
        fs::write(path.as_ref(), json).map_err(|err| {
            format!(
                "failed to write manifest '{}': {err}",
                path.as_ref().display()
            )
        })
    }

    pub fn sample_bundle_manifest(
        &self,
        plan: &ExportPlan,
        sample_id: impl Into<String>,
        source: impl Into<String>,
        channels: Vec<SampleChannelArtifact>,
    ) -> SampleBundleManifest {
        SampleBundleManifest {
            dataset_name: plan.spec.dataset_name.clone(),
            sample_id: sample_id.into(),
            generated_at: Utc::now().to_rfc3339(),
            format: plan.format,
            source: source.into(),
            channel_count: channels.len(),
            channels,
        }
    }

    pub fn write_sample_bundle_manifest(
        &self,
        path: impl AsRef<Path>,
        manifest: &SampleBundleManifest,
    ) -> Result<(), String> {
        let json = serde_json::to_string_pretty(manifest)
            .map_err(|err| format!("failed to serialize sample manifest: {err}"))?;
        fs::write(path.as_ref(), json).map_err(|err| {
            format!(
                "failed to write sample manifest '{}': {err}",
                path.as_ref().display()
            )
        })
    }

    pub fn write_npy_f32_grid(&self, path: impl AsRef<Path>, grid: &Grid2D) -> Result<(), String> {
        let mut bytes = build_npy_f32_header(grid.ny, grid.nx)?;
        for value in &grid.values {
            bytes.extend_from_slice(&(*value as f32).to_le_bytes());
        }
        fs::write(path.as_ref(), bytes)
            .map_err(|err| format!("failed to write npy '{}': {err}", path.as_ref().display()))
    }
}

fn build_npy_f32_header(ny: usize, nx: usize) -> Result<Vec<u8>, String> {
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({ny}, {nx}), }}");
    let preamble_len = 10usize;
    let total_without_padding = preamble_len + header.len() + 1;
    let padding = (16 - (total_without_padding % 16)) % 16;
    header.push_str(&" ".repeat(padding));
    header.push('\n');
    let header_len: u16 = header
        .len()
        .try_into()
        .map_err(|_| "npy header exceeds version 1.0 size limit".to_string())?;

    let mut bytes = Vec::with_capacity(preamble_len + header.len());
    bytes.extend_from_slice(b"\x93NUMPY");
    bytes.push(1);
    bytes.push(0);
    bytes.extend_from_slice(&header_len.to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use wx_types::Grid2D;

    #[test]
    fn builds_sample_bundle_manifest() {
        let engine = ExportEngine::new();
        let plan = ExportPlan {
            spec: TrainingSpec {
                dataset_name: "demo".to_string(),
                channels: Vec::new(),
                labels: Vec::new(),
            },
            format: ExportFormat::NpyDirectory,
            shard_count: 1,
        };
        let manifest = engine.sample_bundle_manifest(
            &plan,
            "sample-001",
            "examples/sample.grib2",
            vec![SampleChannelArtifact {
                message_no: 1,
                name: "2t".to_string(),
                level: "2 m above ground".to_string(),
                units: "K".to_string(),
                width: 2,
                height: 2,
                missing_count: 0,
                data_file: "2t.npy".to_string(),
                preview_file: Some("2t.png".to_string()),
                stats: Some(ChannelStats {
                    min: 251.0,
                    mean: 252.5,
                    max: 254.0,
                    std: 1.118,
                    count: 4,
                    nan_count: 0,
                }),
            }],
        );

        assert_eq!(manifest.dataset_name, "demo");
        assert_eq!(manifest.channel_count, 1);
        assert_eq!(manifest.channels[0].name, "2t");
    }

    #[test]
    fn writes_npy_grid() {
        let engine = ExportEngine::new();
        let grid = Grid2D::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should move forward")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("wxforge_grid_{suffix}.npy"));

        engine
            .write_npy_f32_grid(&path, &grid)
            .expect("npy write should succeed");
        let bytes = fs::read(&path).expect("npy file should exist");
        let _ = fs::remove_file(&path);

        assert!(bytes.starts_with(b"\x93NUMPY"));
        assert!(
            String::from_utf8_lossy(&bytes[..64]).contains("'shape': (2, 2)")
                || String::from_utf8_lossy(&bytes[..80]).contains("'shape': (2, 2)")
        );
        assert!(bytes.len() >= 16 + 16);
    }
}
