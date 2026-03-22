//! Rendering surfaces for diagnostics and ML products.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use png::Encoder;
use wx_types::Grid2D;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderTarget {
    Png,
    Tile256,
    OverlayMask,
    TrainingRaster,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderJob {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub target: RenderTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMap {
    Gray,
    Heat,
    Radar,
}

#[derive(Debug, Default)]
pub struct RenderEngine;

impl RenderEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn design_goal(&self) -> &'static str {
        "Deterministic meteorological rendering for diagnostics, tiles, and ML dataset generation."
    }

    pub fn validate_job(&self, job: &RenderJob) -> bool {
        job.width > 0 && job.height > 0
    }
}

pub fn render_scalar_grid(grid: &Grid2D, min: f64, max: f64, cmap: ColorMap) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(grid.values.len() * 4);
    for value in &grid.values {
        let t = if (max - min).abs() < f64::EPSILON {
            0.5
        } else {
            ((*value - min) / (max - min)).clamp(0.0, 1.0)
        };
        let color = sample_colormap(cmap, t as f32);
        rgba.extend_from_slice(&color);
    }
    rgba
}

pub fn write_png_rgba(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> Result<(), String> {
    let file = File::create(path.as_ref())
        .map_err(|err| format!("failed to create png '{}': {err}", path.as_ref().display()))?;
    let writer = BufWriter::new(file);
    let mut encoder = Encoder::new(writer, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);

    let mut png_writer = encoder
        .write_header()
        .map_err(|err| format!("failed to write png header: {err}"))?;
    png_writer
        .write_image_data(rgba)
        .map_err(|err| format!("failed to write png data: {err}"))?;
    Ok(())
}

fn sample_colormap(cmap: ColorMap, t: f32) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    match cmap {
        ColorMap::Gray => {
            let channel = (255.0 * t).round() as u8;
            [channel, channel, channel, 255]
        }
        ColorMap::Heat => {
            if t < 0.5 {
                let local = t / 0.5;
                [lerp(0, 255, local), 0, 0, 255]
            } else {
                let local = (t - 0.5) / 0.5;
                [255, lerp(0, 255, local), 0, 255]
            }
        }
        ColorMap::Radar => {
            if t < 0.33 {
                let local = t / 0.33;
                [0, lerp(32, 255, local), lerp(64, 0, local), 255]
            } else if t < 0.66 {
                let local = (t - 0.33) / 0.33;
                [lerp(0, 255, local), 255, 0, 255]
            } else {
                let local = (t - 0.66) / 0.34;
                [255, lerp(255, 0, local), lerp(0, 255, local), 255]
            }
        }
    }
}

fn lerp(start: u8, end: u8, t: f32) -> u8 {
    (start as f32 + ((end as f32) - (start as f32)) * t.clamp(0.0, 1.0)).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_grayscale_grid() {
        let grid = Grid2D::new(2, 2, vec![0.0, 0.5, 1.0, 0.25]);
        let rgba = render_scalar_grid(&grid, 0.0, 1.0, ColorMap::Gray);
        assert_eq!(rgba.len(), 16);
        assert_eq!(&rgba[0..4], &[0, 0, 0, 255]);
        assert_eq!(&rgba[8..12], &[255, 255, 255, 255]);
    }
}
