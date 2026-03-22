//! Radar ingest and analysis boundaries.

use serde::{Deserialize, Serialize};
use wx_types::{RadarProduct, RadarVolumeRef};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RadarCapabilities {
    pub supports_level2: bool,
    pub supports_level3: bool,
    pub supports_color_table_transforms: bool,
    pub supports_rotation_detection: bool,
    pub supports_cell_tracking: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RadarRenderIntent {
    pub product: RadarProduct,
    pub color_table_name: String,
    pub data_space_transforming: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColorStop {
    pub value: i32,
    pub rgba: [u8; 4],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColorTable {
    pub name: String,
    pub stops: Vec<ColorStop>,
}

impl ColorTable {
    pub fn from_pal_str(name: &str, pal: &str) -> Result<Self, String> {
        let mut stops = Vec::new();

        for (line_no, line) in pal.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() < 4 {
                return Err(format!("invalid palette line {}: {}", line_no + 1, trimmed));
            }

            let value = parts[0]
                .parse::<i32>()
                .map_err(|err| format!("invalid value on line {}: {err}", line_no + 1))?;
            let red = parts[1]
                .parse::<u8>()
                .map_err(|err| format!("invalid red channel on line {}: {err}", line_no + 1))?;
            let green = parts[2]
                .parse::<u8>()
                .map_err(|err| format!("invalid green channel on line {}: {err}", line_no + 1))?;
            let blue = parts[3]
                .parse::<u8>()
                .map_err(|err| format!("invalid blue channel on line {}: {err}", line_no + 1))?;
            let alpha = if parts.len() >= 5 {
                parts[4].parse::<u8>().map_err(|err| {
                    format!("invalid alpha channel on line {}: {err}", line_no + 1)
                })?
            } else {
                255
            };

            stops.push(ColorStop {
                value,
                rgba: [red, green, blue, alpha],
            });
        }

        if stops.is_empty() {
            return Err("palette must contain at least one stop".to_string());
        }

        stops.sort_by_key(|stop| stop.value);
        Ok(Self {
            name: name.to_string(),
            stops,
        })
    }

    pub fn sample(&self, value: i32) -> [u8; 4] {
        if value <= self.stops[0].value {
            return self.stops[0].rgba;
        }
        if value >= self.stops[self.stops.len() - 1].value {
            return self.stops[self.stops.len() - 1].rgba;
        }

        for pair in self.stops.windows(2) {
            let left = &pair[0];
            let right = &pair[1];
            if value >= left.value && value <= right.value {
                let span = (right.value - left.value) as f32;
                let t = (value - left.value) as f32 / span;
                return [
                    lerp_u8(left.rgba[0], right.rgba[0], t),
                    lerp_u8(left.rgba[1], right.rgba[1], t),
                    lerp_u8(left.rgba[2], right.rgba[2], t),
                    lerp_u8(left.rgba[3], right.rgba[3], t),
                ];
            }
        }

        self.stops[self.stops.len() - 1].rgba
    }
}

fn lerp_u8(start: u8, end: u8, t: f32) -> u8 {
    let value = (start as f32) + ((end as f32) - (start as f32)) * t.clamp(0.0, 1.0);
    value.round() as u8
}

#[derive(Debug, Default)]
pub struct RadarEngine;

impl RadarEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn capabilities(&self) -> RadarCapabilities {
        RadarCapabilities {
            supports_level2: true,
            supports_level3: true,
            supports_color_table_transforms: true,
            supports_rotation_detection: true,
            supports_cell_tracking: true,
        }
    }

    pub fn describe_volume(&self, volume: &RadarVolumeRef) -> String {
        format!(
            "Radar volume {} at {} with {} sweeps",
            volume.site, volume.scan_time, volume.sweeps
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_palette_and_samples_midpoint() {
        let palette = "\
0 0 0 0
50 255 0 0
100 255 255 0
";
        let table = ColorTable::from_pal_str("demo", palette).expect("palette should parse");
        assert_eq!(table.sample(0), [0, 0, 0, 255]);
        assert_eq!(table.sample(75), [255, 128, 0, 255]);
    }
}
