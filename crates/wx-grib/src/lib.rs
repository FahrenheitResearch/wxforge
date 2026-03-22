//! GRIB inventory and decode boundaries.

use std::fs;
use std::path::Path;

use dicom_toolkit_jpeg2000::{DecodeSettings as Jpeg2000DecodeSettings, Image as Jpeg2000Image};
use rust_aec::{decode as decode_aec, flags_from_grib2_ccsds_flags, AecFlags, AecParams};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use wx_types::{
    DataFormat, Field2D, FieldKey, FieldMeta, Grid2D, GridShape, GridSpec, ProjectionKind,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GribEdition {
    Grib1,
    Grib2,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageDescriptor {
    pub message_no: u64,
    pub message_id: String,
    pub offset_bytes: u64,
    pub length_bytes: u64,
    pub edition: Option<GribEdition>,
    pub reference: String,
    pub variable: String,
    pub parameter_name: Option<String>,
    pub units: Option<String>,
    pub level: String,
    pub reference_time: Option<String>,
    pub forecast_time_value: Option<u32>,
    pub forecast_time_unit: Option<String>,
    pub discipline: Option<u8>,
    pub category: Option<u8>,
    pub parameter_number: Option<u8>,
    pub level_type: Option<u8>,
    pub level_value: Option<f64>,
    pub grid_template: Option<u16>,
    pub nx: Option<u32>,
    pub ny: Option<u32>,
    pub extra: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GribInventory {
    pub format: DataFormat,
    pub messages: Vec<MessageDescriptor>,
}

/// Flip rows in-place (reverses row order for bottom-to-top scan modes).
fn flip_rows(values: &mut [f64], nx: usize, ny: usize) {
    for j in 0..ny / 2 {
        let top = j * nx;
        let bot = (ny - 1 - j) * nx;
        for i in 0..nx {
            values.swap(top + i, bot + i);
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedField {
    pub descriptor: MessageDescriptor,
    pub grid: Grid2D,
    pub grid_spec: GridSpec,
    pub x_axis: Option<CoordinateAxis>,
    pub y_axis: Option<CoordinateAxis>,
    pub missing_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoordinateAxis {
    pub name: String,
    pub units: String,
    pub values: Vec<f64>,
}

impl DecodedField {
    pub fn summary_field(&self) -> Field2D {
        Field2D {
            meta: FieldMeta {
                key: FieldKey {
                    dataset: "decoded_grib".to_string(),
                    variable: self.descriptor.variable.clone(),
                    level: self.descriptor.level.clone(),
                    valid_time: self.descriptor.reference_time.clone(),
                },
                units: self
                    .descriptor
                    .units
                    .clone()
                    .unwrap_or_else(|| "?".to_string()),
                source_format: match self.descriptor.edition {
                    Some(GribEdition::Grib1) => DataFormat::Grib1,
                    _ => DataFormat::Grib2,
                },
            },
            grid: self.grid_spec.clone(),
            values_len: self.grid.len(),
        }
    }

    pub fn min_mean_max(&self) -> Option<(f64, f64, f64)> {
        let mut count = 0usize;
        let mut sum = 0.0;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for value in &self.grid.values {
            if value.is_nan() {
                continue;
            }
            count += 1;
            sum += *value;
            min = min.min(*value);
            max = max.max(*value);
        }

        if count == 0 {
            None
        } else {
            Some((min, sum / count as f64, max))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderCapabilities {
    pub supports_grib1: bool,
    pub supports_grib2: bool,
    pub supports_ccsds: bool,
    pub supports_jpeg2000: bool,
    pub supports_png_packing: bool,
}

#[derive(Debug, Default)]
pub struct GribEngine;

#[derive(Debug, Clone, Copy, PartialEq)]
struct GridSectionMetadata {
    template: u16,
    nx: usize,
    ny: usize,
    scan_mode: u8,
    projection: ProjectionKind,
    lat1: Option<f64>,
    lon1: Option<f64>,
    lat2: Option<f64>,
    lon2: Option<f64>,
    dx: Option<f64>,
    dy: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct Grib1ProductMetadata {
    table_version: u8,
    center: u8,
    has_gds: bool,
    has_bms: bool,
    parameter: u8,
    level_type: u8,
    level_value: u16,
    time_unit: u8,
    forecast_time: u32,
    decimal_scale: i16,
    reference_time: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Grib1BinaryMetadata {
    binary_scale: i16,
    reference_value: f32,
    bits_per_value: u8,
    unused_bits: u8,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct DataRepresentationMetadata {
    template: u16,
    reference_value: f32,
    binary_scale: i16,
    decimal_scale: i16,
    bits_per_value: u8,
    num_points: usize,
    original_value_type: u8,
    compression_type: u8,
    target_compression_ratio: u8,
    ccsds_flags: u8,
    ccsds_block_size: u8,
    ccsds_reference_sample_interval: u16,
    group_splitting_method: u8,
    missing_value_management: u8,
    primary_missing_substitute: u32,
    secondary_missing_substitute: u32,
    number_of_groups: usize,
    group_width_reference: u8,
    group_width_bits: u8,
    group_length_reference: u32,
    group_length_increment: u8,
    last_group_length: u32,
    group_length_bits: u8,
    spatial_differencing_order: u8,
    extra_descriptor_octets: u8,
}

#[derive(Debug, Clone, PartialEq)]
struct SpatialDifferencingMetadata {
    order: usize,
    initial_values: Vec<i64>,
    minimum: i64,
}

impl GribEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn capabilities(&self) -> DecoderCapabilities {
        DecoderCapabilities {
            supports_grib1: true,
            supports_grib2: true,
            supports_ccsds: true,
            supports_jpeg2000: true,
            supports_png_packing: true,
        }
    }

    pub fn design_goal(&self) -> &'static str {
        "One canonical Rust GRIB engine with inventory, decode, and fixture-backed validation."
    }

    pub fn parse_idx_text(&self, text: &str) -> Result<GribInventory, String> {
        let mut messages = Vec::new();
        for (line_index, line) in text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .enumerate()
        {
            let message_no = (line_index + 1) as u64;
            if line.trim_start().starts_with('{') {
                messages.push(parse_json_index_line(line, message_no)?);
            } else {
                messages.push(parse_idx_line(line, message_no)?);
            }
        }
        finalize_lengths(&mut messages);
        Ok(GribInventory {
            format: DataFormat::Grib2,
            messages,
        })
    }

    pub fn scan_file(&self, path: impl AsRef<Path>) -> Result<GribInventory, String> {
        let data = fs::read(path.as_ref())
            .map_err(|err| format!("failed to read '{}': {err}", path.as_ref().display()))?;
        self.scan_bytes(&data)
    }

    pub fn decode_file_message(
        &self,
        path: impl AsRef<Path>,
        message_no: u64,
    ) -> Result<DecodedField, String> {
        let data = fs::read(path.as_ref())
            .map_err(|err| format!("failed to read '{}': {err}", path.as_ref().display()))?;
        self.decode_bytes_message(&data, message_no)
    }

    pub fn scan_bytes(&self, data: &[u8]) -> Result<GribInventory, String> {
        let mut messages = Vec::new();
        let mut cursor = 0usize;

        while let Some(offset) = find_grib_magic(data, cursor) {
            if offset + 8 > data.len() {
                break;
            }

            let edition = data[offset + 7];
            let result = match edition {
                1 => scan_grib1_message(data, offset, messages.len() as u64 + 1),
                2 => scan_grib2_message(data, offset, messages.len() as u64 + 1),
                _ => Err(format!(
                    "unsupported GRIB edition {edition} at byte offset {offset}"
                )),
            };

            match result {
                Ok((message, next_cursor)) => {
                    messages.push(message);
                    cursor = next_cursor;
                }
                Err(_) => {
                    cursor = offset + 4;
                }
            }
        }

        if messages.is_empty() {
            return Err("no GRIB messages found".to_string());
        }

        let format = if messages
            .iter()
            .any(|message| message.edition == Some(GribEdition::Grib1))
        {
            DataFormat::Grib1
        } else {
            DataFormat::Grib2
        };

        Ok(GribInventory { format, messages })
    }

    pub fn decode_bytes_message(
        &self,
        data: &[u8],
        message_no: u64,
    ) -> Result<DecodedField, String> {
        let inventory = self.scan_bytes(data)?;
        let descriptor = inventory
            .messages
            .iter()
            .find(|message| message.message_no == message_no)
            .cloned()
            .ok_or_else(|| format!("message {message_no} not found"))?;

        match descriptor.edition {
            Some(GribEdition::Grib2) => {
                let start = descriptor.offset_bytes as usize;
                let end = start + descriptor.length_bytes as usize;
                decode_grib2_message(&data[start..end], descriptor)
            }
            Some(GribEdition::Grib1) => {
                let start = descriptor.offset_bytes as usize;
                let end = start + descriptor.length_bytes as usize;
                decode_grib1_message(&data[start..end], descriptor)
            }
            None => Err("message edition is unknown".to_string()),
        }
    }

    pub fn search<'a>(
        &self,
        inventory: &'a GribInventory,
        needle: &str,
    ) -> Vec<&'a MessageDescriptor> {
        let needle = needle.to_ascii_lowercase();
        inventory
            .messages
            .iter()
            .filter(|message| {
                message.reference.to_ascii_lowercase().contains(&needle)
                    || message.variable.to_ascii_lowercase().contains(&needle)
                    || message
                        .parameter_name
                        .as_ref()
                        .is_some_and(|name| name.to_ascii_lowercase().contains(&needle))
                    || message.level.to_ascii_lowercase().contains(&needle)
                    || message
                        .units
                        .as_ref()
                        .is_some_and(|units| units.to_ascii_lowercase().contains(&needle))
                    || message
                        .extra
                        .iter()
                        .any(|item| item.to_ascii_lowercase().contains(&needle))
            })
            .collect()
    }

    pub fn placeholder_field(&self) -> Field2D {
        Field2D {
            meta: FieldMeta {
                key: FieldKey {
                    dataset: "placeholder".to_string(),
                    variable: "TMP".to_string(),
                    level: "2m".to_string(),
                    valid_time: None,
                },
                units: "K".to_string(),
                source_format: DataFormat::Grib2,
            },
            grid: GridSpec {
                projection: ProjectionKind::LatLon,
                shape: GridShape { ny: 1, nx: 1 },
                dx_m: None,
                dy_m: None,
            },
            values_len: 1,
        }
    }
}

fn parse_idx_line(line: &str, fallback_message_no: u64) -> Result<MessageDescriptor, String> {
    let parts: Vec<&str> = line.split(':').collect();
    if parts.len() < 5 {
        return Err(format!("invalid idx line: {line}"));
    }

    let raw_message_id = parts[0].trim().to_string();
    let message_no = parts[0]
        .parse::<u64>()
        .or_else(|_| parts[0].parse::<f64>().map(|_| fallback_message_no))
        .unwrap_or(fallback_message_no);
    let offset_bytes = parts[1]
        .parse::<u64>()
        .map_err(|err| format!("invalid byte offset in '{line}': {err}"))?;

    let variable = parts[3].trim().to_string();
    let level = parts[4].trim().to_string();
    let extra: Vec<String> = parts[5..]
        .iter()
        .map(|part| part.trim().to_string())
        .collect();

    Ok(MessageDescriptor {
        message_no,
        message_id: raw_message_id,
        offset_bytes,
        length_bytes: 0,
        edition: Some(GribEdition::Grib2),
        reference: line.trim().to_string(),
        variable,
        parameter_name: None,
        units: None,
        level,
        reference_time: None,
        forecast_time_value: None,
        forecast_time_unit: None,
        discipline: None,
        category: None,
        parameter_number: None,
        level_type: None,
        level_value: None,
        grid_template: None,
        nx: None,
        ny: None,
        extra,
    })
}

fn parse_json_index_line(line: &str, message_no: u64) -> Result<MessageDescriptor, String> {
    let value: Value =
        serde_json::from_str(line).map_err(|err| format!("invalid JSON index line: {err}"))?;
    let offset_bytes = value
        .get("_offset")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("JSON index line missing _offset: {line}"))?;
    let length_bytes = value.get("_length").and_then(Value::as_u64).unwrap_or(0);
    let parameter_name = value
        .get("param")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let level = format_json_level(&value);
    let step = value
        .get("step")
        .and_then(Value::as_str)
        .map(str::to_string);
    let reference_time = value
        .get("date")
        .and_then(Value::as_str)
        .and_then(format_compact_reference_time);

    let variable = parameter_name.to_ascii_lowercase();
    let mut extra = Vec::new();
    if let Some(step) = &step {
        extra.push(step.clone());
    }
    if let Some(stream) = value.get("stream").and_then(Value::as_str) {
        extra.push(stream.to_string());
    }
    if let Some(class) = value.get("class").and_then(Value::as_str) {
        extra.push(class.to_string());
    }

    Ok(MessageDescriptor {
        message_no,
        message_id: message_no.to_string(),
        offset_bytes,
        length_bytes,
        edition: Some(GribEdition::Grib2),
        reference: line.trim().to_string(),
        variable,
        parameter_name: Some(parameter_name),
        units: None,
        level,
        reference_time,
        forecast_time_value: parse_json_step_hours(step.as_deref()),
        forecast_time_unit: Some("hour".to_string()),
        discipline: None,
        category: None,
        parameter_number: None,
        level_type: None,
        level_value: None,
        grid_template: None,
        nx: None,
        ny: None,
        extra,
    })
}

fn finalize_lengths(messages: &mut [MessageDescriptor]) {
    for idx in 0..messages.len() {
        let next_offset = messages[idx + 1..]
            .iter()
            .map(|message| message.offset_bytes)
            .find(|offset| *offset > messages[idx].offset_bytes);
        messages[idx].length_bytes = match next_offset {
            Some(next) if next > messages[idx].offset_bytes => next - messages[idx].offset_bytes,
            _ => 0,
        };
    }
}

fn find_grib_magic(data: &[u8], from: usize) -> Option<usize> {
    if from + 4 > data.len() {
        return None;
    }
    data[from..]
        .windows(4)
        .position(|window| window == b"GRIB")
        .map(|relative| from + relative)
}

fn scan_grib1_message(
    data: &[u8],
    offset: usize,
    message_no: u64,
) -> Result<(MessageDescriptor, usize), String> {
    if offset + 8 > data.len() {
        return Err(format!("truncated GRIB1 message at byte offset {offset}"));
    }

    let length_bytes = read_u24_be(data, offset + 4)? as usize;
    if length_bytes < 8 || offset + length_bytes > data.len() {
        return Err(format!(
            "invalid GRIB1 length {length_bytes} at byte offset {offset}"
        ));
    }
    let message = &data[offset..offset + length_bytes];
    let pds_length = read_u24_be(message, 8)? as usize;
    if message.len() < 8 + pds_length {
        return Err(format!("truncated GRIB1 PDS in message {message_no}"));
    }
    let pds = &message[8..8 + pds_length];
    let product = parse_grib1_pds(pds)?;

    let cursor = 8 + pds_length;
    let mut grid = None;
    if product.has_gds {
        if message.len() < cursor + 3 {
            return Err(format!("truncated GRIB1 GDS in message {message_no}"));
        }
        let gds_length = read_u24_be(message, cursor)? as usize;
        if message.len() < cursor + gds_length {
            return Err(format!("truncated GRIB1 GDS in message {message_no}"));
        }
        grid = Some(parse_grib1_gds(&message[cursor..cursor + gds_length])?);
    }

    let short_name = grib1_short_name(product.center, product.table_version, product.parameter);
    let parameter_name =
        grib1_parameter_name(product.center, product.table_version, product.parameter);
    let units = grib1_parameter_units(product.center, product.table_version, product.parameter);
    let level = format_grib1_level(product.level_type, product.level_value);
    let descriptor = MessageDescriptor {
        message_no,
        message_id: message_no.to_string(),
        offset_bytes: offset as u64,
        length_bytes: length_bytes as u64,
        edition: Some(GribEdition::Grib1),
        reference: format!("GRIB1:offset={offset}:len={length_bytes}"),
        variable: short_name.to_string(),
        parameter_name: Some(parameter_name.to_string()),
        units: Some(units.to_string()),
        level,
        reference_time: product.reference_time,
        forecast_time_value: Some(product.forecast_time),
        forecast_time_unit: Some(time_unit_name(product.time_unit).to_string()),
        discipline: None,
        category: None,
        parameter_number: Some(product.parameter),
        level_type: Some(product.level_type),
        level_value: Some(product.level_value as f64),
        grid_template: grid.map(|item| item.template),
        nx: grid.map(|item| item.nx as u32),
        ny: grid.map(|item| item.ny as u32),
        extra: Vec::new(),
    };

    Ok((descriptor, offset + length_bytes))
}

fn parse_grib1_pds(section: &[u8]) -> Result<Grib1ProductMetadata, String> {
    if section.len() < 28 {
        return Err("GRIB1 PDS too short".to_string());
    }
    let flag = section[7];
    let year_of_century = section[12] as i32;
    let century = section[24] as i32;
    let year = if century > 0 {
        (century - 1) * 100 + year_of_century
    } else if year_of_century >= 70 {
        1900 + year_of_century
    } else {
        2000 + year_of_century
    };
    let reference_time = Some(format!(
        "{year:04}-{:02}-{:02}T{:02}:{:02}:00Z",
        section[13], section[14], section[15], section[16]
    ));

    Ok(Grib1ProductMetadata {
        table_version: section[3],
        center: section[4],
        has_gds: flag & 0x80 != 0,
        has_bms: flag & 0x40 != 0,
        parameter: section[8],
        level_type: section[9],
        level_value: read_u16_be(section, 10)?,
        time_unit: section[17],
        forecast_time: section[18] as u32,
        decimal_scale: read_sign_magnitude_i16_be(section, 26)?,
        reference_time,
    })
}

fn parse_grib1_gds(section: &[u8]) -> Result<GridSectionMetadata, String> {
    if section.len() < 28 {
        return Err("GRIB1 GDS too short".to_string());
    }
    let data_representation_type = section[5];
    if data_representation_type != 0 {
        return Err(format!(
            "GRIB1 GDS data representation type {data_representation_type} is not implemented yet"
        ));
    }
    let nx = read_u16_be(section, 6)? as usize;
    let ny = read_u16_be(section, 8)? as usize;
    let lat1 = read_signed_i24_sign_magnitude(section, 10)? as f64 / 1000.0;
    let mut lon1 = read_signed_i24_sign_magnitude(section, 13)? as f64 / 1000.0;
    let lat2 = read_signed_i24_sign_magnitude(section, 17)? as f64 / 1000.0;
    let mut lon2 = read_signed_i24_sign_magnitude(section, 20)? as f64 / 1000.0;
    if lon1 > 180.0 {
        lon1 -= 360.0;
    }
    if lon2 > 180.0 {
        lon2 -= 360.0;
    }

    Ok(GridSectionMetadata {
        template: 0,
        nx,
        ny,
        scan_mode: section[27],
        projection: ProjectionKind::LatLon,
        lat1: Some(lat1),
        lon1: Some(lon1),
        lat2: Some(lat2),
        lon2: Some(lon2),
        dx: Some(read_u16_be(section, 23)? as f64 / 1000.0),
        dy: Some(read_u16_be(section, 25)? as f64 / 1000.0),
    })
}

fn scan_grib2_message(
    data: &[u8],
    offset: usize,
    message_no: u64,
) -> Result<(MessageDescriptor, usize), String> {
    if offset + 16 > data.len() {
        return Err(format!(
            "truncated GRIB2 indicator section at byte offset {offset}"
        ));
    }

    let discipline = data[offset + 6];
    let length_bytes = read_u64_be(data, offset + 8)? as usize;
    if length_bytes < 20 || offset + length_bytes > data.len() {
        return Err(format!(
            "invalid GRIB2 length {length_bytes} at byte offset {offset}"
        ));
    }

    let message = &data[offset..offset + length_bytes];
    if &message[length_bytes - 4..length_bytes] != b"7777" {
        return Err(format!("missing GRIB2 end marker at byte offset {offset}"));
    }

    let mut cursor = 16usize;
    let mut reference_time = None;
    let mut grid_template = None;
    let mut nx = None;
    let mut ny = None;
    let mut category = None;
    let mut parameter_number = None;
    let mut forecast_time_value = None;
    let mut forecast_time_unit = None;
    let mut level_type = None;
    let mut level_value = None;

    while cursor + 5 <= message.len() {
        if cursor + 4 <= message.len() && &message[cursor..cursor + 4] == b"7777" {
            break;
        }

        let section_length = read_u32_be(message, cursor)? as usize;
        let section_number = message[cursor + 4];
        if section_length < 5 || cursor + section_length > message.len() {
            return Err(format!(
                "invalid GRIB2 section {section_number} length {section_length} in message {message_no}"
            ));
        }

        let section = &message[cursor..cursor + section_length];
        match section_number {
            1 => reference_time = parse_section1_reference_time(section),
            3 => {
                let grid = parse_section3_metadata(section)?;
                grid_template = Some(grid.template);
                nx = Some(grid.nx as u32);
                ny = Some(grid.ny as u32);
            }
            4 => {
                let product = parse_section4_product(section)?;
                category = Some(product.category);
                parameter_number = Some(product.number);
                forecast_time_value = Some(product.forecast_time);
                forecast_time_unit = Some(time_unit_name(product.time_unit).to_string());
                level_type = Some(product.level_type);
                level_value = Some(product.level_value);
            }
            _ => {}
        }

        cursor += section_length;
    }

    let category = category.ok_or_else(|| "GRIB2 message missing section 4".to_string())?;
    let parameter_number =
        parameter_number.ok_or_else(|| "GRIB2 message missing parameter number".to_string())?;
    let level_type = level_type.unwrap_or(255);
    let level_value = level_value.unwrap_or(0.0);
    let short_name = grib2_short_name(
        discipline,
        category,
        parameter_number,
        level_type,
        level_value,
    );
    let parameter_name = grib2_parameter_name(discipline, category, parameter_number);
    let units = grib2_parameter_units(discipline, category, parameter_number);
    let level = format_level(level_type, level_value);
    let reference = format!(
        "{}:{}:{}",
        short_name,
        level,
        forecast_summary(forecast_time_value, forecast_time_unit.as_deref())
    );

    let descriptor = MessageDescriptor {
        message_no,
        message_id: message_no.to_string(),
        offset_bytes: offset as u64,
        length_bytes: length_bytes as u64,
        edition: Some(GribEdition::Grib2),
        reference,
        variable: short_name.to_string(),
        parameter_name: Some(parameter_name.to_string()),
        units: Some(units.to_string()),
        level,
        reference_time,
        forecast_time_value,
        forecast_time_unit,
        discipline: Some(discipline),
        category: Some(category),
        parameter_number: Some(parameter_number),
        level_type: Some(level_type),
        level_value: Some(level_value),
        grid_template,
        nx,
        ny,
        extra: Vec::new(),
    };

    Ok((descriptor, offset + length_bytes))
}

fn parse_section1_reference_time(section: &[u8]) -> Option<String> {
    if section.len() < 19 {
        return None;
    }
    let year = read_u16_be(section, 12).ok()?;
    let month = *section.get(14)?;
    let day = *section.get(15)?;
    let hour = *section.get(16)?;
    let minute = *section.get(17)?;
    let second = *section.get(18)?;
    Some(format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z"
    ))
}

fn parse_section3_metadata(section: &[u8]) -> Result<GridSectionMetadata, String> {
    if section.len() < 38 {
        return Err("section 3 too short".to_string());
    }
    let template = read_u16_be(section, 12)?;
    let nx = read_u32_be(section, 30)? as usize;
    let ny = read_u32_be(section, 34)? as usize;

    match template {
        0 => {
            if section.len() < 72 {
                return Err("section 3 template 0 too short".to_string());
            }
            let basic_angle = read_u32_be(section, 38)?;
            let subdivisions = read_u32_be(section, 42)?;
            let divisor = if basic_angle == 0 || subdivisions == 0 {
                1_000_000.0
            } else {
                subdivisions as f64 / basic_angle as f64
            };

            Ok(GridSectionMetadata {
                template,
                nx,
                ny,
                scan_mode: section[71],
                projection: ProjectionKind::LatLon,
                lat1: Some(read_signed_i32_sign_magnitude(section, 46)? as f64 / divisor),
                lon1: Some(read_signed_i32_sign_magnitude(section, 50)? as f64 / divisor),
                lat2: Some(read_signed_i32_sign_magnitude(section, 55)? as f64 / divisor),
                lon2: Some(read_signed_i32_sign_magnitude(section, 59)? as f64 / divisor),
                dx: Some(read_u32_be(section, 63)? as f64 / divisor),
                dy: Some(read_u32_be(section, 67)? as f64 / divisor),
            })
        }
        30 => {
            if section.len() < 81 {
                return Err("section 3 template 30 too short".to_string());
            }
            Ok(GridSectionMetadata {
                template,
                nx,
                ny,
                scan_mode: section[64],
                projection: ProjectionKind::LambertConformal,
                lat1: Some(read_signed_i32_sign_magnitude(section, 38)? as f64 / 1_000_000.0),
                lon1: Some(read_signed_i32_sign_magnitude(section, 42)? as f64 / 1_000_000.0),
                lat2: None,
                lon2: None,
                dx: Some(read_u32_be(section, 55)? as f64 / 1000.0),
                dy: Some(read_u32_be(section, 59)? as f64 / 1000.0),
            })
        }
        _ => Ok(GridSectionMetadata {
            template,
            nx,
            ny,
            scan_mode: match template {
                10 if section.len() > 58 => section[58],
                20 if section.len() > 63 => section[63],
                _ => 0,
            },
            projection: match template {
                10 => ProjectionKind::Mercator,
                20 => ProjectionKind::PolarStereographic,
                _ => ProjectionKind::LatLon,
            },
            lat1: None,
            lon1: None,
            lat2: None,
            lon2: None,
            dx: None,
            dy: None,
        }),
    }
}

#[derive(Debug, Clone, Copy)]
struct ProductMetadata {
    category: u8,
    number: u8,
    time_unit: u8,
    forecast_time: u32,
    level_type: u8,
    level_value: f64,
}

fn parse_section4_product(section: &[u8]) -> Result<ProductMetadata, String> {
    if section.len() < 28 {
        return Err("section 4 too short".to_string());
    }

    let category = section[9];
    let number = section[10];
    let time_unit = section[17];
    let forecast_time = read_u32_be(section, 18)?;
    let level_type = section[22];
    let scale_factor = section[23];
    let scaled_value = read_u32_be(section, 24)? as f64;
    let level_value = if scale_factor < 128 {
        scaled_value / 10.0_f64.powi(scale_factor as i32)
    } else {
        scaled_value * 10.0_f64.powi((256 - scale_factor as i32).abs())
    };

    Ok(ProductMetadata {
        category,
        number,
        time_unit,
        forecast_time,
        level_type,
        level_value,
    })
}

fn parse_section5_metadata(section: &[u8]) -> Result<DataRepresentationMetadata, String> {
    if section.len() < 20 {
        return Err("section 5 too short".to_string());
    }

    let template = read_u16_be(section, 9)?;
    let mut metadata = DataRepresentationMetadata {
        template,
        num_points: read_u32_be(section, 5)? as usize,
        reference_value: read_f32_be(section, 11)?,
        binary_scale: read_sign_magnitude_i16_be(section, 15)?,
        decimal_scale: read_sign_magnitude_i16_be(section, 17)?,
        bits_per_value: section[19],
        original_value_type: 0,
        compression_type: 0,
        target_compression_ratio: 0,
        ccsds_flags: 0,
        ccsds_block_size: 0,
        ccsds_reference_sample_interval: 0,
        group_splitting_method: 0,
        missing_value_management: 0,
        primary_missing_substitute: 0,
        secondary_missing_substitute: 0,
        number_of_groups: 0,
        group_width_reference: 0,
        group_width_bits: 0,
        group_length_reference: 0,
        group_length_increment: 0,
        last_group_length: 0,
        group_length_bits: 0,
        spatial_differencing_order: 0,
        extra_descriptor_octets: 0,
    };

    match template {
        2 | 3 => {
            if section.len() < 47 {
                return Err(format!("section 5 template {template} too short"));
            }
            metadata.group_splitting_method = section[21];
            metadata.missing_value_management = section[22];
            metadata.primary_missing_substitute = read_u32_be(section, 23)?;
            metadata.secondary_missing_substitute = read_u32_be(section, 27)?;
            metadata.number_of_groups = read_u32_be(section, 31)? as usize;
            metadata.group_width_reference = section[35];
            metadata.group_width_bits = section[36];
            metadata.group_length_reference = read_u32_be(section, 37)?;
            metadata.group_length_increment = section[41];
            metadata.last_group_length = read_u32_be(section, 42)?;
            metadata.group_length_bits = section[46];

            if template == 3 {
                if section.len() < 49 {
                    return Err("section 5 template 3 too short".to_string());
                }
                metadata.spatial_differencing_order = section[47];
                metadata.extra_descriptor_octets = section[48];
            }
        }
        40 => {
            if section.len() < 23 {
                return Err("section 5 template 40 too short".to_string());
            }
            metadata.original_value_type = section[20];
            metadata.compression_type = section[21];
            metadata.target_compression_ratio = section[22];
        }
        42 => {
            if section.len() < 25 {
                return Err("section 5 template 42 too short".to_string());
            }
            metadata.original_value_type = section[20];
            metadata.ccsds_flags = section[21];
            metadata.ccsds_block_size = section[22];
            metadata.ccsds_reference_sample_interval = read_u16_be(section, 23)?;
        }
        _ => {}
    }

    Ok(metadata)
}

fn parse_section6_bitmap(
    section: &[u8],
    total_points: Option<usize>,
) -> Result<Option<Vec<bool>>, String> {
    if section.len() < 6 {
        return Err("section 6 too short".to_string());
    }

    match section[5] {
        255 => Ok(None),
        0 => {
            let mut bits = Vec::with_capacity((section.len() - 6) * 8);
            for &byte in &section[6..] {
                for shift in (0..8).rev() {
                    bits.push(((byte >> shift) & 1) == 1);
                }
            }
            if let Some(total_points) = total_points {
                bits.truncate(total_points);
            }
            Ok(Some(bits))
        }
        254 => Err("bitmap reuse indicator 254 is not implemented yet".to_string()),
        indicator => Err(format!("unsupported bitmap indicator {indicator}")),
    }
}

fn parse_section7_payload(section: &[u8]) -> &[u8] {
    if section.len() <= 5 {
        &[]
    } else {
        &section[5..]
    }
}

fn forecast_summary(value: Option<u32>, unit: Option<&str>) -> String {
    match (value, unit) {
        (Some(value), Some(unit)) => format!("forecast={value} {unit}"),
        (Some(value), None) => format!("forecast={value}"),
        _ => "forecast=unknown".to_string(),
    }
}

fn format_level(level_type: u8, level_value: f64) -> String {
    match level_type {
        1 => "surface".to_string(),
        10 | 200 => "entire atmosphere".to_string(),
        100 => format!("{:.0} hPa", level_value),
        103 => format!("{:.0} m above ground", level_value),
        107 => format!("{:.0} K isentropic", level_value),
        109 => format!("{:.0} PVU", level_value),
        220 => "planetary boundary layer".to_string(),
        _ => format!("level_type_{level_type}:{level_value}"),
    }
}

fn format_json_level(value: &Value) -> String {
    if let Some(levelist) = value.get("levelist").and_then(Value::as_str) {
        if let Some(level_type) = value.get("levtype").and_then(Value::as_str) {
            return match level_type {
                "pl" => format!("{levelist} hPa"),
                "sfc" => "surface".to_string(),
                "sol" => format!("soil level {levelist}"),
                other => format!("{levelist} {other}"),
            };
        }
        return levelist.to_string();
    }

    value
        .get("levtype")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| "unknown".to_string())
}

fn format_compact_reference_time(raw: &str) -> Option<String> {
    if raw.len() == 8 {
        Some(format!(
            "{}-{}-{}T00:00:00Z",
            &raw[0..4],
            &raw[4..6],
            &raw[6..8]
        ))
    } else if raw.len() == 10 {
        Some(format!(
            "{}-{}-{}T{}:00:00Z",
            &raw[0..4],
            &raw[4..6],
            &raw[6..8],
            &raw[8..10]
        ))
    } else {
        None
    }
}

fn parse_json_step_hours(raw: Option<&str>) -> Option<u32> {
    raw.and_then(|raw| raw.trim_end_matches('h').parse::<u32>().ok())
}

fn time_unit_name(code: u8) -> &'static str {
    match code {
        0 => "minute",
        1 => "hour",
        2 => "day",
        10 => "3 hours",
        11 => "6 hours",
        12 => "12 hours",
        13 => "second",
        _ => "unit",
    }
}

fn grib2_short_name(
    discipline: u8,
    category: u8,
    number: u8,
    level_type: u8,
    level_value: f64,
) -> &'static str {
    let base = match (discipline, category, number) {
        (0, 0, 0) => "t",
        (0, 0, 6) => "d",
        (0, 1, 0) => "q",
        (0, 1, 1) => "r",
        (0, 1, 3) => "pwat",
        (0, 1, 7) => "prate",
        (0, 1, 8) => "tp",
        (0, 2, 0) => "wdir",
        (0, 2, 1) => "ws",
        (0, 2, 2) => "u",
        (0, 2, 3) => "v",
        (0, 2, 8) => "w",
        (0, 2, 10) => "absv",
        (0, 2, 12) => "vo",
        (0, 2, 22) => "gust",
        (0, 3, 0) => "sp",
        (0, 3, 1) => "msl",
        (0, 3, 5) => "gh",
        (0, 6, 1) => "tcc",
        (0, 7, 6) => "cape",
        (0, 7, 7) => "cin",
        (0, 7, 8) => "hlcy",
        (0, 19, 0) => "vis",
        _ => "unknown",
    };

    match (base, level_type, level_value as u32) {
        ("t", 103, 2) => "2t",
        ("d", 103, 2) => "2d",
        ("u", 103, 10) => "10u",
        ("v", 103, 10) => "10v",
        ("u", 103, 100) => "100u",
        ("v", 103, 100) => "100v",
        _ => base,
    }
}

fn grib2_parameter_name(discipline: u8, category: u8, number: u8) -> &'static str {
    match (discipline, category, number) {
        (0, 0, 0) => "Temperature",
        (0, 0, 6) => "Dewpoint Temperature",
        (0, 1, 0) => "Specific Humidity",
        (0, 1, 1) => "Relative Humidity",
        (0, 1, 3) => "Precipitable Water",
        (0, 1, 7) => "Precipitation Rate",
        (0, 1, 8) => "Total Precipitation",
        (0, 2, 0) => "Wind Direction",
        (0, 2, 1) => "Wind Speed",
        (0, 2, 2) => "U-Component of Wind",
        (0, 2, 3) => "V-Component of Wind",
        (0, 2, 8) => "Vertical Velocity (Pressure)",
        (0, 2, 10) => "Absolute Vorticity",
        (0, 2, 12) => "Relative Vorticity",
        (0, 2, 22) => "Wind Speed (Gust)",
        (0, 3, 0) => "Pressure",
        (0, 3, 1) => "Pressure Reduced to MSL",
        (0, 3, 5) => "Geopotential Height",
        (0, 6, 1) => "Total Cloud Cover",
        (0, 7, 6) => "Convective Available Potential Energy",
        (0, 7, 7) => "Convective Inhibition",
        (0, 7, 8) => "Storm Relative Helicity",
        (0, 19, 0) => "Visibility",
        _ => "Unknown",
    }
}

fn grib2_parameter_units(discipline: u8, category: u8, number: u8) -> &'static str {
    match (discipline, category, number) {
        (0, 0, 0) | (0, 0, 6) => "K",
        (0, 1, 0) => "kg/kg",
        (0, 1, 1) => "%",
        (0, 1, 3) => "kg/m^2",
        (0, 1, 7) => "kg/m^2/s",
        (0, 1, 8) => "kg/m^2",
        (0, 2, 0) => "degree",
        (0, 2, 1) | (0, 2, 2) | (0, 2, 3) | (0, 2, 22) => "m/s",
        (0, 2, 8) => "Pa/s",
        (0, 2, 10) | (0, 2, 12) => "1/s",
        (0, 3, 0) | (0, 3, 1) => "Pa",
        (0, 3, 5) => "gpm",
        (0, 6, 1) => "%",
        (0, 7, 6) | (0, 7, 7) => "J/kg",
        (0, 7, 8) => "m^2/s^2",
        (0, 19, 0) => "m",
        _ => "?",
    }
}

fn data_representation_template_name(template: u16) -> &'static str {
    match template {
        0 => "grid point data - simple packing",
        2 => "grid point data - complex packing",
        3 => "grid point data - complex packing with spatial differencing",
        4 => "grid point data - IEEE floating point",
        40 => "grid point data - JPEG 2000 code stream",
        41 => "grid point data - PNG",
        42 => "grid point data - CCSDS/AEC",
        _ => "unknown",
    }
}

fn grib1_short_name(center: u8, _table_version: u8, parameter: u8) -> &'static str {
    match (center, parameter) {
        (98, 129) => "gh",
        (98, 130) => "t",
        (98, 131) => "u",
        (98, 132) => "v",
        (98, 133) => "q",
        (98, 134) => "sp",
        (98, 135) => "w",
        (98, 151) => "msl",
        (98, 157) => "r",
        (98, 164) => "tcc",
        (98, 165) => "10u",
        (98, 166) => "10v",
        (98, 167) => "2t",
        (98, 168) => "2d",
        (98, 172) => "lsm",
        (98, 228) => "tp",
        (_, 11) => "t",
        (_, 33) => "u",
        (_, 34) => "v",
        (_, 52) => "r",
        _ => "grib1",
    }
}

fn grib1_parameter_name(center: u8, _table_version: u8, parameter: u8) -> &'static str {
    match (center, parameter) {
        (98, 129) => "Geopotential",
        (98, 130) => "Temperature",
        (98, 131) => "U Component of Wind",
        (98, 132) => "V Component of Wind",
        (98, 133) => "Specific Humidity",
        (98, 134) => "Surface Pressure",
        (98, 135) => "Vertical Velocity",
        (98, 151) => "Mean Sea Level Pressure",
        (98, 157) => "Relative Humidity",
        (98, 164) => "Total Cloud Cover",
        (98, 165) => "10 metre U wind component",
        (98, 166) => "10 metre V wind component",
        (98, 167) => "2 metre temperature",
        (98, 168) => "2 metre dewpoint temperature",
        (98, 172) => "Land-sea mask",
        (98, 228) => "Total precipitation",
        _ => "GRIB1 parameter",
    }
}

fn grib1_parameter_units(center: u8, _table_version: u8, parameter: u8) -> &'static str {
    match (center, parameter) {
        (98, 129) => "m^2/s^2",
        (98, 130) | (98, 167) | (98, 168) => "K",
        (98, 131) | (98, 132) | (98, 135) | (98, 165) | (98, 166) => "m/s",
        (98, 133) => "kg/kg",
        (98, 134) | (98, 151) => "Pa",
        (98, 157) | (98, 164) => "%",
        (98, 172) => "0-1",
        (98, 228) => "m",
        _ => "?",
    }
}

fn format_grib1_level(level_type: u8, level_value: u16) -> String {
    match level_type {
        1 => "surface".to_string(),
        100 => format!("{level_value} hPa"),
        105 => format!("{level_value} m above ground"),
        109 => "hybrid level".to_string(),
        _ => format!("level_type_{level_type}:{level_value}"),
    }
}

fn decode_grib1_message(
    message: &[u8],
    descriptor: MessageDescriptor,
) -> Result<DecodedField, String> {
    if message.len() < 8 || &message[0..4] != b"GRIB" {
        return Err("input is not a GRIB1 message".to_string());
    }

    let pds_length = read_u24_be(message, 8)? as usize;
    if message.len() < 8 + pds_length {
        return Err("truncated GRIB1 PDS".to_string());
    }
    let pds = &message[8..8 + pds_length];
    let product = parse_grib1_pds(pds)?;
    let mut cursor = 8 + pds_length;

    let grid = if product.has_gds {
        let gds_length = read_u24_be(message, cursor)? as usize;
        if message.len() < cursor + gds_length {
            return Err("truncated GRIB1 GDS".to_string());
        }
        let parsed = parse_grib1_gds(&message[cursor..cursor + gds_length])?;
        cursor += gds_length;
        parsed
    } else {
        return Err("GRIB1 decode currently requires a GDS".to_string());
    };

    let bitmap = if product.has_bms {
        let bms_length = read_u24_be(message, cursor)? as usize;
        if message.len() < cursor + bms_length {
            return Err("truncated GRIB1 BMS".to_string());
        }
        let parsed = parse_grib1_bms(&message[cursor..cursor + bms_length], grid.nx * grid.ny)?;
        cursor += bms_length;
        Some(parsed)
    } else {
        None
    };

    let bds_length = read_u24_be(message, cursor)? as usize;
    if message.len() < cursor + bds_length {
        return Err("truncated GRIB1 BDS".to_string());
    }
    let bds = &message[cursor..cursor + bds_length];
    let binary = parse_grib1_bds_metadata(bds)?;
    let packed_points = bitmap
        .as_ref()
        .map(|bits| bits.iter().filter(|flag| **flag).count())
        .unwrap_or(grid.nx * grid.ny);
    let mut values = unpack_grib1_simple(bds, binary, product.decimal_scale, packed_points)?;
    if let Some(bitmap) = bitmap {
        values = apply_bitmap(values, &bitmap);
    }
    if values.len() != grid.nx * grid.ny {
        return Err(format!(
            "decoded GRIB1 value count {} does not match grid dimensions {}x{}",
            values.len(),
            grid.nx,
            grid.ny
        ));
    }

    let missing_count = values.iter().filter(|value| value.is_nan()).count();
    let grid_spec = GridSpec {
        projection: grid.projection,
        shape: GridShape {
            ny: grid.ny,
            nx: grid.nx,
        },
        dx_m: None,
        dy_m: None,
    };
    // Flip rows if scan mode indicates bottom-to-top (bit 2 / 0x40)
    if grid.scan_mode & 0x40 != 0 {
        flip_rows(&mut values, grid.nx, grid.ny);
    }

    // Flip rows if scan mode indicates bottom-to-top (bit 2 / 0x40)
    if grid.scan_mode & 0x40 != 0 {
        flip_rows(&mut values, grid.nx, grid.ny);
    }

    let (x_axis, y_axis) = build_axes(&grid);
    Ok(DecodedField {
        descriptor,
        grid: Grid2D::new(grid.nx, grid.ny, values),
        grid_spec,
        x_axis,
        y_axis,
        missing_count,
    })
}

fn parse_grib1_bms(section: &[u8], total_points: usize) -> Result<Vec<bool>, String> {
    if section.len() < 6 {
        return Err("GRIB1 BMS too short".to_string());
    }
    let predefined_bitmap = read_u16_be(section, 4)?;
    if predefined_bitmap != 0 {
        return Err(format!(
            "GRIB1 predefined bitmap {predefined_bitmap} is not implemented yet"
        ));
    }
    let unused_bits = section[3] as usize;
    let mut bits = Vec::with_capacity(total_points);
    let total_bits = (section.len() - 6) * 8;
    let payload_bits = total_bits.saturating_sub(unused_bits);
    for bit_index in 0..payload_bits.min(total_points) {
        let byte = section[6 + bit_index / 8];
        let shift = 7 - (bit_index % 8);
        bits.push(((byte >> shift) & 1) == 1);
    }
    while bits.len() < total_points {
        bits.push(false);
    }
    Ok(bits)
}

fn parse_grib1_bds_metadata(section: &[u8]) -> Result<Grib1BinaryMetadata, String> {
    if section.len() < 11 {
        return Err("GRIB1 BDS too short".to_string());
    }
    if section[3] & 0xF0 != 0 {
        return Err(format!(
            "GRIB1 BDS flags {:08b} are not implemented beyond simple packing",
            section[3]
        ));
    }
    Ok(Grib1BinaryMetadata {
        binary_scale: read_sign_magnitude_i16_be(section, 4)?,
        reference_value: read_ibm_f32_be(section, 6)?,
        bits_per_value: section[10],
        unused_bits: section[3] & 0x0F,
    })
}

fn unpack_grib1_simple(
    section: &[u8],
    metadata: Grib1BinaryMetadata,
    decimal_scale: i16,
    num_points: usize,
) -> Result<Vec<f64>, String> {
    let payload = section
        .get(11..)
        .ok_or_else(|| "GRIB1 BDS missing payload".to_string())?;
    if metadata.bits_per_value == 0 {
        let value = apply_grib1_scaling(
            0,
            metadata.reference_value,
            metadata.binary_scale,
            decimal_scale,
        );
        return Ok(vec![value; num_points]);
    }
    let total_bits = payload.len() * 8;
    let payload_bits = total_bits.saturating_sub(metadata.unused_bits as usize);
    if payload_bits < num_points * metadata.bits_per_value as usize {
        return Err(format!(
            "GRIB1 payload has {payload_bits} usable bits but {} are required",
            num_points * metadata.bits_per_value as usize
        ));
    }

    let mut reader = BitReader::new(payload);
    let mut values = Vec::with_capacity(num_points);
    for _ in 0..num_points {
        let raw = reader.read_bits(metadata.bits_per_value as usize)?;
        values.push(apply_grib1_scaling(
            raw,
            metadata.reference_value,
            metadata.binary_scale,
            decimal_scale,
        ));
    }
    Ok(values)
}

fn apply_grib1_scaling(
    raw: u64,
    reference_value: f32,
    binary_scale: i16,
    decimal_scale: i16,
) -> f64 {
    let scaled = reference_value as f64 + raw as f64 * 2_f64.powi(binary_scale as i32);
    scaled / 10_f64.powi(decimal_scale as i32)
}

fn decode_grib2_message(
    message: &[u8],
    descriptor: MessageDescriptor,
) -> Result<DecodedField, String> {
    if message.len() < 20 || &message[0..4] != b"GRIB" {
        return Err("input is not a GRIB message".to_string());
    }

    let mut cursor = 16usize;
    let mut grid = None;
    let mut data_representation = None;
    let mut bitmap = None;
    let mut data_section = None;

    while cursor + 5 <= message.len() {
        if cursor + 4 <= message.len() && &message[cursor..cursor + 4] == b"7777" {
            break;
        }

        let section_length = read_u32_be(message, cursor)? as usize;
        let section_number = message[cursor + 4];
        if section_length < 5 || cursor + section_length > message.len() {
            return Err(format!(
                "invalid GRIB2 section {section_number} length {section_length}"
            ));
        }

        let section = &message[cursor..cursor + section_length];
        match section_number {
            3 => grid = Some(parse_section3_metadata(section)?),
            5 => data_representation = Some(parse_section5_metadata(section)?),
            6 => {
                let total_points = grid.as_ref().map(|meta| meta.nx * meta.ny);
                bitmap = parse_section6_bitmap(section, total_points)?;
            }
            7 => data_section = Some(parse_section7_payload(section)),
            _ => {}
        }

        cursor += section_length;
    }

    let grid = grid.ok_or_else(|| "GRIB2 message missing section 3".to_string())?;
    let data_representation =
        data_representation.ok_or_else(|| "GRIB2 message missing section 5".to_string())?;
    let data_section = data_section.ok_or_else(|| "GRIB2 message missing section 7".to_string())?;

    let packed_value_count = bitmap
        .as_ref()
        .map(|values| values.iter().filter(|flag| **flag).count())
        .unwrap_or_else(|| {
            if data_representation.num_points > 0 {
                data_representation.num_points
            } else {
                grid.nx * grid.ny
            }
        });

    let mut values = match data_representation.template {
        0 => unpack_simple(data_section, data_representation, packed_value_count)?,
        2 | 3 => unpack_complex(data_section, data_representation, packed_value_count)?,
        4 => unpack_ieee(data_section, data_representation, packed_value_count)?,
        40 => unpack_jpeg2000(data_section, data_representation, packed_value_count)?,
        42 => unpack_ccsds(data_section, data_representation, packed_value_count)?,
        template => {
            return Err(format!(
                "GRIB2 data representation template {template} ({}) is not implemented yet",
                data_representation_template_name(template)
            ))
        }
    };

    if let Some(bitmap) = bitmap {
        values = apply_bitmap(values, &bitmap);
    }

    if values.len() != grid.nx * grid.ny {
        return Err(format!(
            "decoded value count {} does not match grid dimensions {}x{}",
            values.len(),
            grid.nx,
            grid.ny
        ));
    }

    let missing_count = values.iter().filter(|value| value.is_nan()).count();
    let grid_spec = GridSpec {
        projection: grid.projection,
        shape: GridShape {
            ny: grid.ny,
            nx: grid.nx,
        },
        dx_m: if grid.projection == ProjectionKind::LatLon {
            None
        } else {
            grid.dx
        },
        dy_m: if grid.projection == ProjectionKind::LatLon {
            None
        } else {
            grid.dy
        },
    };
    // Flip rows if scan mode indicates bottom-to-top (bit 2 / 0x40)
    if grid.scan_mode & 0x40 != 0 {
        flip_rows(&mut values, grid.nx, grid.ny);
    }

    let (x_axis, y_axis) = build_axes(&grid);
    Ok(DecodedField {
        descriptor,
        grid: Grid2D::new(grid.nx, grid.ny, values),
        grid_spec,
        x_axis,
        y_axis,
        missing_count,
    })
}

fn unpack_simple(
    data: &[u8],
    metadata: DataRepresentationMetadata,
    num_points: usize,
) -> Result<Vec<f64>, String> {
    if metadata.bits_per_value == 0 {
        let value = apply_scaling(&[0u64], metadata)[0];
        return Ok(vec![value; num_points]);
    }

    let mut reader = BitReader::new(data);
    let mut raw = Vec::with_capacity(num_points);
    for _ in 0..num_points {
        raw.push(reader.read_bits(metadata.bits_per_value as usize)?);
    }
    Ok(apply_scaling(&raw, metadata))
}

fn unpack_ieee(
    data: &[u8],
    metadata: DataRepresentationMetadata,
    num_points: usize,
) -> Result<Vec<f64>, String> {
    match metadata.bits_per_value {
        32 => {
            if data.len() < num_points * 4 {
                return Err(format!(
                    "section 7 has {} bytes but {} are required for {} f32 values",
                    data.len(),
                    num_points * 4,
                    num_points
                ));
            }
            let mut values = Vec::with_capacity(num_points);
            for chunk in data.chunks_exact(4).take(num_points) {
                values.push(f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64);
            }
            Ok(values)
        }
        64 => {
            if data.len() < num_points * 8 {
                return Err(format!(
                    "section 7 has {} bytes but {} are required for {} f64 values",
                    data.len(),
                    num_points * 8,
                    num_points
                ));
            }
            let mut values = Vec::with_capacity(num_points);
            for chunk in data.chunks_exact(8).take(num_points) {
                values.push(f64::from_be_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]));
            }
            Ok(values)
        }
        other => Err(format!(
            "GRIB2 IEEE float template expects 32 or 64 bits per value, got {other}"
        )),
    }
}

fn unpack_complex(
    data: &[u8],
    metadata: DataRepresentationMetadata,
    num_points: usize,
) -> Result<Vec<f64>, String> {
    if metadata.number_of_groups == 0 {
        return Err("complex packing metadata has zero groups".to_string());
    }

    let mut cursor = 0usize;
    let spatial = if metadata.template == 3 {
        let spatial = parse_spatial_differencing_metadata(&data[cursor..], metadata)?;
        cursor += (spatial.order + 1) * metadata.extra_descriptor_octets as usize;
        Some(spatial)
    } else {
        None
    };

    let mut reader = BitReader::new(&data[cursor..]);
    let group_references = read_packed_values(
        &mut reader,
        metadata.bits_per_value as usize,
        metadata.number_of_groups,
    )?;
    reader.align_to_octet();

    let group_width_deltas = read_packed_values(
        &mut reader,
        metadata.group_width_bits as usize,
        metadata.number_of_groups,
    )?;
    reader.align_to_octet();

    let group_length_deltas = read_packed_values(
        &mut reader,
        metadata.group_length_bits as usize,
        metadata.number_of_groups,
    )?;
    reader.align_to_octet();

    let group_widths = group_width_deltas
        .into_iter()
        .map(|delta| {
            metadata
                .group_width_reference
                .checked_add(delta as u8)
                .ok_or_else(|| "group width overflow".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut group_lengths = group_length_deltas
        .into_iter()
        .map(|delta| {
            metadata.group_length_reference + delta as u32 * metadata.group_length_increment as u32
        })
        .collect::<Vec<_>>();
    if let Some(last) = group_lengths.last_mut() {
        *last = metadata.last_group_length;
    }

    let mut raw = Vec::with_capacity(num_points);
    for group_index in 0..metadata.number_of_groups {
        let group_reference = group_references[group_index];
        let group_width = group_widths[group_index] as usize;
        let group_length = group_lengths[group_index] as usize;

        if group_width == 0 {
            let constant = if is_missing_code(
                group_reference,
                metadata.bits_per_value as usize,
                metadata.missing_value_management,
            ) {
                None
            } else {
                Some(group_reference as i64)
            };
            raw.extend(std::iter::repeat_n(constant, group_length));
            continue;
        }

        for _ in 0..group_length {
            let value = reader.read_bits(group_width)?;
            if is_missing_code(value, group_width, metadata.missing_value_management) {
                raw.push(None);
            } else {
                raw.push(Some(group_reference as i64 + value as i64));
            }
        }
    }

    if raw.len() != num_points {
        return Err(format!(
            "complex packed value count {} does not match expected {}",
            raw.len(),
            num_points
        ));
    }

    if let Some(spatial) = spatial {
        raw = undo_spatial_differencing(raw, &spatial)?;
    }

    Ok(apply_scaling_with_missing(&raw, metadata))
}

fn unpack_jpeg2000(
    data: &[u8],
    metadata: DataRepresentationMetadata,
    num_points: usize,
) -> Result<Vec<f64>, String> {
    let settings = Jpeg2000DecodeSettings {
        resolve_palette_indices: false,
        strict: false,
        target_resolution: None,
    };
    let image = Jpeg2000Image::new(data, &settings)
        .map_err(|err| format!("jpeg2000 codestream parse failed: {err}"))?;
    let bitmap = image
        .decode_native()
        .map_err(|err| format!("jpeg2000 codestream decode failed: {err}"))?;

    if bitmap.num_components != 1 {
        return Err(format!(
            "jpeg2000 GRIB decode expects 1 component, got {}",
            bitmap.num_components
        ));
    }

    let raw = raw_samples_from_bytes(
        &bitmap.data,
        metadata.bits_per_value,
        bitmap.bytes_per_sample as usize,
        Endianness::Little,
        false,
        num_points,
    )?;
    Ok(apply_scaling(&raw, metadata))
}

fn unpack_ccsds(
    data: &[u8],
    metadata: DataRepresentationMetadata,
    num_points: usize,
) -> Result<Vec<f64>, String> {
    let flags = flags_from_grib2_ccsds_flags(metadata.ccsds_flags);
    let params = AecParams::new(
        metadata.bits_per_value,
        metadata.ccsds_block_size as u32,
        metadata.ccsds_reference_sample_interval as u32,
        flags,
    );
    let bytes = decode_aec(data, params, num_points)
        .map_err(|err| format!("ccsds/aec decode failed: {err}"))?;
    let raw = raw_samples_from_bytes(
        &bytes,
        metadata.bits_per_value,
        bytes_per_sample(metadata.bits_per_value, flags) as usize,
        if flags.contains(AecFlags::MSB) {
            Endianness::Big
        } else {
            Endianness::Little
        },
        flags.contains(AecFlags::DATA_SIGNED),
        num_points,
    )?;
    Ok(apply_scaling(&raw, metadata))
}

fn apply_scaling(raw: &[u64], metadata: DataRepresentationMetadata) -> Vec<f64> {
    raw.iter()
        .map(|value| scale_raw_value(*value as i64, metadata))
        .collect()
}

fn apply_scaling_with_missing(
    raw: &[Option<i64>],
    metadata: DataRepresentationMetadata,
) -> Vec<f64> {
    raw.iter()
        .map(|value| match value {
            Some(value) => scale_raw_value(*value, metadata),
            None => f64::NAN,
        })
        .collect()
}

fn scale_raw_value(raw: i64, metadata: DataRepresentationMetadata) -> f64 {
    let reference = metadata.reference_value as f64;
    let binary_scale = 2.0_f64.powi(metadata.binary_scale as i32);
    let decimal_scale = 10.0_f64.powi(-(metadata.decimal_scale as i32));
    (reference + (raw as f64) * binary_scale) * decimal_scale
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endianness {
    Big,
    Little,
}

fn bytes_per_sample(bits_per_value: u8, flags: AecFlags) -> u8 {
    match bits_per_value {
        1..=8 => 1,
        9..=16 => 2,
        17..=24 => {
            if flags.contains(AecFlags::DATA_3BYTE) {
                3
            } else {
                4
            }
        }
        25..=32 => 4,
        _ => 0,
    }
}

fn raw_samples_from_bytes(
    data: &[u8],
    bits_per_value: u8,
    bytes_per_sample: usize,
    endianness: Endianness,
    signed: bool,
    num_points: usize,
) -> Result<Vec<u64>, String> {
    if bytes_per_sample == 0 {
        return Err("bytes_per_sample must be > 0".to_string());
    }
    let expected = num_points
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| "sample byte count overflow".to_string())?;
    if data.len() < expected {
        return Err(format!(
            "decoded payload has {} bytes but {} are required for {} samples",
            data.len(),
            expected,
            num_points
        ));
    }

    let mut out = Vec::with_capacity(num_points);
    for chunk in data[..expected].chunks_exact(bytes_per_sample) {
        let unsigned = match endianness {
            Endianness::Big => chunk
                .iter()
                .fold(0u64, |acc, byte| (acc << 8) | (*byte as u64)),
            Endianness::Little => chunk
                .iter()
                .rev()
                .fold(0u64, |acc, byte| (acc << 8) | (*byte as u64)),
        };
        if signed {
            out.push(sign_extend_u64(unsigned, bits_per_value as usize)? as u64);
        } else {
            out.push(unsigned);
        }
    }
    Ok(out)
}

fn sign_extend_u64(value: u64, bits: usize) -> Result<i64, String> {
    if bits == 0 || bits > 63 {
        return Err(format!("cannot sign-extend width {bits}"));
    }
    let sign_bit = 1u64 << (bits - 1);
    let mask = (1u64 << bits) - 1;
    let value = value & mask;
    if (value & sign_bit) == 0 {
        Ok(value as i64)
    } else {
        Ok((value as i64) - ((mask + 1) as i64))
    }
}

fn apply_bitmap(values: Vec<f64>, bitmap: &[bool]) -> Vec<f64> {
    let mut result = vec![f64::NAN; bitmap.len()];
    let mut value_index = 0usize;
    for (idx, is_present) in bitmap.iter().enumerate() {
        if *is_present {
            if let Some(value) = values.get(value_index) {
                result[idx] = *value;
                value_index += 1;
            }
        }
    }
    result
}

fn build_axes(grid: &GridSectionMetadata) -> (Option<CoordinateAxis>, Option<CoordinateAxis>) {
    match grid.projection {
        ProjectionKind::LatLon => match (grid.lon1, grid.lon2, grid.lat1, grid.lat2) {
            (Some(lon1), Some(lon2), Some(lat1), Some(lat2)) => {
                let x_start = if lon1 >= 180.0 && lon2 < lon1 {
                    lon1 - 360.0
                } else {
                    lon1
                };
                let x_values = if grid.nx <= 1 {
                    vec![x_start]
                } else if let Some(dx) = grid.dx {
                    (0..grid.nx).map(|idx| x_start + dx * idx as f64).collect()
                } else {
                    let step = (lon2 - x_start) / (grid.nx as f64 - 1.0);
                    (0..grid.nx)
                        .map(|idx| x_start + step * idx as f64)
                        .collect()
                };
                let north = lat1.max(lat2);
                let south = lat1.min(lat2);
                let y_values = if grid.ny <= 1 {
                    vec![north]
                } else if let Some(dy) = grid.dy {
                    (0..grid.ny).map(|idx| north - dy * idx as f64).collect()
                } else {
                    let step = (south - north) / (grid.ny as f64 - 1.0);
                    (0..grid.ny).map(|idx| north + step * idx as f64).collect()
                };
                (
                    Some(CoordinateAxis {
                        name: "longitude".to_string(),
                        units: "degrees_east".to_string(),
                        values: x_values,
                    }),
                    Some(CoordinateAxis {
                        name: "latitude".to_string(),
                        units: "degrees_north".to_string(),
                        values: y_values,
                    }),
                )
            }
            _ => (None, None),
        },
        _ => (None, None),
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    fn read_bits(&mut self, count: usize) -> Result<u64, String> {
        if count == 0 {
            return Ok(0);
        }
        if count > 64 {
            return Err("BitReader only supports reads up to 64 bits".to_string());
        }
        if self.bit_pos + count > self.data.len() * 8 {
            return Err("not enough packed bits remaining".to_string());
        }

        let mut out = 0u64;
        for _ in 0..count {
            let byte_offset = self.bit_pos / 8;
            let bit_offset = 7 - (self.bit_pos % 8);
            let bit = (self.data[byte_offset] >> bit_offset) & 1;
            out = (out << 1) | (bit as u64);
            self.bit_pos += 1;
        }
        Ok(out)
    }

    fn align_to_octet(&mut self) {
        let remainder = self.bit_pos % 8;
        if remainder != 0 {
            self.bit_pos += 8 - remainder;
        }
    }
}

fn read_u16_be(data: &[u8], offset: usize) -> Result<u16, String> {
    let bytes = data
        .get(offset..offset + 2)
        .ok_or_else(|| format!("read_u16_be out of range at offset {offset}"))?;
    Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
}

fn read_u24_be(data: &[u8], offset: usize) -> Result<u32, String> {
    let bytes = data
        .get(offset..offset + 3)
        .ok_or_else(|| format!("read_u24_be out of range at offset {offset}"))?;
    Ok(((bytes[0] as u32) << 16) | ((bytes[1] as u32) << 8) | (bytes[2] as u32))
}

fn read_sign_magnitude_i16_be(data: &[u8], offset: usize) -> Result<i16, String> {
    let bytes = data
        .get(offset..offset + 2)
        .ok_or_else(|| format!("read_sign_magnitude_i16_be out of range at offset {offset}"))?;
    Ok(sign_magnitude_to_i64(u16::from_be_bytes([bytes[0], bytes[1]]) as u64, 16) as i16)
}

fn read_signed_i24_sign_magnitude(data: &[u8], offset: usize) -> Result<i32, String> {
    let raw = read_u24_be(data, offset)? as u64;
    Ok(sign_magnitude_to_i64(raw, 24) as i32)
}

fn parse_spatial_differencing_metadata(
    data: &[u8],
    metadata: DataRepresentationMetadata,
) -> Result<SpatialDifferencingMetadata, String> {
    let order = metadata.spatial_differencing_order as usize;
    let octets = metadata.extra_descriptor_octets as usize;
    if !(order == 1 || order == 2) {
        return Err(format!(
            "unsupported spatial differencing order {}",
            metadata.spatial_differencing_order
        ));
    }
    if octets == 0 {
        return Err("spatial differencing metadata has zero extra descriptor octets".to_string());
    }

    let needed = (order + 1) * octets;
    if data.len() < needed {
        return Err(format!(
            "section 7 too short for spatial differencing descriptors: need {needed} bytes"
        ));
    }

    let mut initial_values = Vec::with_capacity(order);
    for idx in 0..order {
        let start = idx * octets;
        let end = start + octets;
        initial_values.push(read_sign_magnitude_bytes(&data[start..end])?);
    }
    let minimum = read_sign_magnitude_bytes(&data[order * octets..(order + 1) * octets])?;

    Ok(SpatialDifferencingMetadata {
        order,
        initial_values,
        minimum,
    })
}

fn read_packed_values(
    reader: &mut BitReader<'_>,
    width_bits: usize,
    count: usize,
) -> Result<Vec<u64>, String> {
    if width_bits == 0 {
        return Ok(vec![0; count]);
    }

    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(reader.read_bits(width_bits)?);
    }
    Ok(values)
}

fn is_missing_code(value: u64, width_bits: usize, management: u8) -> bool {
    match management {
        0 => false,
        1 => value == all_ones(width_bits),
        2 => {
            let all = all_ones(width_bits);
            value == all || value == all.saturating_sub(1)
        }
        _ => false,
    }
}

fn all_ones(width_bits: usize) -> u64 {
    if width_bits == 0 {
        0
    } else if width_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << width_bits) - 1
    }
}

fn undo_spatial_differencing(
    values: Vec<Option<i64>>,
    spatial: &SpatialDifferencingMetadata,
) -> Result<Vec<Option<i64>>, String> {
    match spatial.order {
        1 => undo_first_order_spatial_differencing(values, spatial),
        2 => undo_second_order_spatial_differencing(values, spatial),
        order => Err(format!("unsupported spatial differencing order {order}")),
    }
}

fn undo_first_order_spatial_differencing(
    values: Vec<Option<i64>>,
    spatial: &SpatialDifferencingMetadata,
) -> Result<Vec<Option<i64>>, String> {
    let first = *spatial
        .initial_values
        .first()
        .ok_or_else(|| "missing first initial value for order-1 differencing".to_string())?;
    let mut output = Vec::with_capacity(values.len());
    let mut previous = first;
    let mut initialized = false;

    for value in values {
        match value {
            Some(_value) if !initialized => {
                output.push(Some(first));
                initialized = true;
            }
            Some(value) => {
                let restored = value + spatial.minimum + previous;
                output.push(Some(restored));
                previous = restored;
            }
            None => output.push(None),
        }
    }

    Ok(output)
}

fn undo_second_order_spatial_differencing(
    values: Vec<Option<i64>>,
    spatial: &SpatialDifferencingMetadata,
) -> Result<Vec<Option<i64>>, String> {
    if spatial.initial_values.len() < 2 {
        return Err("missing initial values for order-2 differencing".to_string());
    }
    let first = spatial.initial_values[0];
    let second = spatial.initial_values[1];
    let mut output = Vec::with_capacity(values.len());
    let mut initialized = 0usize;
    let mut previous_value = second;
    let mut previous_first_difference = second - first;

    for value in values {
        match value {
            Some(_) if initialized == 0 => {
                output.push(Some(first));
                initialized = 1;
            }
            Some(_) if initialized == 1 => {
                output.push(Some(second));
                initialized = 2;
            }
            Some(value) => {
                let first_difference = value + spatial.minimum + previous_first_difference;
                let restored = first_difference + previous_value;
                output.push(Some(restored));
                previous_first_difference = first_difference;
                previous_value = restored;
            }
            None => output.push(None),
        }
    }

    Ok(output)
}

fn read_sign_magnitude_bytes(data: &[u8]) -> Result<i64, String> {
    if data.is_empty() || data.len() > 8 {
        return Err(format!(
            "sign-magnitude descriptor width must be between 1 and 8 bytes, got {}",
            data.len()
        ));
    }
    let mut raw = 0u64;
    for &byte in data {
        raw = (raw << 8) | byte as u64;
    }
    Ok(sign_magnitude_to_i64(raw, data.len() * 8))
}

fn sign_magnitude_to_i64(raw: u64, width_bits: usize) -> i64 {
    if width_bits == 0 {
        return 0;
    }
    let sign_mask = 1u64 << (width_bits - 1);
    let magnitude_mask = sign_mask - 1;
    let magnitude = (raw & magnitude_mask) as i64;
    if raw & sign_mask != 0 {
        -magnitude
    } else {
        magnitude
    }
}

fn read_signed_i32_sign_magnitude(data: &[u8], offset: usize) -> Result<i32, String> {
    let raw = read_u32_be(data, offset)?;
    let sign = (raw >> 31) & 1;
    let magnitude = raw & 0x7FFF_FFFF;
    if sign == 1 {
        Ok(-(magnitude as i32))
    } else {
        Ok(magnitude as i32)
    }
}

fn read_u32_be(data: &[u8], offset: usize) -> Result<u32, String> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or_else(|| format!("read_u32_be out of range at offset {offset}"))?;
    Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_u64_be(data: &[u8], offset: usize) -> Result<u64, String> {
    let bytes = data
        .get(offset..offset + 8)
        .ok_or_else(|| format!("read_u64_be out of range at offset {offset}"))?;
    Ok(u64::from_be_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

fn read_f32_be(data: &[u8], offset: usize) -> Result<f32, String> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or_else(|| format!("read_f32_be out of range at offset {offset}"))?;
    Ok(f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_ibm_f32_be(data: &[u8], offset: usize) -> Result<f32, String> {
    let raw = read_u32_be(data, offset)?;
    if raw == 0 {
        return Ok(0.0);
    }
    let sign = if raw & 0x8000_0000 != 0 { -1.0 } else { 1.0 };
    let exponent = ((raw >> 24) & 0x7f) as i32 - 64;
    let fraction = (raw & 0x00ff_ffff) as f64 / 16_f64.powi(6);
    Ok((sign * 16_f64.powi(exponent) * fraction) as f32)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn parses_inventory_and_searches_messages() {
        let engine = GribEngine::new();
        let text = "\
1:0:d=2026031618:TMP:2 m above ground:anl:
2:120:d=2026031618:RH:2 m above ground:anl:
3:240:d=2026031618:UGRD:10 m above ground:anl:
";
        let inventory = engine.parse_idx_text(text).expect("inventory should parse");

        assert_eq!(inventory.messages.len(), 3);
        assert_eq!(inventory.messages[0].length_bytes, 120);
        assert_eq!(inventory.messages[2].length_bytes, 0);

        let hits = engine.search(&inventory, "2 m above ground");
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn scans_synthetic_grib2_messages() {
        let engine = GribEngine::new();
        let mut bytes = build_test_grib2_message(SimplePackSpec {
            discipline: 0,
            category: 0,
            number: 0,
            level_type: 103,
            level_value: 2,
            forecast_time: 3,
            reference_value: 250.0,
            binary_scale: 0,
            decimal_scale: 0,
            bits_per_value: 8,
            payload: vec![1, 2, 3, 4],
        });
        bytes.extend_from_slice(&build_test_grib2_message(SimplePackSpec {
            discipline: 0,
            category: 2,
            number: 2,
            level_type: 103,
            level_value: 10,
            forecast_time: 6,
            reference_value: 0.0,
            binary_scale: 0,
            decimal_scale: 0,
            bits_per_value: 8,
            payload: vec![5, 6, 7, 8],
        }));

        let inventory = engine
            .scan_bytes(&bytes)
            .expect("synthetic scan should succeed");
        assert_eq!(inventory.messages.len(), 2);
        assert_eq!(inventory.messages[0].variable, "2t");
        assert_eq!(inventory.messages[0].level, "2 m above ground");
        assert_eq!(
            inventory.messages[0].reference_time.as_deref(),
            Some("2026-03-16T18:00:00Z")
        );
        assert_eq!(inventory.messages[0].grid_template, Some(0));
        assert_eq!(inventory.messages[0].nx, Some(2));
        assert_eq!(inventory.messages[1].variable, "10u");

        let hits = engine.search(&inventory, "wind");
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].parameter_name.as_deref(),
            Some("U-Component of Wind")
        );
    }

    #[test]
    fn decodes_simple_packed_grib2_message() {
        let engine = GribEngine::new();
        let bytes = build_test_grib2_message(SimplePackSpec {
            discipline: 0,
            category: 0,
            number: 0,
            level_type: 103,
            level_value: 2,
            forecast_time: 3,
            reference_value: 250.0,
            binary_scale: 0,
            decimal_scale: 0,
            bits_per_value: 8,
            payload: vec![1, 2, 3, 4],
        });

        let field = engine
            .decode_bytes_message(&bytes, 1)
            .expect("simple packed decode should succeed");
        assert_eq!(field.descriptor.variable, "2t");
        assert_eq!(field.grid.nx, 2);
        assert_eq!(field.grid.ny, 2);
        assert_eq!(field.grid.values, vec![251.0, 252.0, 253.0, 254.0]);
        assert_eq!(field.grid_spec.projection, ProjectionKind::LatLon);
        assert_eq!(
            field.x_axis.as_ref().map(|axis| axis.values.clone()),
            Some(vec![100.0, 101.0])
        );
        assert_eq!(
            field.y_axis.as_ref().map(|axis| axis.values.clone()),
            Some(vec![41.0, 40.0])
        );
        assert_eq!(field.min_mean_max(), Some((251.0, 252.5, 254.0)));
    }

    #[test]
    fn decodes_ieee_f32_grib2_message() {
        let engine = GribEngine::new();
        let payload = [1.0f32, 2.0f32, 3.0f32, 4.0f32]
            .into_iter()
            .flat_map(|value| value.to_be_bytes())
            .collect::<Vec<u8>>();
        let bytes = build_test_grib2_message(SimplePackSpec {
            discipline: 0,
            category: 2,
            number: 2,
            level_type: 103,
            level_value: 10,
            forecast_time: 6,
            reference_value: 0.0,
            binary_scale: 0,
            decimal_scale: 0,
            bits_per_value: 32,
            payload,
        });

        let field = engine
            .decode_bytes_message(&bytes, 1)
            .expect("ieee decode should succeed");
        assert_eq!(field.grid.values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rejects_short_lines() {
        let engine = GribEngine::new();
        let err = engine
            .parse_idx_text("1:0:broken")
            .expect_err("short line should fail");
        assert!(err.contains("invalid idx line"));
    }

    #[test]
    fn decodes_operational_hrrr_template3_if_present() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("examples")
            .join("hrrr_2t_subset.grib2");
        if !path.exists() {
            return;
        }

        let engine = GribEngine::new();
        let field = engine
            .decode_file_message(&path, 1)
            .expect("template-3 HRRR fixture should decode");
        let (min, mean, max) = field
            .min_mean_max()
            .expect("template-3 HRRR fixture should have finite values");

        assert_eq!(field.grid.nx, 1799);
        assert_eq!(field.grid.ny, 1059);
        assert_eq!(field.missing_count, 0);
        assert!((min - 242.0531).abs() < 0.01, "min={min}");
        assert!((mean - 279.9295).abs() < 0.02, "mean={mean}");
        assert!((max - 302.4906).abs() < 0.02, "max={max}");

        let preview = field
            .grid
            .values
            .iter()
            .take(8)
            .copied()
            .collect::<Vec<_>>();
        let expected = vec![
            293.3031, 293.3031, 293.3031, 293.3031, 293.3031, 293.2406, 293.2406, 293.2406,
        ];
        for (actual, expected) in preview.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 0.02,
                "actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn decodes_operational_rap_template40_if_present() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("examples")
            .join("rap_2t_subset.grib2");
        if !path.exists() {
            return;
        }

        let engine = GribEngine::new();
        let field = engine
            .decode_file_message(&path, 1)
            .expect("template-40 RAP fixture should decode");
        let (min, mean, max) = field
            .min_mean_max()
            .expect("template-40 RAP fixture should have finite values");

        assert_eq!(field.grid.nx, 451);
        assert_eq!(field.grid.ny, 337);
        assert_eq!(field.missing_count, 0);
        assert!((min - 235.9981).abs() < 0.02, "min={min}");
        assert!((mean - 277.2502).abs() < 0.02, "mean={mean}");
        assert!((max - 300.6856).abs() < 0.02, "max={max}");

        let preview = field
            .grid
            .values
            .iter()
            .take(8)
            .copied()
            .collect::<Vec<_>>();
        let expected = vec![
            297.2481, 297.2481, 297.1856, 297.1231, 297.0606, 297.0606, 297.0606, 296.9981,
        ];
        for (actual, expected) in preview.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 0.02,
                "actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn decodes_operational_ecmwf_template42_if_present() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("examples")
            .join("ecmwf_2t_subset.grib2");
        if !path.exists() {
            return;
        }

        let engine = GribEngine::new();
        let field = engine
            .decode_file_message(&path, 1)
            .expect("template-42 ECMWF fixture should decode");
        let (min, mean, max) = field
            .min_mean_max()
            .expect("template-42 ECMWF fixture should have finite values");

        assert_eq!(field.grid.nx, 1440);
        assert_eq!(field.grid.ny, 721);
        assert_eq!(field.missing_count, 0);
        assert!((min - 212.9503).abs() < 0.02, "min={min}");
        assert!((mean - 277.2192).abs() < 0.02, "mean={mean}");
        assert!((max - 315.3878).abs() < 0.02, "max={max}");

        let preview = field
            .grid
            .values
            .iter()
            .take(8)
            .copied()
            .collect::<Vec<_>>();
        let expected = vec![
            257.1066, 257.1066, 257.1066, 257.1066, 257.1066, 257.1066, 257.1066, 257.1066,
        ];
        for (actual, expected) in preview.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 0.02,
                "actual={actual} expected={expected}"
            );
        }

        let x_axis = field
            .x_axis
            .expect("ecmwf lat/lon grid should have longitude axis");
        let y_axis = field
            .y_axis
            .expect("ecmwf lat/lon grid should have latitude axis");
        assert_eq!(x_axis.values.len(), 1440);
        assert_eq!(y_axis.values.len(), 721);
        assert!((x_axis.values[0] - (-180.0)).abs() < 1e-6);
        assert!((x_axis.values[1] - (-179.75)).abs() < 1e-6);
        assert!((y_axis.values[0] - 90.0).abs() < 1e-6);
        assert!((y_axis.values[1] - 89.75).abs() < 1e-6);
    }

    #[test]
    fn decodes_operational_era5_surface_grib1_if_present() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("examples")
            .join("era5_2t_subset.grib");
        if !path.exists() {
            return;
        }

        let engine = GribEngine::new();
        let field = engine
            .decode_file_message(&path, 1)
            .expect("ERA5 surface GRIB1 fixture should decode");
        let (min, mean, max) = field
            .min_mean_max()
            .expect("ERA5 surface GRIB1 fixture should have finite values");

        assert_eq!(field.grid.nx, 281);
        assert_eq!(field.grid.ny, 141);
        assert_eq!(field.descriptor.variable, "2t");
        assert_eq!(field.descriptor.level, "surface");
        assert_eq!(field.missing_count, 0);
        assert!((min - 255.9097).abs() < 0.02, "min={min}");
        assert!((mean - 280.8969).abs() < 0.02, "mean={mean}");
        assert!((max - 301.1733).abs() < 0.02, "max={max}");

        let preview = field
            .grid
            .values
            .iter()
            .take(8)
            .copied()
            .collect::<Vec<_>>();
        let expected = vec![
            274.8218, 274.2827, 273.2827, 272.8491, 272.9761, 272.3296, 272.1128, 271.5952,
        ];
        for (actual, expected) in preview.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 0.02,
                "actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn decodes_operational_era5_pressure_grib1_if_present() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("examples")
            .join("era5_t850_subset.grib");
        if !path.exists() {
            return;
        }

        let engine = GribEngine::new();
        let field = engine
            .decode_file_message(&path, 1)
            .expect("ERA5 pressure GRIB1 fixture should decode");
        let (min, mean, max) = field
            .min_mean_max()
            .expect("ERA5 pressure GRIB1 fixture should have finite values");

        assert_eq!(field.grid.nx, 281);
        assert_eq!(field.grid.ny, 141);
        assert_eq!(field.descriptor.variable, "t");
        assert_eq!(field.descriptor.level, "850 hPa");
        assert_eq!(field.missing_count, 0);
        assert!((min - 253.5692).abs() < 0.02, "min={min}");
        assert!((mean - 275.6466).abs() < 0.02, "mean={mean}");
        assert!((max - 295.8519).abs() < 0.02, "max={max}");

        let preview = field
            .grid
            .values
            .iter()
            .take(8)
            .copied()
            .collect::<Vec<_>>();
        let expected = vec![
            268.7352, 268.8075, 268.6180, 268.3798, 268.2372, 268.1513, 268.1083, 268.1962,
        ];
        for (actual, expected) in preview.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 0.02,
                "actual={actual} expected={expected}"
            );
        }
    }

    #[derive(Debug, Clone)]
    struct SimplePackSpec {
        discipline: u8,
        category: u8,
        number: u8,
        level_type: u8,
        level_value: u32,
        forecast_time: u32,
        reference_value: f32,
        binary_scale: i16,
        decimal_scale: i16,
        bits_per_value: u8,
        payload: Vec<u8>,
    }

    fn build_test_grib2_message(spec: SimplePackSpec) -> Vec<u8> {
        let section1 = build_section1();
        let section3 = build_section3();
        let section4 = build_section4(
            spec.category,
            spec.number,
            spec.level_type,
            spec.level_value,
            spec.forecast_time,
        );
        let template = if spec.bits_per_value == 32 || spec.bits_per_value == 64 {
            4
        } else {
            0
        };
        let section5 = build_section5(
            template,
            4,
            spec.reference_value,
            spec.binary_scale,
            spec.decimal_scale,
            spec.bits_per_value,
        );
        let section6 = vec![0, 0, 0, 6, 6, 255];
        let section7 = build_section7(&spec.payload);

        let total_length = 16
            + section1.len()
            + section3.len()
            + section4.len()
            + section5.len()
            + section6.len()
            + section7.len()
            + 4;

        let mut out = Vec::with_capacity(total_length);
        out.extend_from_slice(b"GRIB");
        out.extend_from_slice(&[0, 0]);
        out.push(spec.discipline);
        out.push(2);
        out.extend_from_slice(&(total_length as u64).to_be_bytes());
        out.extend_from_slice(&section1);
        out.extend_from_slice(&section3);
        out.extend_from_slice(&section4);
        out.extend_from_slice(&section5);
        out.extend_from_slice(&section6);
        out.extend_from_slice(&section7);
        out.extend_from_slice(b"7777");
        out
    }

    fn build_section1() -> Vec<u8> {
        let mut sec = vec![0u8; 21];
        sec[0..4].copy_from_slice(&(21u32).to_be_bytes());
        sec[4] = 1;
        sec[5..7].copy_from_slice(&(7u16).to_be_bytes());
        sec[7..9].copy_from_slice(&(0u16).to_be_bytes());
        sec[9] = 28;
        sec[10] = 0;
        sec[11] = 1;
        sec[12..14].copy_from_slice(&(2026u16).to_be_bytes());
        sec[14] = 3;
        sec[15] = 16;
        sec[16] = 18;
        sec[17] = 0;
        sec[18] = 0;
        sec[19] = 0;
        sec[20] = 1;
        sec
    }

    fn build_section3() -> Vec<u8> {
        let mut sec = vec![0u8; 72];
        sec[0..4].copy_from_slice(&(72u32).to_be_bytes());
        sec[4] = 3;
        sec[5] = 0;
        sec[6..10].copy_from_slice(&(4u32).to_be_bytes());
        sec[10] = 0;
        sec[11] = 0;
        sec[12..14].copy_from_slice(&(0u16).to_be_bytes());
        sec[14] = 6;
        sec[30..34].copy_from_slice(&(2u32).to_be_bytes());
        sec[34..38].copy_from_slice(&(2u32).to_be_bytes());
        sec[46..50].copy_from_slice(&(41_000_000u32).to_be_bytes());
        sec[50..54].copy_from_slice(&(100_000_000u32).to_be_bytes());
        sec[55..59].copy_from_slice(&(40_000_000u32).to_be_bytes());
        sec[59..63].copy_from_slice(&(101_000_000u32).to_be_bytes());
        sec[63..67].copy_from_slice(&(1_000_000u32).to_be_bytes());
        sec[67..71].copy_from_slice(&(1_000_000u32).to_be_bytes());
        sec[71] = 0;
        sec
    }

    fn build_section4(
        category: u8,
        number: u8,
        level_type: u8,
        level_value: u32,
        forecast_time: u32,
    ) -> Vec<u8> {
        let mut sec = vec![0u8; 34];
        sec[0..4].copy_from_slice(&(34u32).to_be_bytes());
        sec[4] = 4;
        sec[5..7].copy_from_slice(&(0u16).to_be_bytes());
        sec[7..9].copy_from_slice(&(0u16).to_be_bytes());
        sec[9] = category;
        sec[10] = number;
        sec[11] = 2;
        sec[17] = 1;
        sec[18..22].copy_from_slice(&forecast_time.to_be_bytes());
        sec[22] = level_type;
        sec[23] = 0;
        sec[24..28].copy_from_slice(&level_value.to_be_bytes());
        sec
    }

    fn build_section5(
        template: u16,
        num_points: u32,
        reference_value: f32,
        binary_scale: i16,
        decimal_scale: i16,
        bits_per_value: u8,
    ) -> Vec<u8> {
        let mut sec = vec![0u8; 21];
        sec[0..4].copy_from_slice(&(21u32).to_be_bytes());
        sec[4] = 5;
        sec[5..9].copy_from_slice(&num_points.to_be_bytes());
        sec[9..11].copy_from_slice(&template.to_be_bytes());
        sec[11..15].copy_from_slice(&reference_value.to_be_bytes());
        sec[15..17].copy_from_slice(&binary_scale.to_be_bytes());
        sec[17..19].copy_from_slice(&decimal_scale.to_be_bytes());
        sec[19] = bits_per_value;
        sec[20] = 0;
        sec
    }

    fn build_section7(payload: &[u8]) -> Vec<u8> {
        let mut sec = vec![0u8; 5 + payload.len()];
        sec[0..4].copy_from_slice(&((5 + payload.len()) as u32).to_be_bytes());
        sec[4] = 7;
        sec[5..].copy_from_slice(payload);
        sec
    }
}
