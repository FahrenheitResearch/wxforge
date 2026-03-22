use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};

use clap::Parser;
use serde_json::json;
use wx_calc::{
    dewpoint_from_relative_humidity, lcl, mixed_layer_cape_cin, mixed_parcel,
    most_unstable_cape_cin, most_unstable_parcel, parcel_profile_with_lcl, surface_based_cape_cin,
};
use wx_grib::{DecodedField, GribEngine, MessageDescriptor};

const MIN_PRESSURE_HPA: f64 = 100.0;
const MAX_PRESSURE_HPA: f64 = 1000.0;

#[derive(Debug, Parser)]
#[command(name = "debug-severe-point")]
#[command(about = "Dump wxforge severe-profile internals for a single cropped grid point")]
struct Cli {
    #[arg(
        long,
        default_value = "examples/severe_gfs_verify_gulf/gfs_tmp_pressure.grib2"
    )]
    tmp: PathBuf,
    #[arg(
        long,
        default_value = "examples/severe_gfs_verify_gulf/gfs_rh_pressure.grib2"
    )]
    rh: PathBuf,
    #[arg(
        long,
        default_value = "examples/severe_gfs_verify_gulf/gfs_ugrd_pressure.grib2"
    )]
    ugrd: PathBuf,
    #[arg(
        long,
        default_value = "examples/severe_gfs_verify_gulf/gfs_vgrd_pressure.grib2"
    )]
    vgrd: PathBuf,
    #[arg(
        long,
        default_value = "examples/severe_gfs_verify_gulf/gfs_hgt_pressure.grib2"
    )]
    hgt: PathBuf,
    #[arg(
        long,
        default_value = "examples/severe_gfs_verify_gulf/gfs_pres_pressure.grib2"
    )]
    pres: PathBuf,
    #[arg(long, default_value_t = 22.0)]
    lat_min: f64,
    #[arg(long, default_value_t = 35.0)]
    lat_max: f64,
    #[arg(long, default_value_t = 260.0)]
    lon_min_360: f64,
    #[arg(long, default_value_t = 284.0)]
    lon_max_360: f64,
    #[arg(long, default_value_t = 5)]
    sample_stride: usize,
    #[arg(long)]
    x: usize,
    #[arg(long)]
    y: usize,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct CropSpec {
    x_indices: Vec<usize>,
    y_indices: Vec<usize>,
    nx: usize,
    ny: usize,
    lon_values_deg: Vec<f64>,
    lat_values_deg: Vec<f64>,
}

#[derive(Debug, Clone)]
struct LevelSlice {
    pressure_hpa: f64,
    values: Vec<f64>,
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    let grib = GribEngine::new();

    let base_field = decode_level_message(&grib, &cli.tmp, |message| {
        matches_pressure_level(message, "t", Some(MAX_PRESSURE_HPA))
    })?;
    let crop = build_crop_spec(
        &base_field,
        cli.lat_min,
        cli.lat_max,
        cli.lon_min_360,
        cli.lon_max_360,
        cli.sample_stride,
    )?;

    if cli.x >= crop.nx || cli.y >= crop.ny {
        return Err(format!(
            "point ({}, {}) outside cropped grid {}x{}",
            cli.x, cli.y, crop.nx, crop.ny
        ));
    }

    let temperature_levels = load_pressure_stack(&grib, &cli.tmp, "t", &crop)?;
    let humidity_levels = load_pressure_stack(&grib, &cli.rh, "r", &crop)?;
    let u_levels = load_pressure_stack(&grib, &cli.ugrd, "u", &crop)?;
    let v_levels = load_pressure_stack(&grib, &cli.vgrd, "v", &crop)?;
    let height_levels = load_pressure_stack(&grib, &cli.hgt, "gh", &crop)?;

    let temperature_2m = decode_level_message(&grib, &cli.tmp, |message| {
        matches_surface_level(message, "2t", 2.0)
    })?;
    let relative_humidity_2m = decode_level_message(&grib, &cli.rh, |message| {
        matches_surface_level(message, "r", 2.0)
    })?;
    let u_10m = decode_level_message(&grib, &cli.ugrd, |message| {
        matches_surface_level(message, "10u", 10.0)
    })?;
    let v_10m = decode_level_message(&grib, &cli.vgrd, |message| {
        matches_surface_level(message, "10v", 10.0)
    })?;
    let height_surface = decode_level_message(&grib, &cli.hgt, |message| {
        message.variable.eq_ignore_ascii_case("gh") && message.level == "surface"
    })?;
    let pressure_surface = decode_level_message(&grib, &cli.pres, |message| {
        message.variable.eq_ignore_ascii_case("sp") && message.level == "surface"
    })?;

    let ox = crop.x_indices[cli.x];
    let oy = crop.y_indices[cli.y];
    let psfc_hpa = pressure_surface.grid.get(ox, oy) / 100.0;
    let t2m_c = temperature_2m.grid.get(ox, oy) - 273.15;
    let rh2m_pct = relative_humidity_2m.grid.get(ox, oy);
    let td2m_c = dewpoint_from_relative_humidity(t2m_c, rh2m_pct);
    let u10_ms = u_10m.grid.get(ox, oy);
    let v10_ms = v_10m.grid.get(ox, oy);
    let surface_height_m = height_surface.grid.get(ox, oy);

    let mut pressure_full = vec![psfc_hpa];
    let mut temperature_full_c = vec![t2m_c];
    let mut dewpoint_full_c = vec![td2m_c];
    let mut u_full_ms = vec![u10_ms];
    let mut v_full_ms = vec![v10_ms];
    let mut height_full_m = vec![0.0];

    for idx in 0..temperature_levels.len() {
        let pressure_hpa = temperature_levels[idx].pressure_hpa;
        if pressure_hpa >= psfc_hpa - 0.1 {
            continue;
        }

        let temperature_k = temperature_levels[idx].values[cli.y * crop.nx + cli.x];
        let rh_pct = humidity_levels[idx].values[cli.y * crop.nx + cli.x];
        let u_ms = u_levels[idx].values[cli.y * crop.nx + cli.x];
        let v_ms = v_levels[idx].values[cli.y * crop.nx + cli.x];
        let height_m = height_levels[idx].values[cli.y * crop.nx + cli.x];
        if !temperature_k.is_finite()
            || !rh_pct.is_finite()
            || !u_ms.is_finite()
            || !v_ms.is_finite()
            || !height_m.is_finite()
        {
            continue;
        }

        let height_agl_m = height_m - surface_height_m;
        if height_agl_m < 0.0 {
            continue;
        }

        let temperature_c = temperature_k - 273.15;
        let dewpoint_c = dewpoint_from_relative_humidity(temperature_c, rh_pct);

        pressure_full.push(pressure_hpa);
        temperature_full_c.push(temperature_c);
        dewpoint_full_c.push(dewpoint_c);
        u_full_ms.push(u_ms);
        v_full_ms.push(v_ms);
        height_full_m.push(height_agl_m);
    }

    let sb = build_parcel_dump(
        "sb",
        pressure_full.clone(),
        temperature_full_c.clone(),
        dewpoint_full_c.clone(),
    );

    let (p_ml, t_ml, td_ml) =
        mixed_parcel(&pressure_full, &temperature_full_c, &dewpoint_full_c, 100.0);
    let mut ml_pressure = vec![p_ml];
    let mut ml_temperature = vec![t_ml];
    let mut ml_dewpoint = vec![td_ml];
    for i in 1..pressure_full.len() {
        if pressure_full[i] < p_ml {
            ml_pressure.push(pressure_full[i]);
            ml_temperature.push(temperature_full_c[i]);
            ml_dewpoint.push(dewpoint_full_c[i]);
        }
    }
    let ml = build_parcel_dump("ml", ml_pressure, ml_temperature, ml_dewpoint);

    let (p_mu, _t_mu, _td_mu) =
        most_unstable_parcel(&pressure_full, &temperature_full_c, &dewpoint_full_c, 300.0);
    let start_idx = pressure_full
        .iter()
        .position(|&pressure| (pressure - p_mu).abs() < 1.0e-6)
        .unwrap_or(0);
    let mu = build_parcel_dump(
        "mu",
        pressure_full[start_idx..].to_vec(),
        temperature_full_c[start_idx..].to_vec(),
        dewpoint_full_c[start_idx..].to_vec(),
    );

    let output = json!({
        "crop": {
            "nx": crop.nx,
            "ny": crop.ny,
            "x": cli.x,
            "y": cli.y,
            "source_x": ox,
            "source_y": oy,
            "lat_deg": crop.lat_values_deg[cli.y],
            "lon_deg": crop.lon_values_deg[cli.x],
        },
        "surface": {
            "pressure_hpa": psfc_hpa,
            "temperature_c": t2m_c,
            "dewpoint_c": td2m_c,
            "u10_ms": u10_ms,
            "v10_ms": v10_ms,
            "surface_height_m": surface_height_m,
        },
        "environment": {
            "pressure_hpa": pressure_full,
            "temperature_c": temperature_full_c,
            "dewpoint_c": dewpoint_full_c,
            "u_ms": u_full_ms,
            "v_ms": v_full_ms,
            "height_agl_m": height_full_m,
        },
        "surface_based": sb,
        "mixed_layer": ml,
        "most_unstable": mu,
    });

    let text =
        serde_json::to_string_pretty(&output).map_err(|err| format!("serialize failed: {err}"))?;
    if let Some(path) = cli.output {
        fs::write(&path, text)
            .map_err(|err| format!("failed to write '{}': {err}", path.display()))?;
    } else {
        println!("{text}");
    }

    Ok(())
}

fn build_parcel_dump(
    label: &str,
    pressure_hpa: Vec<f64>,
    temperature_c: Vec<f64>,
    dewpoint_c: Vec<f64>,
) -> serde_json::Value {
    let (start_p, start_t, start_td) = (pressure_hpa[0], temperature_c[0], dewpoint_c[0]);
    let (p_lcl, t_lcl) = lcl(start_p, start_t, start_td);
    let (pressure_aug, parcel_profile_c) =
        parcel_profile_with_lcl(&pressure_hpa, start_t, start_td);
    let (cape, cin) = match label {
        "sb" => surface_based_cape_cin(&pressure_hpa, &temperature_c, &dewpoint_c),
        "ml" => mixed_layer_cape_cin(&pressure_hpa, &temperature_c, &dewpoint_c, 100.0),
        "mu" => most_unstable_cape_cin(&pressure_hpa, &temperature_c, &dewpoint_c),
        _ => (f64::NAN, f64::NAN),
    };

    json!({
        "start_pressure_hpa": start_p,
        "start_temperature_c": start_t,
        "start_dewpoint_c": start_td,
        "lcl_pressure_hpa": p_lcl,
        "lcl_temperature_c": t_lcl,
        "pressure_hpa": pressure_hpa,
        "temperature_c": temperature_c,
        "dewpoint_c": dewpoint_c,
        "parcel_pressure_hpa": pressure_aug,
        "parcel_temperature_c": parcel_profile_c,
        "cape_jkg": cape,
        "cin_jkg": cin,
    })
}

fn build_crop_spec(
    field: &DecodedField,
    lat_min: f64,
    lat_max: f64,
    lon_min_360: f64,
    lon_max_360: f64,
    stride: usize,
) -> Result<CropSpec, String> {
    let x_axis = field
        .x_axis
        .as_ref()
        .ok_or_else(|| "decoded field missing longitude axis".to_string())?;
    let y_axis = field
        .y_axis
        .as_ref()
        .ok_or_else(|| "decoded field missing latitude axis".to_string())?;

    let x_indices = x_axis
        .values
        .iter()
        .enumerate()
        .filter(|(_, lon)| lon_min_360 <= **lon && **lon <= lon_max_360)
        .step_by(stride)
        .map(|(idx, _)| idx)
        .collect::<Vec<_>>();
    let y_indices = y_axis
        .values
        .iter()
        .enumerate()
        .filter(|(_, lat)| lat_min <= **lat && **lat <= lat_max)
        .step_by(stride)
        .map(|(idx, _)| idx)
        .collect::<Vec<_>>();

    if x_indices.is_empty() || y_indices.is_empty() {
        return Err("crop produced no grid points".to_string());
    }

    Ok(CropSpec {
        nx: x_indices.len(),
        ny: y_indices.len(),
        lon_values_deg: x_indices
            .iter()
            .map(|&idx| normalize_longitude(x_axis.values[idx]))
            .collect(),
        lat_values_deg: y_indices.iter().map(|&idx| y_axis.values[idx]).collect(),
        x_indices,
        y_indices,
    })
}

fn load_pressure_stack(
    grib: &GribEngine,
    path: &Path,
    expected_var: &str,
    crop: &CropSpec,
) -> Result<Vec<LevelSlice>, String> {
    let inventory = grib.scan_file(path)?;
    let mut messages = inventory
        .messages
        .iter()
        .filter(|message| matches_pressure_level(message, expected_var, None))
        .map(|message| {
            (
                normalized_pressure_hpa(message.level_value.unwrap_or_default()),
                message.message_no,
            )
        })
        .collect::<Vec<_>>();
    messages.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

    let mut out = Vec::with_capacity(messages.len());
    for (pressure_hpa, message_no) in messages {
        let field = grib.decode_file_message(path, message_no)?;
        out.push(LevelSlice {
            pressure_hpa,
            values: extract_crop(&field, crop),
        });
    }
    Ok(out)
}

fn decode_level_message(
    grib: &GribEngine,
    path: &Path,
    predicate: impl Fn(&MessageDescriptor) -> bool,
) -> Result<DecodedField, String> {
    let inventory = grib.scan_file(path)?;
    let message = inventory
        .messages
        .iter()
        .find(|message| predicate(message))
        .ok_or_else(|| format!("no matching message found in '{}'", path.display()))?;
    grib.decode_file_message(path, message.message_no)
}

fn matches_pressure_level(
    message: &MessageDescriptor,
    expected_var: &str,
    expected_pressure_hpa: Option<f64>,
) -> bool {
    if !message.variable.eq_ignore_ascii_case(expected_var) || message.level_type != Some(100) {
        return false;
    }
    let pressure_hpa = match message.level_value {
        Some(value) => normalized_pressure_hpa(value),
        None => return false,
    };
    if !(MIN_PRESSURE_HPA..=MAX_PRESSURE_HPA).contains(&pressure_hpa) {
        return false;
    }
    expected_pressure_hpa
        .map(|target| (pressure_hpa - target).abs() < 1.0e-6)
        .unwrap_or(true)
}

fn matches_surface_level(message: &MessageDescriptor, expected_var: &str, level_m: f64) -> bool {
    message.variable.eq_ignore_ascii_case(expected_var)
        && message.level_type == Some(103)
        && message
            .level_value
            .is_some_and(|value| (value - level_m).abs() < 1.0e-6)
}

fn normalized_pressure_hpa(value: f64) -> f64 {
    value / 100.0
}

fn extract_crop(field: &DecodedField, crop: &CropSpec) -> Vec<f64> {
    let mut values = Vec::with_capacity(crop.nx * crop.ny);
    for &y in &crop.y_indices {
        for &x in &crop.x_indices {
            values.push(field.grid.get(x, y));
        }
    }
    values
}

fn normalize_longitude(lon_deg: f64) -> f64 {
    if lon_deg > 180.0 {
        lon_deg - 360.0
    } else {
        lon_deg
    }
}
