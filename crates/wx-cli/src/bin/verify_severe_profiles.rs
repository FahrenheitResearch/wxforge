use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};

use clap::Parser;
use serde_json::json;
use wx_calc::{
    bulk_shear_pressure, bunkers_storm_motion, cape_cin, dewpoint_from_relative_humidity,
    get_most_unstable_parcel, k_index, lifted_index, mixed_layer_cape_cin, most_unstable_cape_cin,
    precipitable_water, showalter_index, significant_tornado_parameter, storm_relative_helicity,
    surface_based_cape_cin, total_totals,
};
use wx_export::ExportEngine;
use wx_grib::{DecodedField, GribEngine, MessageDescriptor};
use wx_types::Grid2D;

const MIN_PRESSURE_HPA: f64 = 100.0;
const MAX_PRESSURE_HPA: f64 = 1000.0;

#[derive(Debug, Parser)]
#[command(name = "verify-severe-profiles")]
#[command(about = "Decode cropped severe-weather verification grids as NPY")]
struct Cli {
    #[arg(
        long,
        default_value = "examples/severe_gfs_verify/gfs_tmp_pressure.grib2"
    )]
    tmp: PathBuf,
    #[arg(
        long,
        default_value = "examples/severe_gfs_verify/gfs_rh_pressure.grib2"
    )]
    rh: PathBuf,
    #[arg(
        long,
        default_value = "examples/severe_gfs_verify/gfs_ugrd_pressure.grib2"
    )]
    ugrd: PathBuf,
    #[arg(
        long,
        default_value = "examples/severe_gfs_verify/gfs_vgrd_pressure.grib2"
    )]
    vgrd: PathBuf,
    #[arg(
        long,
        default_value = "examples/severe_gfs_verify/gfs_hgt_pressure.grib2"
    )]
    hgt: PathBuf,
    #[arg(
        long,
        default_value = "examples/severe_gfs_verify/gfs_pres_pressure.grib2"
    )]
    pres: PathBuf,
    #[arg(long, default_value = "examples/severe_gfs_verify/wxforge_output")]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 30.0)]
    lat_min: f64,
    #[arg(long, default_value_t = 40.0)]
    lat_max: f64,
    #[arg(long, default_value_t = 258.0)]
    lon_min_360: f64,
    #[arg(long, default_value_t = 268.0)]
    lon_max_360: f64,
    #[arg(long, default_value_t = 4)]
    sample_stride: usize,
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
struct LevelGrid {
    pressure_hpa: f64,
    grid: Grid2D,
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    fs::create_dir_all(&cli.output_dir)
        .map_err(|err| format!("failed to create '{}': {err}", cli.output_dir.display()))?;

    let grib = GribEngine::new();
    let export = ExportEngine::new();

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

    let temperature_levels = load_pressure_stack(&grib, &cli.tmp, "t", &crop)?;
    let humidity_levels = load_pressure_stack(&grib, &cli.rh, "r", &crop)?;
    let u_levels = load_pressure_stack(&grib, &cli.ugrd, "u", &crop)?;
    let v_levels = load_pressure_stack(&grib, &cli.vgrd, "v", &crop)?;
    let height_levels = load_pressure_stack(&grib, &cli.hgt, "gh", &crop)?;

    let temperature_2m = decode_level_message(&grib, &cli.tmp, |message| {
        matches_surface_level(message, "2t", 2.0)
    })
    .map(|field| extract_crop(&field, &crop))?;
    let relative_humidity_2m = decode_level_message(&grib, &cli.rh, |message| {
        matches_surface_level(message, "r", 2.0)
    })
    .map(|field| extract_crop(&field, &crop))?;
    let u_10m = decode_level_message(&grib, &cli.ugrd, |message| {
        matches_surface_level(message, "10u", 10.0)
    })
    .map(|field| extract_crop(&field, &crop))?;
    let v_10m = decode_level_message(&grib, &cli.vgrd, |message| {
        matches_surface_level(message, "10v", 10.0)
    })
    .map(|field| extract_crop(&field, &crop))?;
    let height_surface = decode_level_message(&grib, &cli.hgt, |message| {
        message.variable.eq_ignore_ascii_case("gh") && message.level == "surface"
    })
    .map(|field| extract_crop(&field, &crop))?;
    let pressure_surface = decode_level_message(&grib, &cli.pres, |message| {
        message.variable.eq_ignore_ascii_case("sp") && message.level == "surface"
    })
    .map(|field| extract_crop(&field, &crop))?;

    let diagnostics = compute_diagnostics(
        &crop,
        &temperature_levels,
        &humidity_levels,
        &u_levels,
        &v_levels,
        &height_levels,
        &temperature_2m,
        &relative_humidity_2m,
        &u_10m,
        &v_10m,
        &height_surface,
        &pressure_surface,
    );

    let latitude = lat_lon_grid(&crop, true);
    let longitude = lat_lon_grid(&crop, false);
    write_grid(&export, &cli.output_dir, "latitude", &latitude)?;
    write_grid(&export, &cli.output_dir, "longitude", &longitude)?;
    for (name, grid) in &diagnostics {
        write_grid(&export, &cli.output_dir, name, grid)?;
    }

    let manifest = json!({
        "crop": {
            "lat_min": cli.lat_min,
            "lat_max": cli.lat_max,
            "lon_min_360": cli.lon_min_360,
            "lon_max_360": cli.lon_max_360,
            "stride": cli.sample_stride,
            "nx": crop.nx,
            "ny": crop.ny,
        },
        "pressure_levels_hpa": temperature_levels.iter().map(|level| level.pressure_hpa).collect::<Vec<_>>(),
        "fields": diagnostics.iter().map(|(name, grid)| (name.clone(), grid_summary(grid))).collect::<serde_json::Map<_, _>>(),
    });
    fs::write(
        cli.output_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)
            .map_err(|err| format!("failed to serialize manifest: {err}"))?,
    )
    .map_err(|err| format!("failed to write manifest: {err}"))?;

    println!("wrote {}", cli.output_dir.display());
    Ok(())
}

fn compute_diagnostics(
    crop: &CropSpec,
    temperature_levels: &[LevelGrid],
    humidity_levels: &[LevelGrid],
    u_levels: &[LevelGrid],
    v_levels: &[LevelGrid],
    height_levels: &[LevelGrid],
    temperature_2m: &Grid2D,
    relative_humidity_2m: &Grid2D,
    u_10m: &Grid2D,
    v_10m: &Grid2D,
    height_surface: &Grid2D,
    pressure_surface: &Grid2D,
) -> Vec<(String, Grid2D)> {
    let mut sbcape = filled_grid(crop.nx, crop.ny, f64::NAN);
    let mut sbcin = filled_grid(crop.nx, crop.ny, f64::NAN);
    let mut h_lcl = filled_grid(crop.nx, crop.ny, f64::NAN);
    let mut mlcape = filled_grid(crop.nx, crop.ny, f64::NAN);
    let mut mlcin = filled_grid(crop.nx, crop.ny, f64::NAN);
    let mut mucape = filled_grid(crop.nx, crop.ny, f64::NAN);
    let mut mucin = filled_grid(crop.nx, crop.ny, f64::NAN);
    let mut mu_start_p = filled_grid(crop.nx, crop.ny, f64::NAN);
    let mut shear06 = filled_grid(crop.nx, crop.ny, f64::NAN);
    let mut rm_u = filled_grid(crop.nx, crop.ny, f64::NAN);
    let mut rm_v = filled_grid(crop.nx, crop.ny, f64::NAN);
    let mut srh01 = filled_grid(crop.nx, crop.ny, f64::NAN);
    let mut stp = filled_grid(crop.nx, crop.ny, f64::NAN);
    let mut li = filled_grid(crop.nx, crop.ny, f64::NAN);
    let mut showalter = filled_grid(crop.nx, crop.ny, f64::NAN);
    let mut kindex = filled_grid(crop.nx, crop.ny, f64::NAN);
    let mut total_totals_grid = filled_grid(crop.nx, crop.ny, f64::NAN);
    let mut pwat = filled_grid(crop.nx, crop.ny, f64::NAN);

    for y in 0..crop.ny {
        for x in 0..crop.nx {
            let psfc_pa = pressure_surface.get(x, y);
            let t2m_k = temperature_2m.get(x, y);
            let rh2m_pct = relative_humidity_2m.get(x, y);
            let u10 = u_10m.get(x, y);
            let v10 = v_10m.get(x, y);
            let surface_height_m = height_surface.get(x, y);

            if !psfc_pa.is_finite()
                || !t2m_k.is_finite()
                || !rh2m_pct.is_finite()
                || !u10.is_finite()
                || !v10.is_finite()
                || !surface_height_m.is_finite()
            {
                continue;
            }

            let psfc_hpa = psfc_pa / 100.0;
            if !(800.0..=1100.0).contains(&psfc_hpa) {
                continue;
            }

            let t2m_c = t2m_k - 273.15;
            let td2m_c = dewpoint_from_relative_humidity(t2m_c, rh2m_pct);

            let mut pressure_profile = Vec::with_capacity(temperature_levels.len());
            let mut temperature_profile_c = Vec::with_capacity(temperature_levels.len());
            let mut dewpoint_profile_c = Vec::with_capacity(temperature_levels.len());
            let mut height_agl = Vec::with_capacity(temperature_levels.len());
            let mut pressure_full = vec![psfc_hpa];
            let mut temperature_full_c = vec![t2m_c];
            let mut dewpoint_full_c = vec![td2m_c];
            let mut u_full = vec![u10];
            let mut v_full = vec![v10];
            let mut height_full = vec![0.0];

            for idx in 0..temperature_levels.len() {
                let pressure_hpa = temperature_levels[idx].pressure_hpa;
                if pressure_hpa >= psfc_hpa - 0.1 {
                    continue;
                }

                let temperature_k = temperature_levels[idx].grid.get(x, y);
                let rh_pct = humidity_levels[idx].grid.get(x, y);
                let u = u_levels[idx].grid.get(x, y);
                let v = v_levels[idx].grid.get(x, y);
                let height_m = height_levels[idx].grid.get(x, y);
                if !temperature_k.is_finite()
                    || !rh_pct.is_finite()
                    || !u.is_finite()
                    || !v.is_finite()
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

                pressure_profile.push(pressure_hpa);
                temperature_profile_c.push(temperature_c);
                dewpoint_profile_c.push(dewpoint_c);
                height_agl.push(height_agl_m);

                pressure_full.push(pressure_hpa);
                temperature_full_c.push(temperature_c);
                dewpoint_full_c.push(dewpoint_c);
                u_full.push(u);
                v_full.push(v);
                height_full.push(height_agl_m);
            }

            if pressure_profile.len() < 6 || pressure_full.len() < 6 {
                continue;
            }

            let (sb_cape, sb_cin) =
                surface_based_cape_cin(&pressure_full, &temperature_full_c, &dewpoint_full_c);
            let (ml_cape, ml_cin) =
                mixed_layer_cape_cin(&pressure_full, &temperature_full_c, &dewpoint_full_c, 100.0);
            let (mu_cape, mu_cin) =
                most_unstable_cape_cin(&pressure_full, &temperature_full_c, &dewpoint_full_c);
            let (mu_pressure_hpa, _, _) = get_most_unstable_parcel(
                &pressure_full,
                &temperature_full_c,
                &dewpoint_full_c,
                300.0,
            );
            let (_, _, lcl_height_m, _) = cape_cin(
                &pressure_profile,
                &temperature_profile_c,
                &dewpoint_profile_c,
                &height_agl,
                psfc_hpa,
                t2m_c,
                td2m_c,
                "sb",
                100.0,
                300.0,
                None,
            );

            let (shear_u_ms, shear_v_ms) =
                bulk_shear_pressure(&pressure_full, &u_full, &v_full, &height_full, 0.0, 6000.0);
            let shear_mag_ms = (shear_u_ms * shear_u_ms + shear_v_ms * shear_v_ms).sqrt();

            let (right_mover, _, _) =
                bunkers_storm_motion(&pressure_full, &u_full, &v_full, &height_full);
            let (_, _, srh_total) = storm_relative_helicity(
                &u_full,
                &v_full,
                &height_full,
                1000.0,
                right_mover.0,
                right_mover.1,
            );
            let stp_value =
                significant_tornado_parameter(sb_cape, lcl_height_m, srh_total, shear_mag_ms);
            let li_value = lifted_index(&pressure_full, &temperature_full_c, &dewpoint_full_c);
            let showalter_value =
                showalter_index(&pressure_full, &temperature_full_c, &dewpoint_full_c);
            let kindex_value = level_triplet_indices(
                &pressure_full,
                &temperature_full_c,
                &dewpoint_full_c,
                |t850, td850, t700, td700, t500| k_index(t850, td850, t700, td700, t500),
            );
            let total_totals_value = level_triplet_indices(
                &pressure_full,
                &temperature_full_c,
                &dewpoint_full_c,
                |t850, td850, _t700, _td700, t500| total_totals(t850, td850, t500),
            );
            let pwat_value = precipitable_water(&pressure_full, &dewpoint_full_c);

            sbcape.set(x, y, sb_cape);
            sbcin.set(x, y, sb_cin);
            h_lcl.set(x, y, lcl_height_m);
            mlcape.set(x, y, ml_cape);
            mlcin.set(x, y, ml_cin);
            mucape.set(x, y, mu_cape);
            mucin.set(x, y, mu_cin);
            mu_start_p.set(x, y, mu_pressure_hpa);
            shear06.set(x, y, shear_mag_ms);
            rm_u.set(x, y, right_mover.0);
            rm_v.set(x, y, right_mover.1);
            srh01.set(x, y, srh_total);
            stp.set(x, y, stp_value);
            li.set(x, y, li_value);
            showalter.set(x, y, showalter_value);
            kindex.set(x, y, kindex_value);
            total_totals_grid.set(x, y, total_totals_value);
            pwat.set(x, y, pwat_value);
        }
    }

    vec![
        ("sbcape".to_string(), sbcape),
        ("sbcin".to_string(), sbcin),
        ("h_lcl".to_string(), h_lcl),
        ("mlcape".to_string(), mlcape),
        ("mlcin".to_string(), mlcin),
        ("mucape".to_string(), mucape),
        ("mucin".to_string(), mucin),
        ("mu_start_p".to_string(), mu_start_p),
        ("shear06".to_string(), shear06),
        ("rm_u".to_string(), rm_u),
        ("rm_v".to_string(), rm_v),
        ("srh01".to_string(), srh01),
        ("stp".to_string(), stp),
        ("li".to_string(), li),
        ("showalter".to_string(), showalter),
        ("k_index".to_string(), kindex),
        ("total_totals".to_string(), total_totals_grid),
        ("pwat".to_string(), pwat),
    ]
}

fn level_triplet_indices(
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    dewpoint_c: &[f64],
    func: impl Fn(f64, f64, f64, f64, f64) -> f64,
) -> f64 {
    let t850 = interp_profile_value(850.0, pressure_hpa, temperature_c);
    let td850 = interp_profile_value(850.0, pressure_hpa, dewpoint_c);
    let t700 = interp_profile_value(700.0, pressure_hpa, temperature_c);
    let td700 = interp_profile_value(700.0, pressure_hpa, dewpoint_c);
    let t500 = interp_profile_value(500.0, pressure_hpa, temperature_c);
    if [t850, td850, t700, td700, t500]
        .iter()
        .all(|value| value.is_finite())
    {
        func(t850, td850, t700, td700, t500)
    } else {
        f64::NAN
    }
}

fn interp_profile_value(target_hpa: f64, pressure_hpa: &[f64], values: &[f64]) -> f64 {
    let n = pressure_hpa.len().min(values.len());
    if n == 0 {
        return f64::NAN;
    }
    if (pressure_hpa[0] - target_hpa).abs() < 1.0e-9 {
        return values[0];
    }
    for i in 1..n {
        let p0 = pressure_hpa[i - 1];
        let p1 = pressure_hpa[i];
        let v0 = values[i - 1];
        let v1 = values[i];
        if !p0.is_finite() || !p1.is_finite() || !v0.is_finite() || !v1.is_finite() {
            continue;
        }
        let crosses =
            (p0 >= target_hpa && p1 <= target_hpa) || (p0 <= target_hpa && p1 >= target_hpa);
        if !crosses {
            continue;
        }
        let log_p0 = p0.ln();
        let log_p1 = p1.ln();
        if (log_p1 - log_p0).abs() < 1.0e-12 {
            return v0;
        }
        let frac = (target_hpa.ln() - log_p0) / (log_p1 - log_p0);
        return v0 + frac * (v1 - v0);
    }
    f64::NAN
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

    let lon_values_deg = x_indices
        .iter()
        .map(|&idx| normalize_longitude(x_axis.values[idx]))
        .collect::<Vec<_>>();
    let lat_values_deg = y_indices
        .iter()
        .map(|&idx| y_axis.values[idx])
        .collect::<Vec<_>>();
    Ok(CropSpec {
        nx: x_indices.len(),
        ny: y_indices.len(),
        x_indices,
        y_indices,
        lon_values_deg,
        lat_values_deg,
    })
}

fn load_pressure_stack(
    grib: &GribEngine,
    path: &Path,
    expected_var: &str,
    crop: &CropSpec,
) -> Result<Vec<LevelGrid>, String> {
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
        out.push(LevelGrid {
            pressure_hpa,
            grid: extract_crop(&field, crop),
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

fn extract_crop(field: &DecodedField, crop: &CropSpec) -> Grid2D {
    let mut values = Vec::with_capacity(crop.nx * crop.ny);
    for &y in &crop.y_indices {
        for &x in &crop.x_indices {
            values.push(field.grid.get(x, y));
        }
    }
    Grid2D::new(crop.nx, crop.ny, values)
}

fn lat_lon_grid(crop: &CropSpec, latitude: bool) -> Grid2D {
    let mut values = Vec::with_capacity(crop.nx * crop.ny);
    for &lat in &crop.lat_values_deg {
        for &lon in &crop.lon_values_deg {
            values.push(if latitude { lat } else { lon });
        }
    }
    Grid2D::new(crop.nx, crop.ny, values)
}

fn normalize_longitude(lon_deg: f64) -> f64 {
    if lon_deg > 180.0 {
        lon_deg - 360.0
    } else {
        lon_deg
    }
}

fn write_grid(
    export: &ExportEngine,
    out_dir: &Path,
    name: &str,
    grid: &Grid2D,
) -> Result<(), String> {
    export.write_npy_f32_grid(out_dir.join(format!("{name}.npy")), grid)
}

fn grid_summary(grid: &Grid2D) -> serde_json::Value {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0;
    let mut count = 0usize;
    for &value in &grid.values {
        if value.is_finite() {
            min = min.min(value);
            max = max.max(value);
            sum += value;
            count += 1;
        }
    }
    json!({
        "count": count,
        "min": if count > 0 { min } else { f64::NAN },
        "mean": if count > 0 { sum / count as f64 } else { f64::NAN },
        "max": if count > 0 { max } else { f64::NAN },
    })
}

fn filled_grid(nx: usize, ny: usize, value: f64) -> Grid2D {
    Grid2D::new(nx, ny, vec![value; nx * ny])
}
