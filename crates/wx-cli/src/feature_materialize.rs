use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

use chrono::{DateTime, Duration, Timelike, Utc};
use wx_calc::{
    bulk_shear_pressure, bunkers_storm_motion, cape_cin, dewpoint_from_relative_humidity,
    downdraft_cape, equivalent_potential_temperature, geospatial_gradient, k_index, lifted_index,
    mixed_layer_cape_cin, most_unstable_cape_cin, potential_temperature, precipitable_water,
    relative_humidity_from_dewpoint, showalter_index, significant_tornado_parameter,
    storm_relative_helicity, supercell_composite_parameter, surface_based_cape_cin, total_totals,
    wet_bulb_potential_temperature, wet_bulb_temperature, wind_direction, wind_speed,
};
use wx_export::{ExportEngine, SampleBundleManifest, SampleChannelArtifact};
use wx_grib::{DecodedField, GribEngine, MessageDescriptor};
use wx_render::{render_scalar_grid, write_png_rgba, ColorMap};
use wx_train::GribSampleInput;
use wx_types::{Grid2D, TrainingChannel};

const MIN_PRESSURE_HPA: f64 = 100.0;
const MAX_PRESSURE_HPA: f64 = 1000.0;

struct LevelField<'a> {
    pressure_hpa: f64,
    field: &'a DecodedField,
}

pub fn write_planned_training_sample(
    sample: &GribSampleInput,
    output_dir: &PathBuf,
    dataset_name: String,
    sample_id: String,
    requested_channels: &[TrainingChannel],
    colormap: ColorMap,
    min: Option<f64>,
    max: Option<f64>,
) -> Result<SampleBundleManifest, String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create output directory '{}': {err}",
            output_dir.display()
        )
    })?;

    let fields = load_sample_fields(sample)?;
    if fields.is_empty() {
        return Err(format!(
            "sample '{}' resolved to zero decoded fields",
            sample.file
        ));
    }
    ensure_matching_grids(&fields)?;

    let export = ExportEngine::new();
    let requested_names = requested_channels
        .iter()
        .map(|channel| channel.name.clone())
        .collect::<Vec<_>>();
    let materialized = materialize_channel_grids(&fields, &requested_names)?;
    let mut artifacts = Vec::with_capacity(requested_channels.len());

    for channel in requested_channels {
        let grid = materialized.get(&channel.name).ok_or_else(|| {
            format!(
                "sample '{}' could not materialize requested channel '{}'",
                sample.file, channel.name
            )
        })?;
        let stem = sanitize_channel_stem(&channel.name);
        let data_name = format!("{stem}.npy");
        let preview_name = format!("{stem}.png");
        let data_path = output_dir.join(&data_name);
        let preview_path = output_dir.join(&preview_name);

        export.write_npy_f32_grid(&data_path, grid)?;
        let stats = crate::compute_channel_stats(grid).map(|full_stats| {
            let rgba = render_scalar_grid(
                grid,
                min.unwrap_or(full_stats.min),
                max.unwrap_or(full_stats.max),
                colormap,
            );
            let _ = write_png_rgba(&preview_path, grid.nx as u32, grid.ny as u32, &rgba);
            full_stats
        });
        let preview_file = if stats.is_some() {
            Some(preview_name)
        } else {
            None
        };

        artifacts.push(SampleChannelArtifact {
            message_no: 0,
            name: channel.name.clone(),
            level: inferred_channel_level(&channel.name),
            units: channel.units.clone(),
            width: grid.nx,
            height: grid.ny,
            missing_count: grid.values.iter().filter(|value| value.is_nan()).count(),
            data_file: data_name,
            preview_file,
            stats,
        });
    }

    let plan = wx_train::TrainingPlan::from_channels(dataset_name, requested_channels.to_vec());
    let dataset_manifest = export.manifest_from_plan(&plan.export);
    export.write_manifest(output_dir.join("dataset_manifest.json"), &dataset_manifest)?;
    let sample_manifest =
        export.sample_bundle_manifest(&plan.export, sample_id, sample.file.clone(), artifacts);
    export
        .write_sample_bundle_manifest(output_dir.join("sample_manifest.json"), &sample_manifest)?;
    crate::write_channel_stats_json(output_dir, &sample_manifest.channels)?;
    Ok(sample_manifest)
}

fn load_sample_fields(sample: &GribSampleInput) -> Result<Vec<DecodedField>, String> {
    let grib = GribEngine::new();
    let mut paths = vec![PathBuf::from(&sample.file)];
    for companion in &sample.companion_files {
        paths.push(PathBuf::from(companion));
    }

    let mut fields = Vec::new();
    for (path_index, path) in paths.iter().enumerate() {
        let inventory = grib.scan_file(path)?;
        let message_ids = if path_index == 0 && !sample.messages.is_empty() {
            sample.messages.clone()
        } else {
            inventory
                .messages
                .iter()
                .map(|message| message.message_no)
                .collect::<Vec<_>>()
        };
        for message_no in message_ids {
            fields.push(grib.decode_file_message(path, message_no)?);
        }
    }
    Ok(fields)
}

fn ensure_matching_grids(fields: &[DecodedField]) -> Result<(), String> {
    let base = &fields[0];
    for field in &fields[1..] {
        if base.grid.nx != field.grid.nx || base.grid.ny != field.grid.ny {
            return Err(format!(
                "decoded field '{}' grid {}x{} does not match base grid {}x{}",
                field.descriptor.variable, field.grid.nx, field.grid.ny, base.grid.nx, base.grid.ny
            ));
        }
    }
    Ok(())
}

fn materialize_channel_grids(
    fields: &[DecodedField],
    requested_names: &[String],
) -> Result<HashMap<String, Grid2D>, String> {
    let requested = requested_names
        .iter()
        .map(|name| name.to_ascii_lowercase())
        .collect::<HashSet<_>>();
    let mut out = HashMap::new();

    for name in &requested {
        if let Some(grid) = materialize_raw_channel(fields, name)? {
            out.insert(name.clone(), grid);
        }
    }

    // Computed surface-level channels: wind_speed, wind_direction, relative_humidity.
    // These derive from pairs of raw fields that are typically available in any GRIB file.
    let needs_surface_wind = requested.iter().any(|name| {
        matches!(name.as_str(), "wind_speed" | "wind_speed_10m" | "wind_direction" | "wind_direction_10m")
    });
    if needs_surface_wind {
        let u_field = find_surface_field(fields, &["10u", "u10"], 10.0)
            .or_else(|| lowest_pressure_field(fields, &["u", "ugrd"]));
        let v_field = find_surface_field(fields, &["10v", "v10"], 10.0)
            .or_else(|| lowest_pressure_field(fields, &["v", "vgrd"]));
        if let (Some(u_field), Some(v_field)) = (u_field, v_field) {
            if requested.contains("wind_speed") || requested.contains("wind_speed_10m") {
                let grid = apply_binary_grid(&u_field.grid, &v_field.grid, |u, v| {
                    wind_speed(u, v)
                })?;
                if requested.contains("wind_speed") {
                    out.insert("wind_speed".to_string(), grid.clone());
                }
                if requested.contains("wind_speed_10m") {
                    out.insert("wind_speed_10m".to_string(), grid);
                }
            }
            if requested.contains("wind_direction") || requested.contains("wind_direction_10m") {
                let grid = apply_binary_grid(&u_field.grid, &v_field.grid, |u, v| {
                    wind_direction(u, v)
                })?;
                if requested.contains("wind_direction") {
                    out.insert("wind_direction".to_string(), grid.clone());
                }
                if requested.contains("wind_direction_10m") {
                    out.insert("wind_direction_10m".to_string(), grid);
                }
            }
        }
    }

    let needs_rh = requested.contains("relative_humidity") || requested.contains("rh2m");
    if needs_rh {
        // Try direct RH field first.
        let rh_direct = find_surface_field(fields, &["r"], 2.0)
            .or_else(|| find_any_field(fields, &["r", "rh"]));
        if let Some(rh_field) = rh_direct {
            if requested.contains("relative_humidity") && !out.contains_key("relative_humidity") {
                out.insert("relative_humidity".to_string(), rh_field.grid.clone());
            }
            if requested.contains("rh2m") && !out.contains_key("rh2m") {
                out.insert("rh2m".to_string(), rh_field.grid.clone());
            }
        } else {
            // Compute from TMP and DPT (both in K in GRIB).
            let t_field = find_surface_field(fields, &["2t", "t2m"], 2.0)
                .or_else(|| lowest_pressure_field(fields, &["t", "tmp"]));
            let d_field = find_surface_field(fields, &["2d", "d2m"], 2.0)
                .or_else(|| lowest_pressure_field(fields, &["d", "dpt"]));
            if let (Some(t_field), Some(d_field)) = (t_field, d_field) {
                let grid = apply_binary_grid(&t_field.grid, &d_field.grid, |t_k, td_k| {
                    relative_humidity_from_dewpoint(t_k - 273.15, td_k - 273.15)
                })?;
                if requested.contains("relative_humidity") {
                    out.insert("relative_humidity".to_string(), grid.clone());
                }
                if requested.contains("rh2m") {
                    out.insert("rh2m".to_string(), grid);
                }
            }
        }
    }

    if requested
        .iter()
        .any(|name| matches!(name.as_str(), "theta850" | "vort500" | "div500" | "tadv850"))
    {
        for (name, grid) in compute_pressure_map_diagnostics(fields)? {
            if requested.contains(&name) {
                out.insert(name, grid);
            }
        }
    }

    let needs_profile = requested.iter().any(|name| {
        matches!(
            name.as_str(),
            "sbcape"
                | "sbcin"
                | "mlcape"
                | "mlcin"
                | "mucape"
                | "mucin"
                | "shear06"
                | "srh01"
                | "srh03"
                | "stp"
                | "scp"
                | "pwat"
                | "lifted_index"
                | "li"
                | "showalter_index"
                | "showalter"
                | "k_index"
                | "total_totals"
                | "theta_e"
                | "wet_bulb"
                | "wet_bulb_potential_temperature"
                | "lcl_height"
                | "lfc_height"
                | "dcape"
        ) || parse_custom_srh_depth_m(name).is_some()
    });
    if needs_profile {
        for (name, grid) in compute_profile_diagnostics(fields, &requested)? {
            out.insert(name, grid);
        }
    }

    if requested.contains("channel_min")
        || requested.contains("channel_mean")
        || requested.contains("channel_max")
    {
        let (stat_min, stat_mean, stat_max) =
            aggregate_channel_stats(&out, fields).ok_or_else(|| {
                "failed to compute aggregate channel statistics for requested tabular stats"
                    .to_string()
            })?;
        let base = &fields[0].grid;
        if requested.contains("channel_min") {
            out.insert(
                "channel_min".to_string(),
                filled_grid(base.nx, base.ny, stat_min),
            );
        }
        if requested.contains("channel_mean") {
            out.insert(
                "channel_mean".to_string(),
                filled_grid(base.nx, base.ny, stat_mean),
            );
        }
        if requested.contains("channel_max") {
            out.insert(
                "channel_max".to_string(),
                filled_grid(base.nx, base.ny, stat_max),
            );
        }
    }

    if requested.contains("valid_hour_sin") || requested.contains("valid_hour_cos") {
        let hour = infer_valid_hour_utc(fields);
        let (sin_value, cos_value) = hour
            .map(|hour| {
                let angle = std::f64::consts::TAU * (hour as f64 / 24.0);
                (angle.sin(), angle.cos())
            })
            .unwrap_or((f64::NAN, f64::NAN));
        let base = &fields[0].grid;
        if requested.contains("valid_hour_sin") {
            out.insert(
                "valid_hour_sin".to_string(),
                filled_grid(base.nx, base.ny, sin_value),
            );
        }
        if requested.contains("valid_hour_cos") {
            out.insert(
                "valid_hour_cos".to_string(),
                filled_grid(base.nx, base.ny, cos_value),
            );
        }
    }

    for name in requested_names {
        if !out.contains_key(&name.to_ascii_lowercase()) {
            return Err(format!(
                "requested channel '{}' is not supported by the current materializer or missing required source fields",
                name
            ));
        }
    }
    Ok(out)
}

fn materialize_raw_channel(fields: &[DecodedField], name: &str) -> Result<Option<Grid2D>, String> {
    let resolved = match name {
        "t2m" => find_surface_field(fields, &["2t", "t2m"], 2.0)
            .or_else(|| lowest_pressure_field(fields, &["t", "tmp"])),
        "u10" => find_surface_field(fields, &["10u", "u10"], 10.0)
            .or_else(|| lowest_pressure_field(fields, &["u", "ugrd"])),
        "v10" => find_surface_field(fields, &["10v", "v10"], 10.0)
            .or_else(|| lowest_pressure_field(fields, &["v", "vgrd"])),
        "mslp" => find_any_field(fields, &["msl", "prmsl", "mslet"]),
        "z500" => find_pressure_field(fields, &["gh", "z"], 500.0),
        "t850" => find_pressure_field(fields, &["t", "tmp"], 850.0),
        "u850" => find_pressure_field(fields, &["u", "ugrd"], 850.0),
        "v850" => find_pressure_field(fields, &["v", "vgrd"], 850.0),
        "d2m" => {
            if let Some(field) = find_surface_field(fields, &["2d", "d2m"], 2.0) {
                return Ok(Some(field.grid.clone()));
            }
            let t2m = find_surface_field(fields, &["2t", "t2m"], 2.0);
            let rh2m = find_surface_field(fields, &["r"], 2.0);
            if let (Some(t2m), Some(rh2m)) = (t2m, rh2m) {
                return Ok(Some(apply_binary_grid(
                    &t2m.grid,
                    &rh2m.grid,
                    |temperature_k, rh_pct| {
                        dewpoint_from_relative_humidity(temperature_k - 273.15, rh_pct) + 273.15
                    },
                )?));
            }
            let t_low = lowest_pressure_field(fields, &["t", "tmp"]);
            let rh_low = lowest_pressure_field(fields, &["r"]);
            if let (Some(t_low), Some(rh_low)) = (t_low, rh_low) {
                return Ok(Some(apply_binary_grid(
                    &t_low.grid,
                    &rh_low.grid,
                    |temperature_k, rh_pct| {
                        dewpoint_from_relative_humidity(temperature_k - 273.15, rh_pct) + 273.15
                    },
                )?));
            }
            None
        }
        "channel_min" | "channel_mean" | "channel_max" | "valid_hour_sin" | "valid_hour_cos"
        | "wind_speed" | "wind_speed_10m" | "wind_direction" | "wind_direction_10m"
        | "relative_humidity" | "rh2m" => {
            None
        }
        _ => find_any_field(fields, &[name]),
    };
    Ok(resolved
        .map(|field| field.grid.clone())
        .or_else(|| match name {
            "mslp" => find_surface_level_string(fields, &["sp", "pres", "ps"], "surface")
                .map(|field| field.grid.clone()),
            _ => None,
        }))
}

fn compute_pressure_map_diagnostics(
    fields: &[DecodedField],
) -> Result<Vec<(String, Grid2D)>, String> {
    let t850 = find_pressure_field(fields, &["t", "tmp"], 850.0).ok_or_else(|| {
        "missing 850 hPa temperature field for pressure-map diagnostics".to_string()
    })?;
    let u500 = find_pressure_field(fields, &["u", "ugrd"], 500.0)
        .ok_or_else(|| "missing 500 hPa u-wind field for pressure-map diagnostics".to_string())?;
    let v500 = find_pressure_field(fields, &["v", "vgrd"], 500.0)
        .ok_or_else(|| "missing 500 hPa v-wind field for pressure-map diagnostics".to_string())?;
    let u850 = find_pressure_field(fields, &["u", "ugrd"], 850.0)
        .ok_or_else(|| "missing 850 hPa u-wind field for pressure-map diagnostics".to_string())?;
    let v850 = find_pressure_field(fields, &["v", "vgrd"], 850.0)
        .ok_or_else(|| "missing 850 hPa v-wind field for pressure-map diagnostics".to_string())?;
    let (lat_grid, lon_grid) = lat_lon_mesh(u500)?;
    let (du500_dx, du500_dy) = geospatial_gradient(&u500.grid, &lat_grid, &lon_grid);
    let (dv500_dx, dv500_dy) = geospatial_gradient(&v500.grid, &lat_grid, &lon_grid);
    let (dt850_dx, dt850_dy) = geospatial_gradient(&t850.grid, &lat_grid, &lon_grid);

    let div500 = apply_binary_grid(&du500_dx, &dv500_dy, |a, b| a + b)?;
    let vort500 = apply_binary_grid(&dv500_dx, &du500_dy, |a, b| a - b)?;
    let tadv850 = apply_quaternary_grid(
        &u850.grid,
        &v850.grid,
        &dt850_dx,
        &dt850_dy,
        |u, v, dtdx, dtdy| -(u * dtdx + v * dtdy),
    )?;
    let theta850 = apply_unary_grid(&t850.grid, |temperature_k| {
        potential_temperature(850.0, temperature_k - 273.15)
    });

    Ok(vec![
        ("div500".to_string(), div500),
        ("vort500".to_string(), vort500),
        ("tadv850".to_string(), tadv850),
        ("theta850".to_string(), theta850),
    ])
}

fn compute_profile_diagnostics(
    fields: &[DecodedField],
    requested: &HashSet<String>,
) -> Result<Vec<(String, Grid2D)>, String> {
    let temperature_levels = collect_pressure_levels(fields, &["t", "tmp"])?;
    let humidity_levels = collect_pressure_levels(fields, &["r"])?;
    let u_levels = collect_pressure_levels(fields, &["u", "ugrd"])?;
    let v_levels = collect_pressure_levels(fields, &["v", "vgrd"])?;
    let height_levels = collect_pressure_levels(fields, &["gh", "z"])?;
    let temperature_2m = find_surface_field(fields, &["2t", "t2m"], 2.0)
        .or_else(|| temperature_levels.first().map(|level| level.field))
        .ok_or_else(|| {
            "missing 2 m or lowest pressure-level temperature field for profile diagnostics"
                .to_string()
        })?;
    let relative_humidity_2m = find_surface_field(fields, &["r"], 2.0)
        .or_else(|| humidity_levels.first().map(|level| level.field))
        .ok_or_else(|| {
            "missing 2 m or lowest pressure-level relative humidity field for profile diagnostics"
                .to_string()
        })?;
    let u_10m = find_surface_field(fields, &["10u", "u10"], 10.0)
        .or_else(|| u_levels.first().map(|level| level.field))
        .ok_or_else(|| {
            "missing 10 m or lowest pressure-level u-wind field for profile diagnostics".to_string()
        })?;
    let v_10m = find_surface_field(fields, &["10v", "v10"], 10.0)
        .or_else(|| v_levels.first().map(|level| level.field))
        .ok_or_else(|| {
            "missing 10 m or lowest pressure-level v-wind field for profile diagnostics".to_string()
        })?;
    let height_surface = find_surface_field(fields, &["gh", "z"], 0.0)
        .or_else(|| find_surface_level_string(fields, &["gh", "z"], "surface"))
        .or_else(|| height_levels.first().map(|level| level.field))
        .ok_or_else(|| {
            "missing surface or lowest pressure-level height field for profile diagnostics"
                .to_string()
        })?;
    let pressure_surface = find_surface_level_string(fields, &["sp", "pres", "ps"], "surface");

    let base = temperature_2m;
    let mut outputs = requested
        .iter()
        .map(|name| {
            (
                name.clone(),
                filled_grid(base.grid.nx, base.grid.ny, f64::NAN),
            )
        })
        .collect::<HashMap<_, _>>();

    for y in 0..base.grid.ny {
        for x in 0..base.grid.nx {
            let psfc_pa = pressure_surface
                .map(|field| field.grid.get(x, y))
                .unwrap_or_else(|| pressure_levels_to_surface_pa(&temperature_levels));
            let t2m_k = temperature_2m.grid.get(x, y);
            let rh2m_pct = relative_humidity_2m.grid.get(x, y);
            let u10 = u_10m.grid.get(x, y);
            let v10 = v_10m.grid.get(x, y);
            let surface_height_m = height_surface.grid.get(x, y);
            if ![psfc_pa, t2m_k, rh2m_pct, u10, v10, surface_height_m]
                .iter()
                .all(|value| value.is_finite())
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
                let temperature_k = temperature_levels[idx].field.grid.get(x, y);
                let rh_pct = humidity_levels[idx].field.grid.get(x, y);
                let u = u_levels[idx].field.grid.get(x, y);
                let v = v_levels[idx].field.grid.get(x, y);
                let height_m = height_levels[idx].field.grid.get(x, y);
                if ![temperature_k, rh_pct, u, v, height_m]
                    .iter()
                    .all(|value| value.is_finite())
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
            let (_, _, lcl_height_m, lfc_height_m) = cape_cin(
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
            let (_, _, srh01_total) = storm_relative_helicity(
                &u_full,
                &v_full,
                &height_full,
                1000.0,
                right_mover.0,
                right_mover.1,
            );
            let (_, _, srh03_total) = storm_relative_helicity(
                &u_full,
                &v_full,
                &height_full,
                3000.0,
                right_mover.0,
                right_mover.1,
            );
            let stp_value =
                significant_tornado_parameter(sb_cape, lcl_height_m, srh01_total, shear_mag_ms);
            let scp_value = supercell_composite_parameter(mu_cape, srh01_total, shear_mag_ms);
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
            let theta_e_value = equivalent_potential_temperature(psfc_hpa, t2m_c, td2m_c);
            let wet_bulb_value = wet_bulb_temperature(psfc_hpa, t2m_c, td2m_c) + 273.15;
            let wet_bulb_theta_value = wet_bulb_potential_temperature(psfc_hpa, t2m_c, td2m_c);
            let dcape_value = downdraft_cape(&pressure_full, &temperature_full_c, &dewpoint_full_c);

            set_output(&mut outputs, "sbcape", x, y, sb_cape);
            set_output(&mut outputs, "sbcin", x, y, sb_cin);
            set_output(&mut outputs, "mlcape", x, y, ml_cape);
            set_output(&mut outputs, "mlcin", x, y, ml_cin);
            set_output(&mut outputs, "mucape", x, y, mu_cape);
            set_output(&mut outputs, "mucin", x, y, mu_cin);
            set_output(&mut outputs, "shear06", x, y, shear_mag_ms);
            set_output(&mut outputs, "srh01", x, y, srh01_total);
            set_output(&mut outputs, "srh03", x, y, srh03_total);
            set_output(&mut outputs, "stp", x, y, stp_value);
            set_output(&mut outputs, "scp", x, y, scp_value);
            set_output(&mut outputs, "pwat", x, y, pwat_value);
            set_output(&mut outputs, "lifted_index", x, y, li_value);
            set_output(&mut outputs, "li", x, y, li_value);
            set_output(&mut outputs, "showalter_index", x, y, showalter_value);
            set_output(&mut outputs, "showalter", x, y, showalter_value);
            set_output(&mut outputs, "k_index", x, y, kindex_value);
            set_output(&mut outputs, "total_totals", x, y, total_totals_value);
            set_output(&mut outputs, "theta_e", x, y, theta_e_value);
            set_output(&mut outputs, "wet_bulb", x, y, wet_bulb_value);
            set_output(
                &mut outputs,
                "wet_bulb_potential_temperature",
                x,
                y,
                wet_bulb_theta_value,
            );
            set_output(&mut outputs, "lcl_height", x, y, lcl_height_m);
            set_output(&mut outputs, "lfc_height", x, y, lfc_height_m);
            set_output(&mut outputs, "dcape", x, y, dcape_value);

            for name in requested {
                if let Some(depth_m) = parse_custom_srh_depth_m(name) {
                    let (_, _, srh_total) = storm_relative_helicity(
                        &u_full,
                        &v_full,
                        &height_full,
                        depth_m,
                        right_mover.0,
                        right_mover.1,
                    );
                    set_output(&mut outputs, name, x, y, srh_total);
                }
            }
        }
    }

    Ok(outputs.into_iter().collect())
}

fn collect_pressure_levels<'a>(
    fields: &'a [DecodedField],
    aliases: &[&str],
) -> Result<Vec<LevelField<'a>>, String> {
    let mut levels = fields
        .iter()
        .filter(|field| matches_pressure_level(&field.descriptor, aliases, None))
        .map(|field| LevelField {
            pressure_hpa: normalized_pressure_hpa(field.descriptor.level_value.unwrap_or_default()),
            field,
        })
        .collect::<Vec<_>>();
    levels.sort_by(|a, b| {
        b.pressure_hpa
            .partial_cmp(&a.pressure_hpa)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if levels.is_empty() {
        return Err(format!(
            "missing pressure-level field for aliases [{}]",
            aliases.join(", ")
        ));
    }
    Ok(levels)
}

fn find_pressure_field<'a>(
    fields: &'a [DecodedField],
    aliases: &[&str],
    pressure_hpa: f64,
) -> Option<&'a DecodedField> {
    fields
        .iter()
        .find(|field| matches_pressure_level(&field.descriptor, aliases, Some(pressure_hpa)))
}

fn lowest_pressure_field<'a>(
    fields: &'a [DecodedField],
    aliases: &[&str],
) -> Option<&'a DecodedField> {
    collect_pressure_levels(fields, aliases)
        .ok()
        .and_then(|levels| levels.into_iter().next().map(|level| level.field))
}

fn find_surface_field<'a>(
    fields: &'a [DecodedField],
    aliases: &[&str],
    level_m: f64,
) -> Option<&'a DecodedField> {
    fields
        .iter()
        .find(|field| matches_surface_level(&field.descriptor, aliases, level_m))
}

fn find_surface_level_string<'a>(
    fields: &'a [DecodedField],
    aliases: &[&str],
    level: &str,
) -> Option<&'a DecodedField> {
    fields.iter().find(|field| {
        aliases
            .iter()
            .any(|alias| field.descriptor.variable.eq_ignore_ascii_case(alias))
            && field.descriptor.level.eq_ignore_ascii_case(level)
    })
}

fn find_any_field<'a>(fields: &'a [DecodedField], aliases: &[&str]) -> Option<&'a DecodedField> {
    fields.iter().find(|field| {
        aliases
            .iter()
            .any(|alias| field.descriptor.variable.eq_ignore_ascii_case(alias))
    })
}

fn matches_pressure_level(
    message: &MessageDescriptor,
    aliases: &[&str],
    expected_pressure_hpa: Option<f64>,
) -> bool {
    if !aliases
        .iter()
        .any(|alias| message.variable.eq_ignore_ascii_case(alias))
        || message.level_type != Some(100)
    {
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

fn matches_surface_level(message: &MessageDescriptor, aliases: &[&str], level_m: f64) -> bool {
    aliases
        .iter()
        .any(|alias| message.variable.eq_ignore_ascii_case(alias))
        && message.level_type == Some(103)
        && message
            .level_value
            .is_some_and(|value| (value - level_m).abs() < 1.0e-6)
}

fn normalized_pressure_hpa(value: f64) -> f64 {
    value / 100.0
}

fn lat_lon_mesh(field: &DecodedField) -> Result<(Grid2D, Grid2D), String> {
    let x_axis = field
        .x_axis
        .as_ref()
        .ok_or_else(|| "decoded field missing longitude axis".to_string())?;
    let y_axis = field
        .y_axis
        .as_ref()
        .ok_or_else(|| "decoded field missing latitude axis".to_string())?;
    let mut lat = Grid2D::zeros(field.grid.nx, field.grid.ny);
    let mut lon = Grid2D::zeros(field.grid.nx, field.grid.ny);
    for y in 0..field.grid.ny {
        for x in 0..field.grid.nx {
            lat.set(x, y, y_axis.values[y]);
            lon.set(x, y, x_axis.values[x]);
        }
    }
    Ok((lat, lon))
}

fn apply_unary_grid(grid: &Grid2D, func: impl Fn(f64) -> f64) -> Grid2D {
    let mut out = Grid2D::zeros(grid.nx, grid.ny);
    for y in 0..grid.ny {
        for x in 0..grid.nx {
            out.set(x, y, func(grid.get(x, y)));
        }
    }
    out
}

fn apply_binary_grid(
    a: &Grid2D,
    b: &Grid2D,
    func: impl Fn(f64, f64) -> f64,
) -> Result<Grid2D, String> {
    if a.nx != b.nx || a.ny != b.ny {
        return Err("binary grid operation requires matching shapes".to_string());
    }
    let mut out = Grid2D::zeros(a.nx, a.ny);
    for y in 0..a.ny {
        for x in 0..a.nx {
            out.set(x, y, func(a.get(x, y), b.get(x, y)));
        }
    }
    Ok(out)
}

fn apply_quaternary_grid(
    a: &Grid2D,
    b: &Grid2D,
    c: &Grid2D,
    d: &Grid2D,
    func: impl Fn(f64, f64, f64, f64) -> f64,
) -> Result<Grid2D, String> {
    if a.nx != b.nx || a.nx != c.nx || a.nx != d.nx || a.ny != b.ny || a.ny != c.ny || a.ny != d.ny
    {
        return Err("quaternary grid operation requires matching shapes".to_string());
    }
    let mut out = Grid2D::zeros(a.nx, a.ny);
    for y in 0..a.ny {
        for x in 0..a.nx {
            out.set(
                x,
                y,
                func(a.get(x, y), b.get(x, y), c.get(x, y), d.get(x, y)),
            );
        }
    }
    Ok(out)
}

fn filled_grid(nx: usize, ny: usize, value: f64) -> Grid2D {
    Grid2D::new(nx, ny, vec![value; nx * ny])
}

fn set_output(outputs: &mut HashMap<String, Grid2D>, name: &str, x: usize, y: usize, value: f64) {
    if let Some(grid) = outputs.get_mut(name) {
        grid.set(x, y, value);
    }
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

fn parse_custom_srh_depth_m(name: &str) -> Option<f64> {
    let lower = name.to_ascii_lowercase();
    let suffix = lower.strip_prefix("custom_srh_")?;
    if let Some(value) = suffix.strip_suffix("km") {
        return value.parse::<f64>().ok().map(|km| km * 1000.0);
    }
    if let Some(value) = suffix.strip_suffix('m') {
        return value.parse::<f64>().ok();
    }
    None
}

fn aggregate_channel_stats(
    outputs: &HashMap<String, Grid2D>,
    fields: &[DecodedField],
) -> Option<(f64, f64, f64)> {
    let mut count = 0usize;
    let mut sum = 0.0;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;

    for (name, grid) in outputs {
        if matches!(
            name.as_str(),
            "channel_min" | "channel_mean" | "channel_max" | "valid_hour_sin" | "valid_hour_cos"
        ) {
            continue;
        }
        for value in &grid.values {
            if !value.is_finite() {
                continue;
            }
            count += 1;
            sum += *value;
            min = min.min(*value);
            max = max.max(*value);
        }
    }

    if count == 0 {
        for field in fields {
            if let Some((field_min, field_mean, field_max)) = field.min_mean_max() {
                count += 1;
                sum += field_mean;
                min = min.min(field_min);
                max = max.max(field_max);
            }
        }
    }

    if count == 0 {
        None
    } else {
        Some((min, sum / count as f64, max))
    }
}

fn infer_valid_hour_utc(fields: &[DecodedField]) -> Option<u32> {
    let descriptor = &fields.first()?.descriptor;
    let reference_time = descriptor.reference_time.as_ref()?;
    let reference = DateTime::parse_from_rfc3339(reference_time).ok()?;
    let mut valid_time = reference.with_timezone(&Utc);
    if let Some(forecast_hours) = descriptor.forecast_time_value {
        let unit = descriptor
            .forecast_time_unit
            .as_deref()
            .unwrap_or("hour")
            .to_ascii_lowercase();
        if unit.contains("hour") {
            valid_time += Duration::hours(forecast_hours as i64);
        }
    }
    Some(valid_time.hour())
}

fn pressure_levels_to_surface_pa(levels: &[LevelField<'_>]) -> f64 {
    levels
        .first()
        .map(|level| level.pressure_hpa * 100.0)
        .unwrap_or(1000.0 * 100.0)
}

fn inferred_channel_level(name: &str) -> String {
    match name.to_ascii_lowercase().as_str() {
        "t2m" | "d2m" => "2 m above ground".to_string(),
        "u10" | "v10" => "10 m above ground".to_string(),
        "t850" | "theta850" | "u850" | "v850" | "tadv850" => "850 hPa".to_string(),
        "z500" | "vort500" | "div500" => "500 hPa".to_string(),
        "theta_e" | "wet_bulb" | "wet_bulb_potential_temperature" => "surface parcel".to_string(),
        "wind_speed" | "wind_speed_10m" | "wind_direction" | "wind_direction_10m" => {
            "10 m above ground".to_string()
        }
        "relative_humidity" | "rh2m" => "2 m above ground".to_string(),
        "sbcape" | "sbcin" | "mlcape" | "mlcin" | "mucape" | "mucin" | "srh01" | "srh03"
        | "shear06" | "stp" | "scp" | "pwat" | "lifted_index" | "li" | "showalter_index"
        | "showalter" | "k_index" | "total_totals" | "lcl_height" | "lfc_height" | "dcape" => {
            "derived".to_string()
        }
        other => other.to_string(),
    }
}

fn sanitize_channel_stem(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    out
}

