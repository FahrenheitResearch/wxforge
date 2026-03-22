use std::fs;
use std::path::{Path, PathBuf};

use clap::Parser;
use serde_json::json;
use wx_calc::{geospatial_gradient, potential_temperature};
use wx_export::ExportEngine;
use wx_grib::{DecodedField, GribEngine};
use wx_types::Grid2D;

#[derive(Debug, Parser)]
#[command(name = "verify-gfs-maps")]
#[command(about = "Decode GFS verification fields and dump wxforge map products as NPY")]
struct Cli {
    #[arg(long)]
    u500: PathBuf,
    #[arg(long)]
    v500: PathBuf,
    #[arg(long)]
    t850: PathBuf,
    #[arg(long)]
    u850: PathBuf,
    #[arg(long)]
    v850: PathBuf,
    #[arg(long)]
    latitude_npy: Option<PathBuf>,
    #[arg(long)]
    longitude_npy: Option<PathBuf>,
    #[arg(long)]
    output_dir: PathBuf,
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    fs::create_dir_all(&cli.output_dir)
        .map_err(|err| format!("failed to create '{}': {err}", cli.output_dir.display()))?;

    let grib = GribEngine::new();
    let export = ExportEngine::new();

    let u500 = grib.decode_file_message(&cli.u500, 1)?;
    let v500 = grib.decode_file_message(&cli.v500, 1)?;
    let t850 = grib.decode_file_message(&cli.t850, 1)?;
    let u850 = grib.decode_file_message(&cli.u850, 1)?;
    let v850 = grib.decode_file_message(&cli.v850, 1)?;

    ensure_matching_grid(&u500, &v500, "u500", "v500")?;
    ensure_matching_grid(&u500, &t850, "u500", "t850")?;
    ensure_matching_grid(&u500, &u850, "u500", "u850")?;
    ensure_matching_grid(&u500, &v850, "u500", "v850")?;

    let (lat_grid, lon_grid) = lat_lon_mesh(
        &u500,
        cli.latitude_npy.as_deref(),
        cli.longitude_npy.as_deref(),
    )?;

    let (du500_dx, du500_dy) = geospatial_gradient(&u500.grid, &lat_grid, &lon_grid);
    let (dv500_dx, dv500_dy) = geospatial_gradient(&v500.grid, &lat_grid, &lon_grid);
    let (dt850_dx, dt850_dy) = geospatial_gradient(&t850.grid, &lat_grid, &lon_grid);

    let div500 = combine_binary(&du500_dx, &dv500_dy, |a, b| a + b);
    let vort500 = combine_binary(&dv500_dx, &du500_dy, |a, b| a - b);
    let tadv850 = combine_ternary(
        &u850.grid,
        &v850.grid,
        &dt850_dx,
        &dt850_dy,
        |u, v, dtdx, dtdy| -(u * dtdx + v * dtdy),
    );
    let theta850 = apply_scalar_map(&t850.grid, |temperature_k| {
        potential_temperature(850.0, temperature_k - 273.15)
    });

    write_grid(&export, &cli.output_dir, "longitude", &lon_grid)?;
    write_grid(&export, &cli.output_dir, "latitude", &lat_grid)?;
    write_grid(&export, &cli.output_dir, "u500", &u500.grid)?;
    write_grid(&export, &cli.output_dir, "v500", &v500.grid)?;
    write_grid(&export, &cli.output_dir, "t850", &t850.grid)?;
    write_grid(&export, &cli.output_dir, "u850", &u850.grid)?;
    write_grid(&export, &cli.output_dir, "v850", &v850.grid)?;
    write_grid(&export, &cli.output_dir, "div500", &div500)?;
    write_grid(&export, &cli.output_dir, "vort500", &vort500)?;
    write_grid(&export, &cli.output_dir, "tadv850", &tadv850)?;
    write_grid(&export, &cli.output_dir, "theta850", &theta850)?;

    let manifest = json!({
        "grid": {
            "nx": u500.grid.nx,
            "ny": u500.grid.ny,
        },
        "fields": {
            "u500": field_summary(&u500),
            "v500": field_summary(&v500),
            "t850": field_summary(&t850),
            "u850": field_summary(&u850),
            "v850": field_summary(&v850),
            "div500": grid_summary(&div500, "1/s"),
            "vort500": grid_summary(&vort500, "1/s"),
            "tadv850": grid_summary(&tadv850, "K/s"),
            "theta850": grid_summary(&theta850, "K"),
        }
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

fn ensure_matching_grid(
    a: &DecodedField,
    b: &DecodedField,
    a_name: &str,
    b_name: &str,
) -> Result<(), String> {
    if a.grid.nx != b.grid.nx || a.grid.ny != b.grid.ny {
        return Err(format!(
            "{a_name} grid {}x{} does not match {b_name} grid {}x{}",
            a.grid.nx, a.grid.ny, b.grid.nx, b.grid.ny
        ));
    }
    if a.grid_spec != b.grid_spec {
        return Err(format!(
            "{a_name} grid spec {:?} does not match {b_name} grid spec {:?}",
            a.grid_spec, b.grid_spec
        ));
    }
    if let (Some(ax), Some(bx), Some(ay), Some(by)) = (
        a.x_axis.as_ref(),
        b.x_axis.as_ref(),
        a.y_axis.as_ref(),
        b.y_axis.as_ref(),
    ) {
        if ax.values.len() != bx.values.len() || ay.values.len() != by.values.len() {
            return Err(format!("{a_name} axis lengths do not match {b_name}"));
        }
        for i in 0..ax.values.len() {
            if (ax.values[i] - bx.values[i]).abs() > 1.0e-6 {
                return Err(format!(
                    "{a_name} longitude axis differs from {b_name} at index {i}"
                ));
            }
        }
        for i in 0..ay.values.len() {
            if (ay.values[i] - by.values[i]).abs() > 1.0e-6 {
                return Err(format!(
                    "{a_name} latitude axis differs from {b_name} at index {i}"
                ));
            }
        }
    }
    Ok(())
}

fn lat_lon_mesh(
    field: &DecodedField,
    latitude_npy: Option<&Path>,
    longitude_npy: Option<&Path>,
) -> Result<(Grid2D, Grid2D), String> {
    if let (Some(lat_path), Some(lon_path)) = (latitude_npy, longitude_npy) {
        let lat = read_npy_grid(lat_path)?;
        let lon = read_npy_grid(lon_path)?;
        if lat.nx != field.grid.nx
            || lat.ny != field.grid.ny
            || lon.nx != field.grid.nx
            || lon.ny != field.grid.ny
        {
            return Err(format!(
                "coordinate mesh shape {}x{} / {}x{} does not match decoded field {}x{}",
                lon.nx, lon.ny, lat.nx, lat.ny, field.grid.nx, field.grid.ny
            ));
        }
        return Ok((lat, lon));
    }

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

fn read_npy_grid(path: &Path) -> Result<Grid2D, String> {
    let bytes =
        fs::read(path).map_err(|err| format!("failed to read '{}': {err}", path.display()))?;
    if bytes.len() < 12 || &bytes[0..6] != b"\x93NUMPY" {
        return Err(format!("'{}' is not a valid NPY file", path.display()));
    }
    let version_major = bytes[6];
    let (header_len, data_start) = match version_major {
        1 => {
            let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (header_len, 10usize)
        }
        2 => {
            if bytes.len() < 14 {
                return Err(format!(
                    "'{}' has a truncated NPY v2 header",
                    path.display()
                ));
            }
            let header_len =
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (header_len, 12usize)
        }
        _ => {
            return Err(format!(
                "'{}' uses unsupported NPY version {}.{}",
                path.display(),
                version_major,
                bytes[7]
            ))
        }
    };
    let header_end = data_start + header_len;
    if bytes.len() < header_end {
        return Err(format!("'{}' has a truncated NPY header", path.display()));
    }
    let header = std::str::from_utf8(&bytes[data_start..header_end])
        .map_err(|err| format!("failed to decode NPY header '{}': {err}", path.display()))?;
    if !header.contains("'descr': '<f4'") && !header.contains("\"descr\": \"<f4\"") {
        return Err(format!(
            "'{}' must be a little-endian float32 NPY array",
            path.display()
        ));
    }
    if header.contains("True") {
        return Err(format!(
            "'{}' uses Fortran order, which this verifier does not support",
            path.display()
        ));
    }

    let shape_start = header
        .find('(')
        .ok_or_else(|| format!("'{}' NPY header is missing a shape tuple", path.display()))?;
    let shape_end = header[shape_start + 1..]
        .find(')')
        .map(|idx| shape_start + 1 + idx)
        .ok_or_else(|| {
            format!(
                "'{}' NPY header has an unterminated shape tuple",
                path.display()
            )
        })?;
    let dims: Vec<usize> = header[shape_start + 1..shape_end]
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value.parse::<usize>().map_err(|err| {
                format!(
                    "failed to parse NPY shape value '{value}' in '{}': {err}",
                    path.display()
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if dims.len() != 2 {
        return Err(format!(
            "'{}' must be a 2D NPY array, found shape {:?}",
            path.display(),
            dims
        ));
    }
    let ny = dims[0];
    let nx = dims[1];
    let expected_len = nx
        .checked_mul(ny)
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| format!("'{}' NPY shape is too large", path.display()))?;
    let data = &bytes[header_end..];
    if data.len() != expected_len {
        return Err(format!(
            "'{}' payload length {} does not match expected {}",
            path.display(),
            data.len(),
            expected_len
        ));
    }
    let mut values = Vec::with_capacity(nx * ny);
    for chunk in data.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64);
    }
    Ok(Grid2D::new(nx, ny, values))
}

fn combine_binary(a: &Grid2D, b: &Grid2D, func: impl Fn(f64, f64) -> f64) -> Grid2D {
    let values = a
        .values
        .iter()
        .zip(b.values.iter())
        .map(|(&lhs, &rhs)| func(lhs, rhs))
        .collect();
    Grid2D::new(a.nx, a.ny, values)
}

fn combine_ternary(
    a: &Grid2D,
    b: &Grid2D,
    c: &Grid2D,
    d: &Grid2D,
    func: impl Fn(f64, f64, f64, f64) -> f64,
) -> Grid2D {
    let values = a
        .values
        .iter()
        .zip(b.values.iter())
        .zip(c.values.iter())
        .zip(d.values.iter())
        .map(|(((&va, &vb), &vc), &vd)| func(va, vb, vc, vd))
        .collect();
    Grid2D::new(a.nx, a.ny, values)
}

fn apply_scalar_map(grid: &Grid2D, func: impl Fn(f64) -> f64) -> Grid2D {
    let values = grid.values.iter().copied().map(func).collect();
    Grid2D::new(grid.nx, grid.ny, values)
}

fn write_grid(
    export: &ExportEngine,
    out_dir: &Path,
    name: &str,
    grid: &Grid2D,
) -> Result<(), String> {
    export.write_npy_f32_grid(out_dir.join(format!("{name}.npy")), grid)
}

fn grid_summary(grid: &Grid2D, units: &str) -> serde_json::Value {
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
        "units": units,
        "min": if count > 0 { min } else { f64::NAN },
        "mean": if count > 0 { sum / count as f64 } else { f64::NAN },
        "max": if count > 0 { max } else { f64::NAN },
    })
}

fn field_summary(field: &DecodedField) -> serde_json::Value {
    let (min, mean, max) = field
        .min_mean_max()
        .unwrap_or((f64::NAN, f64::NAN, f64::NAN));
    json!({
        "variable": field.descriptor.variable,
        "level": field.descriptor.level,
        "units": field.descriptor.units,
        "min": min,
        "mean": mean,
        "max": max,
    })
}
