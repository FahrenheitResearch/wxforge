use std::error::Error;
use std::f64::consts::PI;
use std::fmt;

use wx_types::Grid2D;

use super::{
    absolute_vorticity, cape_cin, drylift, equivalent_potential_temperature, geodesic,
    interpolate_to_points, parcel_profile_with_lcl, satlift, saturation_vapor_pressure,
    virtual_temperature, wobf, G, ROCP, ZEROCNK,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidSoundingError {
    pub message: String,
}

impl InvalidSoundingError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for InvalidSoundingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for InvalidSoundingError {}

#[derive(Debug, Clone, PartialEq)]
pub struct CalcCoordinate {
    pub name: String,
    pub values: Vec<f64>,
    pub units: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CalcVariable {
    pub name: String,
    pub dims: Vec<String>,
    pub values: Vec<f64>,
    pub units: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CalcDataset {
    pub dims: Vec<(String, usize)>,
    pub coords: Vec<CalcCoordinate>,
    pub variables: Vec<CalcVariable>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GridSpec {
    pub nx: usize,
    pub ny: usize,
    pub lat1: f64,
    pub lon1: f64,
    pub lat2: f64,
    pub lon2: f64,
    pub dlat: f64,
    pub dlon: f64,
}

impl GridSpec {
    pub fn regular(
        lat_min: f64,
        lat_max: f64,
        lon_min: f64,
        lon_max: f64,
        resolution: f64,
    ) -> Self {
        let ny = ((lat_max - lat_min) / resolution).round() as usize + 1;
        let nx = ((lon_max - lon_min) / resolution).round() as usize + 1;
        let dlat = if ny > 1 {
            (lat_max - lat_min) / (ny - 1) as f64
        } else {
            resolution
        };
        let dlon = if nx > 1 {
            (lon_max - lon_min) / (nx - 1) as f64
        } else {
            resolution
        };
        Self {
            nx,
            ny,
            lat1: lat_min,
            lon1: lon_min,
            lat2: lat_max,
            lon2: lon_max,
            dlat,
            dlon,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpMethod {
    NearestNeighbor,
    Bilinear,
    Bicubic,
    Budget,
}

fn extract_column(data: &[f64], nx: usize, ny: usize, nz: usize, y: usize, x: usize) -> Vec<f64> {
    let nxy = nx * ny;
    let mut out = Vec::with_capacity(nz);
    for k in 0..nz {
        out.push(data[k * nxy + y * nx + x]);
    }
    out
}

fn dewpoint_from_q(q: f64, p_hpa: f64) -> f64 {
    let q = q.max(1.0e-10);
    let e = (q * p_hpa / (0.622 + q)).max(1.0e-10);
    let ln_e = (e / 6.112).ln();
    (243.5 * ln_e) / (17.67 - ln_e)
}

fn interpolate_monotonic_profile(target: f64, axis: &[f64], values: &[f64], log_axis: bool) -> f64 {
    let n = axis.len().min(values.len());
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return values[0];
    }

    let ascending = axis[n - 1] >= axis[0];
    if (ascending && target <= axis[0]) || (!ascending && target >= axis[0]) {
        return values[0];
    }
    if (ascending && target >= axis[n - 1]) || (!ascending && target <= axis[n - 1]) {
        return values[n - 1];
    }

    for i in 0..n - 1 {
        let a0 = axis[i];
        let a1 = axis[i + 1];
        let in_segment = if ascending {
            a0 <= target && target <= a1
        } else {
            a1 <= target && target <= a0
        };
        if !in_segment {
            continue;
        }
        let frac = if log_axis {
            let ln_t = target.ln();
            let ln_0 = a0.ln();
            let ln_1 = a1.ln();
            if (ln_1 - ln_0).abs() < 1.0e-12 {
                0.0
            } else {
                (ln_t - ln_0) / (ln_1 - ln_0)
            }
        } else if (a1 - a0).abs() < 1.0e-12 {
            0.0
        } else {
            (target - a0) / (a1 - a0)
        };
        return values[i] + frac * (values[i + 1] - values[i]);
    }

    values[n - 1]
}

fn normalize_parcel_type(parcel_type: &str) -> &str {
    match parcel_type {
        "surface" | "surface_based" | "sb" => "sb",
        "mixed_layer" | "ml" => "mixed_layer",
        "most_unstable" | "mu" => "most_unstable",
        _ => parcel_type,
    }
}

fn target_grid_points(target: &GridSpec) -> (Vec<f64>, Vec<f64>) {
    let mut lats = Vec::with_capacity(target.nx * target.ny);
    let mut lons = Vec::with_capacity(target.nx * target.ny);
    for y in 0..target.ny {
        let lat = target.lat1 + y as f64 * target.dlat;
        for x in 0..target.nx {
            lats.push(lat);
            lons.push(target.lon1 + x as f64 * target.dlon);
        }
    }
    (lats, lons)
}

fn infer_source_dims(lats: &[f64], lons: &[f64], values_len: usize) -> (usize, usize) {
    if !lats.is_empty() && !lons.is_empty() && lats.len() * lons.len() == values_len {
        return (lons.len(), lats.len());
    }
    if lats.len() == values_len && lons.len() == values_len && values_len > 1 {
        for i in 1..values_len {
            if (lons[i] - lons[0]).abs() < 1.0e-8 && i > 1 {
                let nx = i;
                let ny = values_len / nx;
                if nx * ny == values_len {
                    return (nx, ny);
                }
                break;
            }
        }
    }
    (values_len, 1)
}

fn fractional_index(target: f64, grid: &[f64]) -> f64 {
    if grid.len() < 2 {
        return 0.0;
    }
    let ascending = grid[grid.len() - 1] > grid[0];
    if ascending {
        if target <= grid[0] {
            return 0.0;
        }
        if target >= grid[grid.len() - 1] {
            return (grid.len() - 1) as f64;
        }
    } else {
        if target >= grid[0] {
            return 0.0;
        }
        if target <= grid[grid.len() - 1] {
            return (grid.len() - 1) as f64;
        }
    }

    let mut low = 0usize;
    let mut high = grid.len() - 1;
    while high - low > 1 {
        let middle = (low + high) / 2;
        let in_lower = if ascending {
            grid[middle] <= target
        } else {
            grid[middle] >= target
        };
        if in_lower {
            low = middle;
        } else {
            high = middle;
        }
    }

    let frac = (target - grid[low]) / (grid[high] - grid[low]);
    low as f64 + frac
}

fn bilinear_regular(
    values: &[f64],
    src_lats: &[f64],
    src_lons: &[f64],
    nx: usize,
    ny: usize,
    target_lat: f64,
    target_lon: f64,
) -> f64 {
    if nx == 0 || ny == 0 || values.len() != nx * ny {
        return f64::NAN;
    }
    if nx == 1 && ny == 1 {
        return values[0];
    }
    let fx = fractional_index(target_lon, src_lons);
    let fy = fractional_index(target_lat, src_lats);
    let x0 = fx.floor().max(0.0) as usize;
    let y0 = fy.floor().max(0.0) as usize;
    let x1 = (x0 + 1).min(nx.saturating_sub(1));
    let y1 = (y0 + 1).min(ny.saturating_sub(1));
    let dx = fx - x0 as f64;
    let dy = fy - y0 as f64;

    let idx = |x: usize, y: usize| y * nx + x;
    let v00 = values[idx(x0, y0)];
    let v10 = values[idx(x1, y0)];
    let v01 = values[idx(x0, y1)];
    let v11 = values[idx(x1, y1)];
    let top = v00 * (1.0 - dx) + v10 * dx;
    let bottom = v01 * (1.0 - dx) + v11 * dx;
    top * (1.0 - dy) + bottom * dy
}

fn nearest_regular(
    values: &[f64],
    src_lats: &[f64],
    src_lons: &[f64],
    nx: usize,
    ny: usize,
    target_lat: f64,
    target_lon: f64,
) -> f64 {
    if nx == 0 || ny == 0 || values.len() != nx * ny {
        return f64::NAN;
    }
    let x = fractional_index(target_lon, src_lons).round() as usize;
    let y = fractional_index(target_lat, src_lats).round() as usize;
    values[y.min(ny - 1) * nx + x.min(nx - 1)]
}

fn grid2d_from_flat(nx: usize, ny: usize, values: &[f64]) -> Grid2D {
    Grid2D::new(nx, ny, values.to_vec())
}

pub fn potential_vorticity_baroclinic(
    potential_temp: &[f64],
    pressure: &[f64; 2],
    theta_below: &[f64],
    theta_above: &[f64],
    u: &[f64],
    v: &[f64],
    lats: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
) -> Vec<f64> {
    let n = nx * ny;
    assert_eq!(potential_temp.len(), n, "potential_temp length mismatch");
    assert_eq!(theta_below.len(), n, "theta_below length mismatch");
    assert_eq!(theta_above.len(), n, "theta_above length mismatch");
    assert_eq!(u.len(), n, "u length mismatch");
    assert_eq!(v.len(), n, "v length mismatch");
    assert_eq!(lats.len(), n, "lats length mismatch");

    let dp = pressure[1] - pressure[0];
    assert!(dp.abs() > 1.0e-10, "pressure levels must differ");

    let u_grid = grid2d_from_flat(nx, ny, u);
    let v_grid = grid2d_from_flat(nx, ny, v);
    let lat_grid = grid2d_from_flat(nx, ny, lats);
    let abs_vort = absolute_vorticity(&u_grid, &v_grid, &lat_grid, dx, dy);

    let mut pv = vec![0.0; n];
    for i in 0..n {
        let _ = potential_temp[i];
        let dthetadp = (theta_above[i] - theta_below[i]) / dp;
        pv[i] = -G * abs_vort.values[i] * dthetadp;
    }
    pv
}

pub fn potential_vorticity_barotropic(
    heights: &[f64],
    u: &[f64],
    v: &[f64],
    lats: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
) -> Vec<f64> {
    let n = nx * ny;
    assert_eq!(heights.len(), n, "heights length mismatch");
    assert_eq!(u.len(), n, "u length mismatch");
    assert_eq!(v.len(), n, "v length mismatch");
    assert_eq!(lats.len(), n, "lats length mismatch");

    let u_grid = grid2d_from_flat(nx, ny, u);
    let v_grid = grid2d_from_flat(nx, ny, v);
    let lat_grid = grid2d_from_flat(nx, ny, lats);
    let abs_vort = absolute_vorticity(&u_grid, &v_grid, &lat_grid, dx, dy);

    let mut pv = vec![f64::NAN; n];
    for i in 0..n {
        if heights[i].abs() >= 1.0e-30 {
            pv[i] = abs_vort.values[i] / heights[i];
        }
    }
    pv
}

pub fn significant_tornado_parameter(
    mlcape: f64,
    lcl_height_m: f64,
    srh_0_1km: f64,
    bulk_shear_0_6km_ms: f64,
) -> f64 {
    let cape_term = mlcape / 1500.0;
    let lcl_term = if lcl_height_m <= 1000.0 {
        1.0
    } else {
        ((2000.0 - lcl_height_m) / 1000.0).clamp(0.0, 1.0)
    };
    let srh_term = srh_0_1km / 150.0;
    let shear_term = if bulk_shear_0_6km_ms < 12.5 {
        0.0
    } else {
        (bulk_shear_0_6km_ms.min(30.0) / 20.0).max(0.0)
    };
    cape_term * lcl_term * srh_term * shear_term
}

pub fn supercell_composite_parameter(mucape: f64, srh_eff: f64, bulk_shear_eff_ms: f64) -> f64 {
    let cape_term = mucape / 1000.0;
    let srh_term = srh_eff / 50.0;
    let shear_term = if bulk_shear_eff_ms < 10.0 {
        0.0
    } else {
        (bulk_shear_eff_ms.min(20.0) / 20.0).max(0.0)
    };
    cape_term * srh_term * shear_term
}

pub fn critical_angle(
    storm_u: f64,
    storm_v: f64,
    u_sfc: f64,
    v_sfc: f64,
    u_500m: f64,
    v_500m: f64,
) -> f64 {
    let inflow_u = u_sfc - storm_u;
    let inflow_v = v_sfc - storm_v;
    let shear_u = u_500m - u_sfc;
    let shear_v = v_500m - v_sfc;

    let mag_inflow = (inflow_u * inflow_u + inflow_v * inflow_v).sqrt();
    let mag_shear = (shear_u * shear_u + shear_v * shear_v).sqrt();
    if mag_inflow < 1.0e-10 || mag_shear < 1.0e-10 {
        return 0.0;
    }

    let cos_angle = (inflow_u * shear_u + inflow_v * shear_v) / (mag_inflow * mag_shear);
    cos_angle.clamp(-1.0, 1.0).acos() * (180.0 / PI)
}

pub fn boyden_index(z1000: f64, z700: f64, t700: f64) -> f64 {
    (z700 - z1000) / 10.0 - t700 - 200.0
}

pub fn bulk_richardson_number(cape: f64, shear_06_ms: f64) -> f64 {
    let denom = 0.5 * shear_06_ms * shear_06_ms;
    if denom < 0.1 {
        f64::NAN
    } else {
        cape / denom
    }
}

pub fn convective_inhibition_depth(p: &[f64], t: &[f64], td: &[f64]) -> f64 {
    if p.is_empty() {
        return 0.0;
    }

    let (p_lcl, t_lcl) = drylift(p[0], t[0], td[0]);
    let theta_c = (t_lcl + ZEROCNK) * (1000.0 / p_lcl).powf(ROCP) - ZEROCNK;
    let thetam = theta_c - wobf(theta_c) + wobf(t_lcl);

    for i in 0..p.len().min(t.len()).min(td.len()) {
        if p[i] > p_lcl {
            continue;
        }
        let t_parcel = satlift(p[i], thetam);
        let tv_env = virtual_temperature(t[i], p[i], td[i]);
        let tv_parcel = virtual_temperature(t_parcel, p[i], t_parcel);
        if tv_parcel > tv_env {
            return p[0] - p[i];
        }
    }

    p[0] - p[p.len() - 1]
}

pub fn dendritic_growth_zone(t_profile: &[f64], p_profile: &[f64]) -> (f64, f64) {
    let n = t_profile.len().min(p_profile.len());
    if n < 2 {
        return (f64::NAN, f64::NAN);
    }

    let mut p_top = f64::NAN;
    let mut p_bottom = f64::NAN;
    for i in 0..n {
        let t = t_profile[i];
        if (-18.0..=-12.0).contains(&t) {
            if p_bottom.is_nan() {
                p_bottom = p_profile[i];
            }
            p_top = p_profile[i];
        }
    }

    if p_bottom.is_nan() {
        return (f64::NAN, f64::NAN);
    }

    for i in 0..n - 1 {
        if (t_profile[i] > -12.0 && t_profile[i + 1] <= -12.0)
            || (t_profile[i] <= -12.0 && t_profile[i + 1] > -12.0)
        {
            let frac = (-12.0 - t_profile[i]) / (t_profile[i + 1] - t_profile[i]);
            p_bottom = p_profile[i] + frac * (p_profile[i + 1] - p_profile[i]);
            break;
        }
    }
    for i in 0..n - 1 {
        if (t_profile[i] > -18.0 && t_profile[i + 1] <= -18.0)
            || (t_profile[i] <= -18.0 && t_profile[i + 1] > -18.0)
        {
            let frac = (-18.0 - t_profile[i]) / (t_profile[i + 1] - t_profile[i]);
            p_top = p_profile[i] + frac * (p_profile[i + 1] - p_profile[i]);
            break;
        }
    }

    (p_top, p_bottom)
}

pub fn fosberg_fire_weather_index(t_f: f64, rh: f64, wspd_mph: f64) -> f64 {
    let rh = rh.clamp(0.0, 100.0);
    let emc = if rh <= 10.0 {
        0.03229 + 0.281073 * rh - 0.000578 * rh * t_f
    } else if rh <= 50.0 {
        2.22749 + 0.160107 * rh - 0.01478 * t_f
    } else {
        21.0606 + 0.005565 * rh * rh - 0.00035 * rh * t_f - 0.483199 * rh
    } / 30.0;

    let m = emc.max(0.0);
    let eta = 1.0 - 2.0 * m + 1.5 * m * m - 0.5 * m * m * m;
    (eta * (1.0 + wspd_mph * wspd_mph).sqrt() * 10.0 / 3.0).clamp(0.0, 100.0)
}

pub fn freezing_rain_composite(t_profile: &[f64], p_profile: &[f64], precip_type: u8) -> f64 {
    let n = t_profile.len().min(p_profile.len());
    if n < 3 || t_profile[0] > 0.0 {
        return 0.0;
    }

    let mut warm_depth = 0.0;
    let mut warm_intensity = 0.0;
    let mut in_warm_layer = false;
    for i in 1..n {
        if t_profile[i] > 0.0 {
            in_warm_layer = true;
            let dp = (p_profile[i - 1] - p_profile[i]).abs();
            warm_depth += dp;
            warm_intensity += t_profile[i] * dp;
        } else if in_warm_layer {
            break;
        }
    }

    if warm_depth < 1.0 {
        return 0.0;
    }

    let depth_factor = (warm_depth / 100.0).clamp(0.0, 1.0);
    let intensity_factor = (warm_intensity / (warm_depth * 3.0)).clamp(0.0, 1.0);
    let precip_boost = if precip_type == 4 { 1.0 } else { 0.5 };
    (depth_factor * intensity_factor * precip_boost).clamp(0.0, 1.0)
}

pub fn haines_index(t_950: f64, t_850: f64, td_850: f64) -> u8 {
    let a = if t_950 - t_850 <= 3.0 {
        1
    } else if t_950 - t_850 <= 7.0 {
        2
    } else {
        3
    };
    let b = if t_850 - td_850 <= 5.0 {
        1
    } else if t_850 - td_850 <= 9.0 {
        2
    } else {
        3
    };
    a + b
}

pub fn hot_dry_windy(t_c: f64, rh: f64, wspd_ms: f64, vpd: f64) -> f64 {
    let vpd_val = if vpd > 0.0 {
        vpd
    } else {
        let es = saturation_vapor_pressure(t_c);
        let ea = es * (rh / 100.0);
        (es - ea).max(0.0)
    };
    vpd_val * wspd_ms
}

pub fn warm_nose_check(t_profile: &[f64], p_profile: &[f64]) -> bool {
    let n = t_profile.len().min(p_profile.len());
    if n < 3 {
        return false;
    }

    let mut found_below_zero = false;
    for i in 0..n {
        if t_profile[i] <= 0.0 {
            found_below_zero = true;
        }
        if found_below_zero && t_profile[i] > 0.0 {
            return true;
        }
    }
    false
}

pub fn galvez_davison_index(
    t950: f64,
    t850: f64,
    t700: f64,
    t500: f64,
    td950: f64,
    td850: f64,
    td700: f64,
    sst: f64,
) -> f64 {
    let thetae_950 = equivalent_potential_temperature(950.0, t950, td950);
    let thetae_850 = equivalent_potential_temperature(850.0, t850, td850);
    let thetae_700 = equivalent_potential_temperature(700.0, t700, td700);
    let cbi = (thetae_950 + thetae_850) / 2.0 - thetae_700;
    let mwi = ((t500 + ZEROCNK) - 243.15) * 1.5;
    let inflow_index = (sst - 25.0).max(0.0) * 5.0;
    cbi + inflow_index - mwi
}

fn compute_srh_column(heights: &[f64], u_prof: &[f64], v_prof: &[f64], top_m: f64) -> f64 {
    let nz = heights.len().min(u_prof.len()).min(v_prof.len());
    if nz < 2 {
        return 0.0;
    }

    let mean_depth = 6000.0;
    let mut sum_u = 0.0;
    let mut sum_v = 0.0;
    let mut sum_dz = 0.0;
    for k in 0..nz - 1 {
        if heights[k] >= mean_depth {
            break;
        }
        let h_bot = heights[k];
        let h_top = heights[k + 1].min(mean_depth);
        let dz = h_top - h_bot;
        if dz <= 0.0 {
            continue;
        }
        let u_mid = 0.5 * (u_prof[k] + u_prof[k + 1]);
        let v_mid = 0.5 * (v_prof[k] + v_prof[k + 1]);
        sum_u += u_mid * dz;
        sum_v += v_mid * dz;
        sum_dz += dz;
    }
    if sum_dz <= 0.0 {
        return 0.0;
    }

    let mean_u = sum_u / sum_dz;
    let mean_v = sum_v / sum_dz;
    let u_sfc = u_prof[0];
    let v_sfc = v_prof[0];
    let u_6km = interpolate_monotonic_profile(mean_depth, heights, u_prof, false);
    let v_6km = interpolate_monotonic_profile(mean_depth, heights, v_prof, false);
    let shear_u = u_6km - u_sfc;
    let shear_v = v_6km - v_sfc;
    let shear_mag = (shear_u * shear_u + shear_v * shear_v).sqrt();

    let (dev_u, dev_v) = if shear_mag > 0.1 {
        let scale = 7.5 / shear_mag;
        (shear_v * scale, -shear_u * scale)
    } else {
        (0.0, 0.0)
    };
    let storm_u = mean_u + dev_u;
    let storm_v = mean_v + dev_v;

    let mut srh = 0.0;
    for k in 0..nz - 1 {
        if heights[k] >= top_m {
            break;
        }
        let h_bot = heights[k];
        let h_top = heights[k + 1].min(top_m);
        if h_top <= h_bot {
            continue;
        }
        let u_bot = u_prof[k];
        let v_bot = v_prof[k];
        let (u_top, v_top) = if h_top < heights[k + 1] {
            let frac = (h_top - heights[k]) / (heights[k + 1] - heights[k]);
            (
                u_prof[k] + frac * (u_prof[k + 1] - u_prof[k]),
                v_prof[k] + frac * (v_prof[k + 1] - v_prof[k]),
            )
        } else {
            (u_prof[k + 1], v_prof[k + 1])
        };
        let sr_u_bot = u_bot - storm_u;
        let sr_v_bot = v_bot - storm_v;
        let sr_u_top = u_top - storm_u;
        let sr_v_top = v_top - storm_v;
        srh += sr_u_top * sr_v_bot - sr_u_bot * sr_v_top;
    }
    srh
}

pub fn compute_cape_cin(
    pressure_3d: &[f64],
    temperature_c_3d: &[f64],
    qvapor_3d: &[f64],
    height_agl_3d: &[f64],
    psfc: &[f64],
    t2: &[f64],
    q2: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    parcel_type: &str,
    top_m: Option<f64>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n2d = nx * ny;
    let mut cape_2d = Vec::with_capacity(n2d);
    let mut cin_2d = Vec::with_capacity(n2d);
    let mut lcl_2d = Vec::with_capacity(n2d);
    let mut lfc_2d = Vec::with_capacity(n2d);

    for idx in 0..n2d {
        let y = idx / nx;
        let x = idx % nx;

        let mut p_hpa: Vec<f64> = extract_column(pressure_3d, nx, ny, nz, y, x)
            .into_iter()
            .map(|p| p / 100.0)
            .collect();
        let mut t_c = extract_column(temperature_c_3d, nx, ny, nz, y, x);
        let q_col = extract_column(qvapor_3d, nx, ny, nz, y, x);
        let mut h_agl = extract_column(height_agl_3d, nx, ny, nz, y, x);
        let mut td_c = Vec::with_capacity(nz);
        for k in 0..nz {
            td_c.push(dewpoint_from_q(q_col[k], p_hpa[k]));
        }

        if p_hpa.len() > 1 && p_hpa[0] < p_hpa[p_hpa.len() - 1] {
            p_hpa.reverse();
            t_c.reverse();
            td_c.reverse();
            h_agl.reverse();
        }

        let psfc_hpa = psfc[idx] / 100.0;
        let t2m_c = t2[idx] - ZEROCNK;
        let td2m_c = dewpoint_from_q(q2[idx], psfc_hpa);
        let (cape_val, cin_val, lcl_val, lfc_val) = cape_cin(
            &p_hpa,
            &t_c,
            &td_c,
            &h_agl,
            psfc_hpa,
            t2m_c,
            td2m_c,
            normalize_parcel_type(parcel_type),
            100.0,
            300.0,
            top_m,
        );
        cape_2d.push(cape_val);
        cin_2d.push(cin_val);
        lcl_2d.push(lcl_val);
        lfc_2d.push(lfc_val);
    }

    (cape_2d, cin_2d, lcl_2d, lfc_2d)
}

pub fn compute_srh(
    u_3d: &[f64],
    v_3d: &[f64],
    height_agl_3d: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    top_m: f64,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(nx * ny);
    for idx in 0..nx * ny {
        let y = idx / nx;
        let x = idx % nx;
        let mut u_prof = extract_column(u_3d, nx, ny, nz, y, x);
        let mut v_prof = extract_column(v_3d, nx, ny, nz, y, x);
        let mut h_prof = extract_column(height_agl_3d, nx, ny, nz, y, x);
        if h_prof.len() > 1 && h_prof[0] > h_prof[h_prof.len() - 1] {
            h_prof.reverse();
            u_prof.reverse();
            v_prof.reverse();
        }
        out.push(compute_srh_column(&h_prof, &u_prof, &v_prof, top_m));
    }
    out
}

pub fn compute_shear(
    u_3d: &[f64],
    v_3d: &[f64],
    height_agl_3d: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    bottom_m: f64,
    top_m: f64,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(nx * ny);
    for idx in 0..nx * ny {
        let y = idx / nx;
        let x = idx % nx;
        let mut u_prof = extract_column(u_3d, nx, ny, nz, y, x);
        let mut v_prof = extract_column(v_3d, nx, ny, nz, y, x);
        let mut h_prof = extract_column(height_agl_3d, nx, ny, nz, y, x);
        if h_prof.len() > 1 && h_prof[0] > h_prof[h_prof.len() - 1] {
            h_prof.reverse();
            u_prof.reverse();
            v_prof.reverse();
        }
        let u_bot = interpolate_monotonic_profile(bottom_m, &h_prof, &u_prof, false);
        let v_bot = interpolate_monotonic_profile(bottom_m, &h_prof, &v_prof, false);
        let u_top = interpolate_monotonic_profile(top_m, &h_prof, &u_prof, false);
        let v_top = interpolate_monotonic_profile(top_m, &h_prof, &v_prof, false);
        let du = u_top - u_bot;
        let dv = v_top - v_bot;
        out.push((du * du + dv * dv).sqrt());
    }
    out
}

pub fn compute_lapse_rate(
    temperature_c_3d: &[f64],
    qvapor_3d: &[f64],
    height_agl_3d: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    bottom_km: f64,
    top_km: f64,
) -> Vec<f64> {
    let bottom_m = bottom_km * 1000.0;
    let top_m = top_km * 1000.0;
    let mut out = Vec::with_capacity(nx * ny);
    for idx in 0..nx * ny {
        let y = idx / nx;
        let x = idx % nx;
        let mut t_prof = extract_column(temperature_c_3d, nx, ny, nz, y, x);
        let mut q_prof = extract_column(qvapor_3d, nx, ny, nz, y, x);
        let mut h_prof = extract_column(height_agl_3d, nx, ny, nz, y, x);
        if h_prof.len() > 1 && h_prof[0] > h_prof[h_prof.len() - 1] {
            h_prof.reverse();
            t_prof.reverse();
            q_prof.reverse();
        }
        let tv_prof: Vec<f64> = t_prof
            .iter()
            .zip(q_prof.iter())
            .map(|(&t, &q)| (t + ZEROCNK) * (1.0 + 0.61 * q.max(0.0)) - ZEROCNK)
            .collect();
        let tv_bot = interpolate_monotonic_profile(bottom_m, &h_prof, &tv_prof, false);
        let tv_top = interpolate_monotonic_profile(top_m, &h_prof, &tv_prof, false);
        let dz_km = top_km - bottom_km;
        out.push(if dz_km.abs() < 1.0e-3 {
            0.0
        } else {
            (tv_bot - tv_top) / dz_km
        });
    }
    out
}

pub fn compute_pw(
    qvapor_3d: &[f64],
    pressure_3d: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(nx * ny);
    for idx in 0..nx * ny {
        let y = idx / nx;
        let x = idx % nx;
        let mut q_prof = extract_column(qvapor_3d, nx, ny, nz, y, x);
        let mut p_prof = extract_column(pressure_3d, nx, ny, nz, y, x);
        if p_prof.len() > 1 && p_prof[0] < p_prof[p_prof.len() - 1] {
            p_prof.reverse();
            q_prof.reverse();
        }
        let mut pw_val = 0.0;
        for k in 0..p_prof.len().saturating_sub(1) {
            let dp = (p_prof[k] - p_prof[k + 1]).abs();
            let q_avg = 0.5 * (q_prof[k].max(0.0) + q_prof[k + 1].max(0.0));
            pw_val += q_avg * dp;
        }
        out.push(pw_val / G);
    }
    out
}

pub fn compute_stp(cape: &[f64], lcl: &[f64], srh_1km: &[f64], shear_6km: &[f64]) -> Vec<f64> {
    let n = cape
        .len()
        .min(lcl.len())
        .min(srh_1km.len())
        .min(shear_6km.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let cape_term = (cape[i] / 1500.0).max(0.0);
        let lcl_term = ((2000.0 - lcl[i]) / 1000.0).clamp(0.0, 2.0);
        let srh_term = (srh_1km[i] / 150.0).max(0.0);
        let shear_term = (shear_6km[i] / 20.0).min(1.5).max(0.0);
        out.push(cape_term * lcl_term * srh_term * shear_term);
    }
    out
}

pub fn compute_scp(mucape: &[f64], srh_3km: &[f64], shear_6km: &[f64]) -> Vec<f64> {
    let n = mucape.len().min(srh_3km.len()).min(shear_6km.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let cape_term = (mucape[i] / 1000.0).max(0.0);
        let srh_term = (srh_3km[i] / 50.0).max(0.0);
        let shear_term = (shear_6km[i] / 40.0).max(0.0);
        out.push(cape_term * srh_term * shear_term);
    }
    out
}

pub fn compute_ehi(cape: &[f64], srh: &[f64]) -> Vec<f64> {
    let n = cape.len().min(srh.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push((cape[i] * srh[i]) / 160000.0);
    }
    out
}

pub fn compute_ship(
    cape: &[f64],
    shear06: &[f64],
    t500: &[f64],
    lr_700_500: &[f64],
    mr: &[f64],
    nx: usize,
    ny: usize,
) -> Vec<f64> {
    let n = (nx * ny)
        .min(cape.len())
        .min(shear06.len())
        .min(t500.len())
        .min(lr_700_500.len())
        .min(mr.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mucape = cape[i].max(0.0);
        let mr_val = mr[i].max(0.0);
        let lr = lr_700_500[i].max(0.0);
        let t5 = (-t500[i]).max(0.0);
        let s06 = shear06[i].max(0.0);
        let ship = (mucape * mr_val * lr * t5 * s06) / 42_000_000.0;
        out.push(if mucape < 1300.0 {
            ship * (mucape / 1300.0)
        } else {
            ship
        });
    }
    out
}

pub fn compute_dcp(
    dcape: &[f64],
    mu_cape: &[f64],
    shear06: &[f64],
    mu_mixing_ratio: &[f64],
    nx: usize,
    ny: usize,
) -> Vec<f64> {
    let n = (nx * ny)
        .min(dcape.len())
        .min(mu_cape.len())
        .min(shear06.len())
        .min(mu_mixing_ratio.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let dcape_term = (dcape[i] / 980.0).max(0.0);
        let cape_term = (mu_cape[i] / 2000.0).max(0.0);
        let shear_term = (shear06[i] / 20.0).max(0.0);
        let mr_term = (mu_mixing_ratio[i] / 11.0).max(0.0);
        out.push(dcape_term * cape_term * shear_term * mr_term);
    }
    out
}

pub fn compute_grid_scp(
    mu_cape: &[f64],
    srh: &[f64],
    shear_06: &[f64],
    mu_cin: &[f64],
    nx: usize,
    ny: usize,
) -> Vec<f64> {
    let n = (nx * ny)
        .min(mu_cape.len())
        .min(srh.len())
        .min(shear_06.len())
        .min(mu_cin.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let cape_term = (mu_cape[i] / 1000.0).max(0.0);
        let srh_term = (srh[i] / 50.0).max(0.0);
        let shear_term = (shear_06[i] / 40.0).max(0.0);
        let cin_term = if mu_cin[i] > -40.0 {
            1.0
        } else {
            -40.0 / mu_cin[i].min(-0.01)
        };
        out.push(cape_term * srh_term * shear_term * cin_term);
    }
    out
}

pub fn compute_grid_critical_angle(
    u_storm: &[f64],
    v_storm: &[f64],
    u_shear: &[f64],
    v_shear: &[f64],
    nx: usize,
    ny: usize,
) -> Vec<f64> {
    let n = (nx * ny)
        .min(u_storm.len())
        .min(v_storm.len())
        .min(u_shear.len())
        .min(v_shear.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let inflow_u = -u_storm[i];
        let inflow_v = -v_storm[i];
        let shear_u = u_shear[i];
        let shear_v = v_shear[i];
        let mag_inflow = (inflow_u * inflow_u + inflow_v * inflow_v).sqrt();
        let mag_shear = (shear_u * shear_u + shear_v * shear_v).sqrt();
        out.push(if mag_inflow < 0.01 || mag_shear < 0.01 {
            f64::NAN
        } else {
            let cos_angle =
                ((inflow_u * shear_u) + (inflow_v * shear_v)) / (mag_inflow * mag_shear);
            cos_angle.clamp(-1.0, 1.0).acos().to_degrees()
        });
    }
    out
}

pub fn composite_reflectivity(refl_3d: &[f64], nx: usize, ny: usize, nz: usize) -> Vec<f64> {
    let n2d = nx * ny;
    let mut out = Vec::with_capacity(n2d);
    for idx in 0..n2d {
        let mut max_dbz = -999.0_f64;
        for k in 0..nz {
            max_dbz = max_dbz.max(refl_3d[k * n2d + idx]);
        }
        out.push(max_dbz.max(-30.0));
    }
    out
}

pub fn composite_reflectivity_from_hydrometeors(
    pressure_3d: &[f64],
    temperature_c_3d: &[f64],
    qrain_3d: &[f64],
    qsnow_3d: &[f64],
    qgraup_3d: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Vec<f64> {
    let n2d = nx * ny;
    let mut out = Vec::with_capacity(n2d);
    for idx in 0..n2d {
        let mut max_dbz = -999.0_f64;
        for k in 0..nz {
            let flat = k * n2d + idx;
            let rho = pressure_3d[flat] / (287.058 * (temperature_c_3d[flat] + ZEROCNK));
            let qr = qrain_3d[flat].max(0.0);
            let qs = qsnow_3d[flat].max(0.0);
            let qg = qgraup_3d[flat].max(0.0);
            let z_rain = 3.63e9 * (rho * qr).powf(1.75);
            let z_snow = 9.80e8 * (rho * qs).powf(1.75);
            let z_graupel = 4.33e9 * (rho * qg).powf(1.75);
            let z_total = z_rain + z_snow + z_graupel;
            let dbz = if z_total > 0.0 {
                10.0 * z_total.log10()
            } else {
                -999.0
            };
            max_dbz = max_dbz.max(dbz);
        }
        out.push(max_dbz.max(-30.0));
    }
    out
}

pub fn parcel_profile_with_lcl_as_dataset(
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    dewpoint_c: &[f64],
) -> Result<CalcDataset, InvalidSoundingError> {
    if pressure_hpa.is_empty() || temperature_c.is_empty() || dewpoint_c.is_empty() {
        return Err(InvalidSoundingError::new(
            "pressure, temperature, and dewpoint must be non-empty",
        ));
    }
    let n = pressure_hpa
        .len()
        .min(temperature_c.len())
        .min(dewpoint_c.len());
    let (p_aug, parcel_t) =
        parcel_profile_with_lcl(&pressure_hpa[..n], temperature_c[0], dewpoint_c[0]);
    let ambient_t: Vec<f64> = p_aug
        .iter()
        .map(|&p| interpolate_monotonic_profile(p, &pressure_hpa[..n], &temperature_c[..n], true))
        .collect();
    let ambient_td: Vec<f64> = p_aug
        .iter()
        .map(|&p| interpolate_monotonic_profile(p, &pressure_hpa[..n], &dewpoint_c[..n], true))
        .collect();

    Ok(CalcDataset {
        dims: vec![("isobaric".to_string(), p_aug.len())],
        coords: vec![CalcCoordinate {
            name: "isobaric".to_string(),
            values: p_aug.clone(),
            units: "hPa".to_string(),
        }],
        variables: vec![
            CalcVariable {
                name: "ambient_temperature".to_string(),
                dims: vec!["isobaric".to_string()],
                values: ambient_t,
                units: "degC".to_string(),
            },
            CalcVariable {
                name: "ambient_dew_point".to_string(),
                dims: vec!["isobaric".to_string()],
                values: ambient_td,
                units: "degC".to_string(),
            },
            CalcVariable {
                name: "parcel_temperature".to_string(),
                dims: vec!["isobaric".to_string()],
                values: parcel_t,
                units: "degC".to_string(),
            },
        ],
    })
}

pub fn isentropic_interpolation_as_dataset(
    theta_levels_k: &[f64],
    pressure_3d_hpa: &[f64],
    temperature_3d_k: &[f64],
    fields: &[(&str, &[f64], &str)],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Result<CalcDataset, InvalidSoundingError> {
    if theta_levels_k.is_empty() {
        return Err(InvalidSoundingError::new(
            "theta_levels_k must be non-empty",
        ));
    }
    let field_slices: Vec<&[f64]> = fields.iter().map(|(_, values, _)| *values).collect();
    let interpolated = super::isentropic_interpolation(
        theta_levels_k,
        pressure_3d_hpa,
        temperature_3d_k,
        &field_slices,
        nx,
        ny,
        nz,
    );
    let mut variables = vec![
        CalcVariable {
            name: "pressure".to_string(),
            dims: vec![
                "isentropic_level".to_string(),
                "y".to_string(),
                "x".to_string(),
            ],
            values: interpolated[0].clone(),
            units: "hPa".to_string(),
        },
        CalcVariable {
            name: "temperature".to_string(),
            dims: vec![
                "isentropic_level".to_string(),
                "y".to_string(),
                "x".to_string(),
            ],
            values: interpolated[1].clone(),
            units: "K".to_string(),
        },
    ];
    for (i, (name, _, units)) in fields.iter().enumerate() {
        variables.push(CalcVariable {
            name: (*name).to_string(),
            dims: vec![
                "isentropic_level".to_string(),
                "y".to_string(),
                "x".to_string(),
            ],
            values: interpolated[2 + i].clone(),
            units: (*units).to_string(),
        });
    }
    Ok(CalcDataset {
        dims: vec![
            ("isentropic_level".to_string(), theta_levels_k.len()),
            ("y".to_string(), ny),
            ("x".to_string(), nx),
        ],
        coords: vec![CalcCoordinate {
            name: "isentropic_level".to_string(),
            values: theta_levels_k.to_vec(),
            units: "K".to_string(),
        }],
        variables,
    })
}

pub fn zoom_xarray(input: &Grid2D, zoom_y: f64, zoom_x: Option<f64>) -> Grid2D {
    let zoom_x = zoom_x.unwrap_or(zoom_y);
    let out_ny = ((input.ny as f64) * zoom_y).round().max(1.0) as usize;
    let out_nx = ((input.nx as f64) * zoom_x).round().max(1.0) as usize;
    let mut out = Grid2D::zeros(out_nx, out_ny);
    for y in 0..out_ny {
        for x in 0..out_nx {
            let src_x = if out_nx <= 1 {
                0.0
            } else {
                x as f64 * (input.nx.saturating_sub(1)) as f64 / (out_nx.saturating_sub(1)) as f64
            };
            let src_y = if out_ny <= 1 {
                0.0
            } else {
                y as f64 * (input.ny.saturating_sub(1)) as f64 / (out_ny.saturating_sub(1)) as f64
            };
            let x0 = src_x.floor() as usize;
            let y0 = src_y.floor() as usize;
            let x1 = (x0 + 1).min(input.nx - 1);
            let y1 = (y0 + 1).min(input.ny - 1);
            let dx = src_x - x0 as f64;
            let dy = src_y - y0 as f64;
            let v00 = input.get(x0, y0);
            let v10 = input.get(x1, y0);
            let v01 = input.get(x0, y1);
            let v11 = input.get(x1, y1);
            let top = v00 * (1.0 - dx) + v10 * dx;
            let bottom = v01 * (1.0 - dx) + v11 * dx;
            out.set(x, y, top * (1.0 - dy) + bottom * dy);
        }
    }
    out
}

pub fn inverse_distance_to_grid(
    obs_x: &[f64],
    obs_y: &[f64],
    obs_values: &[f64],
    target: &GridSpec,
    radius: f64,
    gamma: Option<f64>,
    kappa: Option<f64>,
    min_neighbors: usize,
    kind: &str,
) -> Vec<f64> {
    let (grid_y, grid_x) = target_grid_points(target);
    let kind_int = match kind {
        "barnes" => 1,
        "cressman" => 2,
        _ => 0,
    };
    super::inverse_distance_to_points(
        obs_x,
        obs_y,
        obs_values,
        &grid_x,
        &grid_y,
        radius,
        min_neighbors,
        kind_int,
        kappa.unwrap_or(100_000.0),
        gamma.unwrap_or(0.2),
    )
}

pub fn natural_neighbor_to_grid(
    source_lats: &[f64],
    source_lons: &[f64],
    source_values: &[f64],
    target: &GridSpec,
) -> Vec<f64> {
    let (target_lats, target_lons) = target_grid_points(target);
    super::natural_neighbor_to_points(
        source_lats,
        source_lons,
        source_values,
        &target_lats,
        &target_lons,
    )
}

pub fn interpolate_to_grid(
    values: &[f64],
    src_lats: &[f64],
    src_lons: &[f64],
    target: &GridSpec,
    method: InterpMethod,
) -> Vec<f64> {
    let (src_nx, src_ny) = infer_source_dims(src_lats, src_lons, values.len());
    let (target_lats, target_lons) = target_grid_points(target);
    let mut out = Vec::with_capacity(target.nx * target.ny);
    for i in 0..target_lats.len() {
        let val = match method {
            InterpMethod::NearestNeighbor => nearest_regular(
                values,
                src_lats,
                src_lons,
                src_nx,
                src_ny,
                target_lats[i],
                target_lons[i],
            ),
            InterpMethod::Bilinear | InterpMethod::Bicubic | InterpMethod::Budget => {
                bilinear_regular(
                    values,
                    src_lats,
                    src_lons,
                    src_nx,
                    src_ny,
                    target_lats[i],
                    target_lons[i],
                )
            }
        };
        out.push(val);
    }
    out
}

pub fn cross_section(
    values: &[f64],
    lats: &[f64],
    lons: &[f64],
    nx: usize,
    ny: usize,
    start: (f64, f64),
    end: (f64, f64),
    steps: usize,
    interp_type: &str,
) -> (Vec<f64>, Vec<f64>) {
    let (slice_lats, slice_lons) = geodesic(start, end, steps.max(2));
    let values = interpolate_to_points(lats, lons, values, &slice_lats, &slice_lons, interp_type);
    let mut distances = Vec::with_capacity(slice_lats.len());
    let total = super::geodesic(start, end, 2);
    let _ = total;
    for i in 0..slice_lats.len() {
        let f = if slice_lats.len() > 1 {
            i as f64 / (slice_lats.len() - 1) as f64
        } else {
            0.0
        };
        let dist = f * haversine_km(start.0, start.1, end.0, end.1);
        distances.push(dist);
    }
    let _ = (nx, ny);
    (values, distances)
}

fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    2.0 * 6371.0 * a.sqrt().asin()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_scalar_severe_suite() {
        let stp = significant_tornado_parameter(1500.0, 1000.0, 150.0, 20.0);
        assert!((stp - 1.0).abs() < 1.0e-10);
        let scp = supercell_composite_parameter(1000.0, 50.0, 20.0);
        assert!((scp - 1.0).abs() < 1.0e-10);
        let angle = critical_angle(10.0, 0.0, 20.0, 0.0, 20.0, 10.0);
        assert!(angle.is_finite() && angle > 0.0);
        assert_eq!(haines_index(22.0, 14.0, 6.0), 5);
        assert!(hot_dry_windy(35.0, 15.0, 10.0, 0.0) > 0.0);
    }

    #[test]
    fn computes_grid_and_dataset_wrappers() {
        let pressure = vec![100000.0, 90000.0];
        let temp = vec![25.0, 15.0];
        let q = vec![0.012, 0.006];
        let height = vec![0.0, 1000.0];
        let (cape, cin, lcl, lfc) = compute_cape_cin(
            &pressure,
            &temp,
            &q,
            &height,
            &[100000.0],
            &[298.15],
            &[0.012],
            1,
            1,
            2,
            "surface",
            None,
        );
        assert_eq!(cape.len(), 1);
        assert_eq!(cin.len(), 1);
        assert_eq!(lcl.len(), 1);
        assert_eq!(lfc.len(), 1);

        let dataset = parcel_profile_with_lcl_as_dataset(
            &[1000.0, 900.0, 800.0],
            &[25.0, 18.0, 10.0],
            &[20.0, 12.0, 2.0],
        )
        .unwrap();
        assert_eq!(dataset.coords[0].name, "isobaric");
        assert_eq!(dataset.variables.len(), 3);
    }

    #[test]
    fn computes_regrid_helpers() {
        let target = GridSpec::regular(30.0, 31.0, -100.0, -99.0, 1.0);
        let out = interpolate_to_grid(
            &[1.0, 2.0, 3.0, 4.0],
            &[30.0, 31.0],
            &[-100.0, -99.0],
            &target,
            InterpMethod::Bilinear,
        );
        assert_eq!(out.len(), 4);

        let nn = natural_neighbor_to_grid(
            &[30.0, 31.0, 30.0, 31.0],
            &[-100.0, -100.0, -99.0, -99.0],
            &[1.0, 2.0, 3.0, 4.0],
            &target,
        );
        assert_eq!(nn.len(), 4);

        let zoomed = zoom_xarray(&Grid2D::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]), 2.0, None);
        assert_eq!(zoomed.nx, 4);
        assert_eq!(zoomed.ny, 4);
    }
}
