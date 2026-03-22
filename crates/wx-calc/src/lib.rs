//! Canonical meteorological calculation layer.

mod inventory;
mod parity_tail;

pub use inventory::{
    calc_port_inventory, calc_port_summary, missing_names, CalcCategory, CalcCategorySummary,
    CalcPortEntry, CalcPortSummary, PortStatus,
};
pub use parity_tail::*;

use wx_types::Grid2D;

const RD: f64 = 287.047_490_977_184_57;
const RV_METPY: f64 = 461.523_115_726_060_84;
const CP_D: f64 = 1004.666_218_420_146_2;
const CP_V: f64 = 1875.0;
const G: f64 = 9.80665;
const ROCP: f64 = 0.285_714_285_714_285_7;
const ZEROCNK: f64 = 273.15;
const EPSILON: f64 = 0.621_956_910_057_703_3;
const LV: f64 = 2_500_840.0;
const OMEGA: f64 = 7.292_115_9e-5;

const P0_STD: f64 = 1013.25;
const T0_STD: f64 = 288.0;
const LAPSE_STD: f64 = 0.0065;
const M_AIR: f64 = 0.028_964_4;
const R_STAR: f64 = 8.31447;
const BARO_EXP: f64 = G * M_AIR / (R_STAR * LAPSE_STD);

const SVP_T0: f64 = 273.16;
const SAT_PRESSURE_0C: f64 = 611.2;
const CP_L: f64 = 4219.4;
const CP_V_METPY: f64 = 1860.078_011_865_639;
const LV_0: f64 = 2_500_840.0;

const DIRECTIONS_16: [&str; 16] = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW",
    "NNW",
];

fn moist_air_gas_constant_specific_humidity(specific_humidity_kgkg: f64) -> f64 {
    RD + specific_humidity_kgkg * (RV_METPY - RD)
}

fn moist_air_specific_heat_pressure_specific_humidity(specific_humidity_kgkg: f64) -> f64 {
    CP_D + specific_humidity_kgkg * (CP_V_METPY - CP_D)
}

fn lambert_w_minus_one(z: f64) -> f64 {
    let min_z = -1.0 / std::f64::consts::E;
    if z == min_z {
        return -1.0;
    }
    assert!(
        (min_z..0.0).contains(&z),
        "lambert_w_minus_one requires z in [-1/e, 0)"
    );

    let mut w = if z < -0.333 {
        let p = (2.0 * (std::f64::consts::E * z + 1.0)).sqrt();
        -1.0 - p + p * p / 3.0 - 11.0 * p * p * p / 72.0
    } else {
        let l1 = (-z).ln();
        let l2 = (-l1).ln();
        l1 - l2 + l2 / l1
    };

    for _ in 0..50 {
        let ew = w.exp();
        let f = w * ew - z;
        let denom = ew * (w + 1.0) - (w + 2.0) * f / (2.0 * (w + 1.0));
        let step = f / denom;
        let next = w - step;
        if step.abs() < 1.0e-14 * next.abs().max(1.0) {
            return next;
        }
        w = next;
    }

    w
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationTier {
    Planned,
    Benchmarked,
    ReferenceMatched,
    ProductionLocked,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalculationStatus {
    pub name: String,
    pub tier: ValidationTier,
    pub notes: String,
}

#[derive(Debug, Default)]
pub struct CalcEngine;

impl CalcEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn priorities(&self) -> Vec<CalculationStatus> {
        let summary = calc_port_summary();
        summary
            .categories
            .into_iter()
            .map(|category| CalculationStatus {
                name: category.category.as_str().to_string(),
                tier: if category.missing == 0 {
                    ValidationTier::ProductionLocked
                } else if category.ported > 0 {
                    ValidationTier::Benchmarked
                } else {
                    ValidationTier::Planned
                },
                notes: format!(
                    "{} of {} public calc targets ported; {} still pending",
                    category.ported, category.total, category.missing
                ),
            })
            .collect()
    }
}

pub fn potential_temperature(pressure_hpa: f64, temperature_c: f64) -> f64 {
    let temperature_k = temperature_c + ZEROCNK;
    temperature_k * (1000.0 / pressure_hpa).powf(ROCP)
}

pub fn temperature_from_potential_temperature(pressure_hpa: f64, theta_k: f64) -> f64 {
    theta_k * (pressure_hpa / 1000.0).powf(ROCP)
}

pub fn saturation_vapor_pressure(temperature_c: f64) -> f64 {
    svp_liquid_pa(temperature_c + ZEROCNK) / 100.0
}

pub fn vapor_pressure(dewpoint_c: f64) -> f64 {
    saturation_vapor_pressure(dewpoint_c)
}

pub fn dewpoint(vapor_pressure_hpa: f64) -> f64 {
    dewpoint_from_vapor_pressure(vapor_pressure_hpa)
}

pub fn dewpoint_from_vapor_pressure(vapor_pressure_hpa: f64) -> f64 {
    if vapor_pressure_hpa <= 0.0 {
        return -ZEROCNK;
    }
    let ln_ratio = (vapor_pressure_hpa / 6.112).ln();
    243.5 * ln_ratio / (17.67 - ln_ratio)
}

pub fn dewpoint_from_relative_humidity(temperature_c: f64, relative_humidity_pct: f64) -> f64 {
    let vapor_pressure_hpa =
        saturation_vapor_pressure(temperature_c) * (relative_humidity_pct / 100.0);
    dewpoint_from_vapor_pressure(vapor_pressure_hpa)
}

pub fn relative_humidity_from_dewpoint(temperature_c: f64, dewpoint_c: f64) -> f64 {
    let vapor = vapor_pressure(dewpoint_c);
    let saturation = saturation_vapor_pressure(temperature_c);
    100.0 * (vapor / saturation)
}

pub fn mixing_ratio(pressure_hpa: f64, temperature_c: f64) -> f64 {
    let x = 0.02 * (temperature_c - 12.5 + (7500.0 / pressure_hpa));
    let enhancement = 1.0 + (0.0000045 * pressure_hpa) + (0.0014 * x * x);
    let enhanced_svp = enhancement * vappres_sharppy(temperature_c);
    621.97 * (enhanced_svp / (pressure_hpa - enhanced_svp))
}

pub fn saturation_mixing_ratio(pressure_hpa: f64, temperature_c: f64) -> f64 {
    let es = saturation_vapor_pressure(temperature_c);
    (EPSILON * es / (pressure_hpa - es) * 1000.0).max(0.0)
}

pub fn specific_humidity_from_mixing_ratio(mixing_ratio_kgkg: f64) -> f64 {
    mixing_ratio_kgkg / (1.0 + mixing_ratio_kgkg)
}

pub fn specific_humidity(pressure_hpa: f64, mixing_ratio_gkg: f64) -> f64 {
    let _ = pressure_hpa;
    specific_humidity_from_mixing_ratio(mixing_ratio_gkg / 1000.0)
}

pub fn mixing_ratio_from_specific_humidity(q_kgkg: f64) -> f64 {
    (q_kgkg / (1.0 - q_kgkg)) * 1000.0
}

pub fn mixing_ratio_from_relative_humidity(
    pressure_hpa: f64,
    temperature_c: f64,
    relative_humidity_pct: f64,
) -> f64 {
    let ws = saturation_mixing_ratio(pressure_hpa, temperature_c);
    ws * relative_humidity_pct / 100.0
}

pub fn relative_humidity_from_mixing_ratio(
    pressure_hpa: f64,
    temperature_c: f64,
    mixing_ratio_gkg: f64,
) -> f64 {
    let ws = saturation_mixing_ratio(pressure_hpa, temperature_c);
    if ws <= 0.0 {
        return 0.0;
    }
    (mixing_ratio_gkg / ws) * 100.0
}

pub fn relative_humidity_from_specific_humidity(
    pressure_hpa: f64,
    temperature_c: f64,
    specific_humidity_kgkg: f64,
) -> f64 {
    let w_gkg = mixing_ratio_from_specific_humidity(specific_humidity_kgkg);
    relative_humidity_from_mixing_ratio(pressure_hpa, temperature_c, w_gkg)
}

pub fn specific_humidity_from_dewpoint(pressure_hpa: f64, dewpoint_c: f64) -> f64 {
    let vapor_pressure_hpa = saturation_vapor_pressure(dewpoint_c);
    let mixing_ratio_kgkg = EPSILON * vapor_pressure_hpa / (pressure_hpa - vapor_pressure_hpa);
    specific_humidity_from_mixing_ratio(mixing_ratio_kgkg)
}

pub fn dewpoint_from_specific_humidity(pressure_hpa: f64, specific_humidity_kgkg: f64) -> f64 {
    let mixing_ratio_kgkg = specific_humidity_kgkg / (1.0 - specific_humidity_kgkg);
    let vapor_pressure_hpa = mixing_ratio_kgkg * pressure_hpa / (EPSILON + mixing_ratio_kgkg);
    dewpoint_from_vapor_pressure(vapor_pressure_hpa)
}

pub fn equivalent_potential_temperature(
    pressure_hpa: f64,
    temperature_c: f64,
    dewpoint_c: f64,
) -> f64 {
    let temperature_k = temperature_c + ZEROCNK;
    let dewpoint_k = dewpoint_c + ZEROCNK;
    let t_lcl =
        56.0 + 1.0 / (1.0 / (dewpoint_k - 56.0) + (temperature_k / dewpoint_k).ln() / 800.0);
    let vapor_pressure_hpa = saturation_vapor_pressure(dewpoint_c);
    let mixing_ratio_kgkg = EPSILON * vapor_pressure_hpa / (pressure_hpa - vapor_pressure_hpa);
    let theta_dl = temperature_k
        * (1000.0 / (pressure_hpa - vapor_pressure_hpa)).powf(ROCP)
        * (temperature_k / t_lcl).powf(0.28 * mixing_ratio_kgkg);
    theta_dl
        * ((3036.0 / t_lcl - 1.78) * mixing_ratio_kgkg * (1.0 + 0.448 * mixing_ratio_kgkg)).exp()
}

pub fn virtual_temperature(temperature_c: f64, pressure_hpa: f64, dewpoint_c: f64) -> f64 {
    let mixing_ratio_kgkg = saturation_mixing_ratio(pressure_hpa, dewpoint_c) / 1000.0;
    parcel_virtual_temperature_from_mixing_ratio(temperature_c, mixing_ratio_kgkg)
}

pub fn virtual_temperature_from_dewpoint(
    temperature_c: f64,
    dewpoint_c: f64,
    pressure_hpa: f64,
) -> f64 {
    virtual_temperature(temperature_c, pressure_hpa, dewpoint_c)
}

pub fn virtual_potential_temperature(
    pressure_hpa: f64,
    temperature_c: f64,
    mixing_ratio_gkg: f64,
) -> f64 {
    let theta = potential_temperature(pressure_hpa, temperature_c);
    theta * (1.0 + 0.61 * (mixing_ratio_gkg / 1000.0))
}

pub fn wet_bulb_temperature(pressure_hpa: f64, temperature_c: f64, dewpoint_c: f64) -> f64 {
    let (pressure_lcl_hpa, temperature_lcl_c) = drylift(pressure_hpa, temperature_c, dewpoint_c);
    moist_lapse_from_reference(&[pressure_hpa], temperature_lcl_c, pressure_lcl_hpa)[0]
}

pub fn wet_bulb_potential_temperature(
    pressure_hpa: f64,
    temperature_c: f64,
    dewpoint_c: f64,
) -> f64 {
    let theta_e = equivalent_potential_temperature(pressure_hpa, temperature_c, dewpoint_c);
    if theta_e <= 173.15 {
        return theta_e;
    }
    let x = theta_e / 273.15;
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    let a = 7.101574 - 20.68208 * x + 16.11182 * x2 + 2.574631 * x3 - 5.205688 * x4;
    let b = 1.0 - 3.552497 * x + 3.781782 * x2 - 0.6899655 * x3 - 0.5929340 * x4;
    theta_e - (a / b).exp()
}

pub fn lcl(pressure_hpa: f64, temperature_c: f64, dewpoint_c: f64) -> (f64, f64) {
    drylift(pressure_hpa, temperature_c, dewpoint_c)
}

pub fn lfc(
    pressure_profile_hpa: &[f64],
    temperature_profile_c: &[f64],
    dewpoint_profile_c: &[f64],
) -> Option<(f64, f64)> {
    let (pressure_lcl_hpa, _temperature_lcl_c, parcel_virtual_temperature_c) = lift_parcel_profile(
        pressure_profile_hpa,
        temperature_profile_c,
        dewpoint_profile_c,
    )?;

    for i in 1..pressure_profile_hpa.len() {
        if pressure_profile_hpa[i] > pressure_lcl_hpa {
            continue;
        }
        let tv_env_prev = virtual_temperature(
            temperature_profile_c[i - 1],
            pressure_profile_hpa[i - 1],
            dewpoint_profile_c[i - 1],
        );
        let tv_env = virtual_temperature(
            temperature_profile_c[i],
            pressure_profile_hpa[i],
            dewpoint_profile_c[i],
        );
        let buoy_prev = parcel_virtual_temperature_c[i - 1] - tv_env_prev;
        let buoy = parcel_virtual_temperature_c[i] - tv_env;

        if buoy_prev <= 0.0 && buoy > 0.0 {
            let frac = (0.0 - buoy_prev) / (buoy - buoy_prev);
            let pressure_lfc_hpa = pressure_profile_hpa[i - 1]
                + frac * (pressure_profile_hpa[i] - pressure_profile_hpa[i - 1]);
            let temperature_lfc_c = temperature_profile_c[i - 1]
                + frac * (temperature_profile_c[i] - temperature_profile_c[i - 1]);
            return Some((pressure_lfc_hpa, temperature_lfc_c));
        }

        if buoy > 0.0
            && pressure_profile_hpa[i] <= pressure_lcl_hpa
            && pressure_profile_hpa[i - 1] > pressure_lcl_hpa
        {
            return Some((pressure_profile_hpa[i], temperature_profile_c[i]));
        }
    }

    None
}

pub fn el(
    pressure_profile_hpa: &[f64],
    temperature_profile_c: &[f64],
    dewpoint_profile_c: &[f64],
) -> Option<(f64, f64)> {
    let (pressure_lcl_hpa, _temperature_lcl_c, parcel_virtual_temperature_c) = lift_parcel_profile(
        pressure_profile_hpa,
        temperature_profile_c,
        dewpoint_profile_c,
    )?;

    let mut found_positive = false;
    let mut last_el = None;
    for i in 1..pressure_profile_hpa.len() {
        if pressure_profile_hpa[i] > pressure_lcl_hpa {
            continue;
        }
        let tv_env_prev = virtual_temperature(
            temperature_profile_c[i - 1],
            pressure_profile_hpa[i - 1],
            dewpoint_profile_c[i - 1],
        );
        let tv_env = virtual_temperature(
            temperature_profile_c[i],
            pressure_profile_hpa[i],
            dewpoint_profile_c[i],
        );
        let buoy_prev = parcel_virtual_temperature_c[i - 1] - tv_env_prev;
        let buoy = parcel_virtual_temperature_c[i] - tv_env;

        if buoy > 0.0 {
            found_positive = true;
        }
        if found_positive && buoy_prev > 0.0 && buoy <= 0.0 {
            let frac = (0.0 - buoy_prev) / (buoy - buoy_prev);
            let pressure_el_hpa = pressure_profile_hpa[i - 1]
                + frac * (pressure_profile_hpa[i] - pressure_profile_hpa[i - 1]);
            let temperature_el_c = temperature_profile_c[i - 1]
                + frac * (temperature_profile_c[i] - temperature_profile_c[i - 1]);
            last_el = Some((pressure_el_hpa, temperature_el_c));
        }
    }
    last_el
}

pub fn lifted_index(
    pressure_profile_hpa: &[f64],
    temperature_profile_c: &[f64],
    dewpoint_profile_c: &[f64],
) -> f64 {
    let parcel_profile_c = parcel_profile(
        pressure_profile_hpa,
        temperature_profile_c[0],
        dewpoint_profile_c[0],
    );
    let (parcel_temperature_500_c, _) = get_env_at_pres(
        500.0,
        pressure_profile_hpa,
        &parcel_profile_c,
        &parcel_profile_c,
    );
    let (env_temperature_500_c, _) = get_env_at_pres(
        500.0,
        pressure_profile_hpa,
        temperature_profile_c,
        dewpoint_profile_c,
    );
    env_temperature_500_c - parcel_temperature_500_c
}

pub fn ccl(
    pressure_profile_hpa: &[f64],
    temperature_profile_c: &[f64],
    dewpoint_profile_c: &[f64],
) -> Option<(f64, f64)> {
    let surface_mixing_ratio_gkg = mixing_ratio(pressure_profile_hpa[0], dewpoint_profile_c[0]);
    for i in 1..pressure_profile_hpa.len() {
        let ws_prev = mixing_ratio(pressure_profile_hpa[i - 1], temperature_profile_c[i - 1]);
        let ws_curr = mixing_ratio(pressure_profile_hpa[i], temperature_profile_c[i]);
        if ws_prev >= surface_mixing_ratio_gkg && ws_curr < surface_mixing_ratio_gkg {
            let frac = (surface_mixing_ratio_gkg - ws_prev) / (ws_curr - ws_prev);
            let pressure_ccl_hpa = pressure_profile_hpa[i - 1]
                + frac * (pressure_profile_hpa[i] - pressure_profile_hpa[i - 1]);
            let temperature_ccl_c = temperature_profile_c[i - 1]
                + frac * (temperature_profile_c[i] - temperature_profile_c[i - 1]);
            return Some((pressure_ccl_hpa, temperature_ccl_c));
        }
    }
    None
}

pub fn showalter_index(
    pressure_profile_hpa: &[f64],
    temperature_profile_c: &[f64],
    dewpoint_profile_c: &[f64],
) -> f64 {
    let (temperature_850_c, dewpoint_850_c) = get_env_at_pres(
        850.0,
        pressure_profile_hpa,
        temperature_profile_c,
        dewpoint_profile_c,
    );
    let parcel_profile_c = parcel_profile(&[850.0, 500.0], temperature_850_c, dewpoint_850_c);
    let parcel_temperature_500_c = parcel_profile_c[1];

    let (env_temperature_500_c, _) = get_env_at_pres(
        500.0,
        pressure_profile_hpa,
        temperature_profile_c,
        dewpoint_profile_c,
    );
    env_temperature_500_c - parcel_temperature_500_c
}

pub fn k_index(t850_c: f64, td850_c: f64, t700_c: f64, td700_c: f64, t500_c: f64) -> f64 {
    (t850_c - t500_c) + td850_c - (t700_c - td700_c)
}

pub fn vertical_totals(t850_c: f64, t500_c: f64) -> f64 {
    t850_c - t500_c
}

pub fn cross_totals(td850_c: f64, t500_c: f64) -> f64 {
    td850_c - t500_c
}

pub fn total_totals(t850_c: f64, td850_c: f64, t500_c: f64) -> f64 {
    vertical_totals(t850_c, t500_c) + cross_totals(td850_c, t500_c)
}

pub fn sweat_index(
    t850_c: f64,
    td850_c: f64,
    t500_c: f64,
    dd850_deg: f64,
    dd500_deg: f64,
    ff850_kt: f64,
    ff500_kt: f64,
) -> f64 {
    let tt = total_totals(t850_c, td850_c, t500_c);
    let term_moisture = 12.0 * td850_c.max(0.0);
    let term_tt = if tt >= 49.0 { 20.0 * (tt - 49.0) } else { 0.0 };
    let term_shear = if dd850_deg >= 130.0
        && dd850_deg <= 250.0
        && dd500_deg >= 210.0
        && dd500_deg <= 310.0
        && (dd500_deg - dd850_deg) > 0.0
        && ff850_kt >= 15.0
        && ff500_kt >= 15.0
    {
        125.0 * ((dd500_deg - dd850_deg).to_radians().sin() + 0.2)
    } else {
        0.0
    };
    term_moisture + term_tt + 2.0 * ff850_kt + ff500_kt + term_shear
}

pub fn dry_lapse(pressure_hpa: &[f64], surface_temperature_c: f64) -> Vec<f64> {
    if pressure_hpa.is_empty() {
        return vec![];
    }
    let surface_temperature_k = surface_temperature_c + ZEROCNK;
    let surface_pressure_hpa = pressure_hpa[0];
    pressure_hpa
        .iter()
        .map(|&pressure| {
            surface_temperature_k * (pressure / surface_pressure_hpa).powf(ROCP) - ZEROCNK
        })
        .collect()
}

pub fn moist_lapse(pressure_hpa: &[f64], start_temperature_c: f64) -> Vec<f64> {
    if pressure_hpa.is_empty() {
        return vec![];
    }
    let mut result = Vec::with_capacity(pressure_hpa.len());
    result.push(start_temperature_c);

    let mut temperature_c = start_temperature_c;
    for i in 1..pressure_hpa.len() {
        let dp = pressure_hpa[i] - pressure_hpa[i - 1];
        if dp.abs() < 1e-10 {
            result.push(temperature_c);
            continue;
        }
        let n_steps = ((dp.abs() / 1.0).ceil() as usize).max(8);
        let h = dp / n_steps as f64;
        let mut pressure_current = pressure_hpa[i - 1];
        for _ in 0..n_steps {
            let k1 = h * moist_lapse_rate(pressure_current, temperature_c);
            let k2 = h * moist_lapse_rate(pressure_current + h / 2.0, temperature_c + k1 / 2.0);
            let k3 = h * moist_lapse_rate(pressure_current + h / 2.0, temperature_c + k2 / 2.0);
            let k4 = h * moist_lapse_rate(pressure_current + h, temperature_c + k3);
            temperature_c += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
            pressure_current += h;
        }
        result.push(temperature_c);
    }
    result
}

fn moist_lapse_from_reference(
    pressure_hpa: &[f64],
    start_temperature_c: f64,
    reference_pressure_hpa: f64,
) -> Vec<f64> {
    if pressure_hpa.is_empty() {
        return vec![];
    }
    let mut result = Vec::with_capacity(pressure_hpa.len());
    for &target_pressure_hpa in pressure_hpa {
        if (target_pressure_hpa - reference_pressure_hpa).abs() < 1.0e-10 {
            result.push(start_temperature_c);
            continue;
        }
        let dp = target_pressure_hpa - reference_pressure_hpa;
        let n_steps = ((dp.abs() / 1.0).ceil() as usize).max(8);
        let h = dp / n_steps as f64;
        let mut pressure_current = reference_pressure_hpa;
        let mut temperature_c = start_temperature_c;
        for _ in 0..n_steps {
            let k1 = h * moist_lapse_rate(pressure_current, temperature_c);
            let k2 = h * moist_lapse_rate(pressure_current + h / 2.0, temperature_c + k1 / 2.0);
            let k3 = h * moist_lapse_rate(pressure_current + h / 2.0, temperature_c + k2 / 2.0);
            let k4 = h * moist_lapse_rate(pressure_current + h, temperature_c + k3);
            temperature_c += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
            pressure_current += h;
        }
        result.push(temperature_c);
    }
    result
}

pub fn parcel_profile(
    pressure_hpa: &[f64],
    surface_temperature_c: f64,
    surface_dewpoint_c: f64,
) -> Vec<f64> {
    if pressure_hpa.is_empty() {
        return vec![];
    }
    let (pressure_lcl_hpa, temperature_lcl_exact_c) =
        drylift(pressure_hpa[0], surface_temperature_c, surface_dewpoint_c);
    let surface_temperature_k = surface_temperature_c + ZEROCNK;
    let surface_pressure_hpa = pressure_hpa[0];
    let temperature_lcl_parcel_c =
        surface_temperature_k * (pressure_lcl_hpa / surface_pressure_hpa).powf(ROCP) - ZEROCNK;

    let mut moist_pressures = vec![pressure_lcl_hpa];
    for &pressure in pressure_hpa {
        if pressure < pressure_lcl_hpa {
            moist_pressures.push(pressure);
        }
    }
    let moist_temperatures = moist_lapse(&moist_pressures, temperature_lcl_parcel_c);

    let mut result = Vec::with_capacity(pressure_hpa.len());
    let mut moist_idx = 1usize;
    for &pressure in pressure_hpa {
        if pressure > pressure_lcl_hpa {
            let temperature_k =
                surface_temperature_k * (pressure / surface_pressure_hpa).powf(ROCP);
            result.push(temperature_k - ZEROCNK);
        } else if (pressure - pressure_lcl_hpa).abs() <= 1.0e-9 {
            result.push(temperature_lcl_exact_c);
        } else if moist_idx < moist_temperatures.len() {
            result.push(moist_temperatures[moist_idx]);
            moist_idx += 1;
        } else {
            let theta_k =
                (temperature_lcl_parcel_c + ZEROCNK) * (1000.0 / pressure_lcl_hpa).powf(ROCP);
            let theta_c = theta_k - ZEROCNK;
            let thetam = theta_c - wobf(theta_c) + wobf(temperature_lcl_parcel_c);
            result.push(satlift(pressure, thetam));
        }
    }
    result
}

pub fn parcel_profile_with_lcl(
    pressure_hpa: &[f64],
    surface_temperature_c: f64,
    surface_dewpoint_c: f64,
) -> (Vec<f64>, Vec<f64>) {
    if pressure_hpa.is_empty() {
        return (vec![], vec![]);
    }

    let (pressure_lcl_hpa, _temperature_lcl_c) =
        drylift(pressure_hpa[0], surface_temperature_c, surface_dewpoint_c);

    let mut pressure_aug = Vec::with_capacity(pressure_hpa.len() + 1);
    let mut lcl_inserted = false;
    for &pressure in pressure_hpa {
        if !lcl_inserted && pressure <= pressure_lcl_hpa {
            if (pressure - pressure_lcl_hpa).abs() > 0.01 {
                pressure_aug.push(pressure_lcl_hpa);
            }
            lcl_inserted = true;
        }
        pressure_aug.push(pressure);
    }
    if !lcl_inserted {
        pressure_aug.push(pressure_lcl_hpa);
    }

    let temperature_aug = parcel_profile(&pressure_aug, surface_temperature_c, surface_dewpoint_c);

    (pressure_aug, temperature_aug)
}

#[derive(Clone, Copy)]
enum IntersectionDirection {
    All,
    Increasing,
    Decreasing,
}

fn metpy_parcel_profile_with_lcl(
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    dewpoint_c: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    if pressure_hpa.is_empty() || temperature_c.is_empty() || dewpoint_c.is_empty() {
        return (vec![], vec![], vec![], vec![]);
    }
    let (pressure_aug, parcel_profile_c) =
        parcel_profile_with_lcl(pressure_hpa, temperature_c[0], dewpoint_c[0]);
    let mut temperature_aug = Vec::with_capacity(pressure_aug.len());
    let mut dewpoint_aug = Vec::with_capacity(pressure_aug.len());
    for &pressure in &pressure_aug {
        if pressure >= pressure_hpa[0] {
            temperature_aug.push(temperature_c[0]);
            dewpoint_aug.push(dewpoint_c[0]);
            continue;
        }
        if pressure <= pressure_hpa[pressure_hpa.len() - 1] {
            temperature_aug.push(*temperature_c.last().unwrap_or(&temperature_c[0]));
            dewpoint_aug.push(*dewpoint_c.last().unwrap_or(&dewpoint_c[0]));
            continue;
        }

        let mut inserted = false;
        for i in 0..pressure_hpa.len().saturating_sub(1) {
            if pressure_hpa[i] >= pressure && pressure >= pressure_hpa[i + 1] {
                let frac = if (pressure_hpa[i + 1] - pressure_hpa[i]).abs() < 1.0e-12 {
                    0.0
                } else {
                    (pressure - pressure_hpa[i]) / (pressure_hpa[i + 1] - pressure_hpa[i])
                };
                temperature_aug
                    .push(temperature_c[i] + frac * (temperature_c[i + 1] - temperature_c[i]));
                dewpoint_aug.push(dewpoint_c[i] + frac * (dewpoint_c[i + 1] - dewpoint_c[i]));
                inserted = true;
                break;
            }
        }
        if !inserted {
            temperature_aug.push(*temperature_c.last().unwrap_or(&temperature_c[0]));
            dewpoint_aug.push(*dewpoint_c.last().unwrap_or(&dewpoint_c[0]));
        }
    }
    (
        pressure_aug,
        temperature_aug,
        dewpoint_aug,
        parcel_profile_c,
    )
}

fn remove_nan_profile(
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    dewpoint_c: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = pressure_hpa
        .len()
        .min(temperature_c.len())
        .min(dewpoint_c.len());
    let mut pressure = Vec::with_capacity(n);
    let mut temperature = Vec::with_capacity(n);
    let mut dewpoint = Vec::with_capacity(n);
    for i in 0..n {
        let pressure_i = pressure_hpa[i];
        let temperature_i = temperature_c[i];
        let dewpoint_i = dewpoint_c[i];
        if pressure_i.is_finite()
            && pressure_i > 0.0
            && temperature_i.is_finite()
            && dewpoint_i.is_finite()
        {
            pressure.push(pressure_i);
            temperature.push(temperature_i);
            dewpoint.push(dewpoint_i.min(temperature_i));
        }
    }
    (pressure, temperature, dewpoint)
}

fn find_intersections_log_p(
    pressure_hpa: &[f64],
    a: &[f64],
    b: &[f64],
    direction: IntersectionDirection,
) -> Vec<f64> {
    let n = pressure_hpa.len().min(a.len()).min(b.len());
    if n < 2 {
        return vec![];
    }

    let mut intersections = Vec::new();
    for i in 0..n - 1 {
        let delta0 = a[i] - b[i];
        let delta1 = a[i + 1] - b[i + 1];
        if !delta0.is_finite()
            || !delta1.is_finite()
            || !pressure_hpa[i].is_finite()
            || !pressure_hpa[i + 1].is_finite()
            || pressure_hpa[i] <= 0.0
            || pressure_hpa[i + 1] <= 0.0
        {
            continue;
        }
        if delta0.abs() < 1.0e-12 && delta1.abs() < 1.0e-12 {
            continue;
        }
        if delta0 * delta1 > 0.0 {
            continue;
        }
        if (delta1 - delta0).abs() < 1.0e-12 {
            continue;
        }

        let sign_change = if delta1.abs() > 1.0e-12 {
            delta1.signum()
        } else {
            delta0.signum()
        };
        let keep = match direction {
            IntersectionDirection::All => true,
            IntersectionDirection::Increasing => sign_change > 0.0,
            IntersectionDirection::Decreasing => sign_change < 0.0,
        };
        if !keep {
            continue;
        }

        let log_p0 = pressure_hpa[i].ln();
        let log_p1 = pressure_hpa[i + 1].ln();
        let intersect_log_p = (delta1 * log_p0 - delta0 * log_p1) / (delta1 - delta0);
        let intersect_pressure = intersect_log_p.exp();
        if intersections
            .last()
            .is_none_or(|last| (last - intersect_pressure).abs() > 1.0e-6)
        {
            intersections.push(intersect_pressure);
        }
    }

    intersections
}

fn append_zero_crossings_log_p(pressure_hpa: &[f64], values: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = pressure_hpa.len().min(values.len());
    let mut combined = pressure_hpa
        .iter()
        .take(n)
        .copied()
        .zip(values.iter().take(n).copied())
        .collect::<Vec<_>>();
    if n > 1 {
        for pressure in find_intersections_log_p(
            &pressure_hpa[1..n],
            &values[1..n],
            &vec![0.0; n - 1],
            IntersectionDirection::All,
        ) {
            combined.push((pressure, 0.0));
        }
    }
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    combined.dedup_by(|a, b| (a.0 - b.0).abs() <= 1.0e-6);
    combined.into_iter().unzip()
}

fn trapz(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    for i in 1..n {
        total += 0.5 * (y[i - 1] + y[i]) * (x[i] - x[i - 1]);
    }
    total
}

fn parcel_virtual_temperature_from_mixing_ratio(temperature_c: f64, mixing_ratio_kgkg: f64) -> f64 {
    let temperature_k = temperature_c + ZEROCNK;
    temperature_k * (1.0 + mixing_ratio_kgkg / EPSILON) / (1.0 + mixing_ratio_kgkg) - ZEROCNK
}

fn metpy_lfc_pressure(
    pressure_hpa: &[f64],
    temperature_profile_c: &[f64],
    dewpoint_c: &[f64],
    parcel_profile_c: &[f64],
) -> Option<f64> {
    let n = pressure_hpa
        .len()
        .min(temperature_profile_c.len())
        .min(dewpoint_c.len())
        .min(parcel_profile_c.len());
    if n < 2 {
        return None;
    }

    let start_idx = if (parcel_profile_c[0] - temperature_profile_c[0]).abs() < 1.0e-9 {
        1
    } else {
        0
    };
    let intersections = find_intersections_log_p(
        &pressure_hpa[start_idx..n],
        &parcel_profile_c[start_idx..n],
        &temperature_profile_c[start_idx..n],
        IntersectionDirection::Increasing,
    );
    let pressure_lcl_hpa = lcl(pressure_hpa[0], parcel_profile_c[0], dewpoint_c[0]).0;

    if intersections.is_empty() {
        let has_positive_area = pressure_hpa
            .iter()
            .zip(parcel_profile_c.iter())
            .zip(temperature_profile_c.iter())
            .take(n)
            .any(|((&pressure, &parcel), &environment)| {
                pressure < pressure_lcl_hpa && parcel > environment
            });
        if !has_positive_area {
            None
        } else {
            Some(pressure_lcl_hpa)
        }
    } else {
        let above_lcl = intersections
            .iter()
            .copied()
            .filter(|&pressure| pressure < pressure_lcl_hpa)
            .collect::<Vec<_>>();
        if above_lcl.is_empty() {
            let el_pressures = find_intersections_log_p(
                &pressure_hpa[1..n],
                &parcel_profile_c[1..n],
                &temperature_profile_c[1..n],
                IntersectionDirection::Decreasing,
            );
            if !el_pressures.is_empty()
                && el_pressures.iter().copied().fold(f64::INFINITY, f64::min) > pressure_lcl_hpa
            {
                None
            } else {
                Some(pressure_lcl_hpa)
            }
        } else {
            Some(above_lcl.into_iter().fold(f64::NEG_INFINITY, f64::max))
        }
    }
}

fn metpy_el_pressure(
    pressure_hpa: &[f64],
    temperature_profile_c: &[f64],
    dewpoint_c: &[f64],
    parcel_profile_c: &[f64],
) -> Option<f64> {
    let n = pressure_hpa
        .len()
        .min(temperature_profile_c.len())
        .min(dewpoint_c.len())
        .min(parcel_profile_c.len());
    if n < 2 || parcel_profile_c[n - 1] > temperature_profile_c[n - 1] {
        return None;
    }

    let intersections = find_intersections_log_p(
        &pressure_hpa[1..n],
        &parcel_profile_c[1..n],
        &temperature_profile_c[1..n],
        IntersectionDirection::Decreasing,
    );
    let pressure_lcl_hpa = lcl(pressure_hpa[0], parcel_profile_c[0], dewpoint_c[0]).0;
    let above_lcl = intersections
        .into_iter()
        .filter(|&pressure| pressure < pressure_lcl_hpa)
        .collect::<Vec<_>>();
    if above_lcl.is_empty() {
        None
    } else {
        Some(above_lcl.into_iter().fold(f64::INFINITY, f64::min))
    }
}

fn metpy_cape_cin_from_profile(
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    dewpoint_c: &[f64],
    parcel_profile_c: &[f64],
) -> (f64, f64) {
    let n = pressure_hpa
        .len()
        .min(temperature_c.len())
        .min(dewpoint_c.len())
        .min(parcel_profile_c.len());
    if n < 2 {
        return (0.0, 0.0);
    }

    let pressure_lcl_hpa = lcl(pressure_hpa[0], temperature_c[0], dewpoint_c[0]).0;
    let parcel_surface_mixing_ratio_kgkg =
        saturation_mixing_ratio(pressure_hpa[0], dewpoint_c[0]) / 1000.0;

    let mut env_tv = Vec::with_capacity(n);
    let mut parcel_tv = Vec::with_capacity(n);
    let mut delta_tv = Vec::with_capacity(n);
    for i in 0..n {
        let env_tv_i = virtual_temperature(temperature_c[i], pressure_hpa[i], dewpoint_c[i]);
        let parcel_mixing_ratio_kgkg = if pressure_hpa[i] > pressure_lcl_hpa {
            parcel_surface_mixing_ratio_kgkg
        } else {
            saturation_mixing_ratio(pressure_hpa[i], parcel_profile_c[i]) / 1000.0
        };
        let parcel_tv_i = parcel_virtual_temperature_from_mixing_ratio(
            parcel_profile_c[i],
            parcel_mixing_ratio_kgkg,
        );
        env_tv.push(env_tv_i);
        parcel_tv.push(parcel_tv_i);
        delta_tv.push(parcel_tv_i - env_tv_i);
    }

    let lfc_pressure_hpa =
        match metpy_lfc_pressure(&pressure_hpa[..n], &env_tv, &dewpoint_c[..n], &parcel_tv) {
            Some(pressure) => pressure,
            None => return (0.0, 0.0),
        };
    let el_pressure_hpa =
        metpy_el_pressure(&pressure_hpa[..n], &env_tv, &dewpoint_c[..n], &parcel_tv)
            .unwrap_or(pressure_hpa[n - 1]);

    let (pressure_with_crossings, delta_with_crossings) =
        append_zero_crossings_log_p(&pressure_hpa[..n], &delta_tv);

    let mut cape_pressures = Vec::new();
    let mut cape_values = Vec::new();
    let mut cin_pressures = Vec::new();
    let mut cin_values = Vec::new();
    for (&pressure, &delta) in pressure_with_crossings
        .iter()
        .zip(delta_with_crossings.iter())
    {
        if pressure <= lfc_pressure_hpa && pressure >= el_pressure_hpa {
            cape_pressures.push(pressure.ln());
            cape_values.push(delta);
        }
        if pressure >= lfc_pressure_hpa {
            cin_pressures.push(pressure.ln());
            cin_values.push(delta);
        }
    }

    let cape = RD * trapz(&cape_pressures, &cape_values);
    let mut cin = RD * trapz(&cin_pressures, &cin_values);
    if cin > 0.0 {
        cin = 0.0;
    }
    (cape, cin)
}

pub fn saturation_equivalent_potential_temperature(pressure_hpa: f64, temperature_c: f64) -> f64 {
    equivalent_potential_temperature(pressure_hpa, temperature_c, temperature_c)
}

pub fn static_stability(pressure_hpa: &[f64], temperature_k: &[f64]) -> Vec<f64> {
    let n = pressure_hpa.len().min(temperature_k.len());
    if n < 2 {
        return vec![0.0; n];
    }
    let theta: Vec<f64> = pressure_hpa
        .iter()
        .zip(temperature_k.iter())
        .map(|(&pressure, &temperature)| temperature * (1000.0 / pressure).powf(ROCP))
        .collect();

    let mut result = vec![0.0; n];
    for i in 0..n {
        let (dtheta, dp) = centered_difference_pair(pressure_hpa, &theta, i);
        if dp.abs() < 1e-10 || theta[i].abs() < 1e-10 {
            result[i] = 0.0;
        } else {
            result[i] = -(temperature_k[i] / theta[i]) * (dtheta / (dp * 100.0));
        }
    }
    result
}

pub fn mixed_layer(pressure_hpa: &[f64], values: &[f64], depth_hpa: f64) -> f64 {
    let n = pressure_hpa.len().min(values.len());
    if n == 0 {
        return 0.0;
    }
    if n == 1 || depth_hpa <= 0.0 {
        return values[0];
    }
    let pressure_hpa = &pressure_hpa[..n];
    let values = &values[..n];
    let surface_pressure_hpa = pressure_hpa[0];
    let top_pressure_hpa = surface_pressure_hpa - depth_hpa;
    let mut p_layer = pressure_hpa
        .iter()
        .copied()
        .take_while(|&pressure| pressure >= top_pressure_hpa)
        .collect::<Vec<_>>();
    if p_layer.is_empty() {
        return values[0];
    }
    if p_layer
        .last()
        .is_none_or(|last| (last - top_pressure_hpa).abs() > 1.0e-6)
    {
        p_layer.push(top_pressure_hpa);
    }
    let v_layer = log_interpolate_1d(&p_layer, pressure_hpa, values);
    let actual_depth = (p_layer[0] - p_layer[p_layer.len() - 1]).abs();
    if actual_depth <= 1.0e-10 {
        values[0]
    } else {
        trapz(&p_layer, &v_layer) / -actual_depth
    }
}

pub fn get_mixed_layer_parcel(
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    dewpoint_c: &[f64],
    depth_hpa: f64,
) -> (f64, f64, f64) {
    if pressure_hpa.is_empty() || temperature_c.is_empty() || dewpoint_c.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let surface_pressure_hpa = pressure_hpa[0];
    let theta: Vec<f64> = pressure_hpa
        .iter()
        .zip(temperature_c.iter())
        .map(|(&pressure, &temperature)| (temperature + ZEROCNK) * (1000.0 / pressure).powf(ROCP))
        .collect();
    let mixing_ratio_profile: Vec<f64> = pressure_hpa
        .iter()
        .zip(dewpoint_c.iter())
        .map(|(&pressure, &dewpoint)| saturation_mixing_ratio(pressure, dewpoint))
        .collect();
    let mean_theta = mixed_layer(pressure_hpa, &theta, depth_hpa);
    let mean_mixing_ratio_gkg = mixed_layer(pressure_hpa, &mixing_ratio_profile, depth_hpa);
    let mean_temperature_k = mean_theta * (surface_pressure_hpa / 1000.0).powf(ROCP);
    let mean_temperature_c = mean_temperature_k - ZEROCNK;
    let mean_specific_humidity = specific_humidity(surface_pressure_hpa, mean_mixing_ratio_gkg);
    let mean_dewpoint_c =
        dewpoint_from_specific_humidity(surface_pressure_hpa, mean_specific_humidity);

    (
        surface_pressure_hpa,
        mean_temperature_c,
        mean_dewpoint_c.min(mean_temperature_c),
    )
}

pub fn get_most_unstable_parcel(
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    dewpoint_c: &[f64],
    depth_hpa: f64,
) -> (f64, f64, f64) {
    let surface_pressure_hpa = pressure_hpa[0];
    let limit_pressure_hpa = surface_pressure_hpa - depth_hpa;
    let mut max_theta_e = -999.0;
    let mut best_idx = 0usize;
    let top_idx = pressure_hpa
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (*a - limit_pressure_hpa)
                .abs()
                .partial_cmp(&(*b - limit_pressure_hpa).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(pressure_hpa.len().saturating_sub(1));

    for i in 0..=top_idx {
        let theta_e =
            equivalent_potential_temperature(pressure_hpa[i], temperature_c[i], dewpoint_c[i]);
        if theta_e > max_theta_e {
            max_theta_e = theta_e;
            best_idx = i;
        }
    }
    (
        pressure_hpa[best_idx],
        temperature_c[best_idx],
        dewpoint_c[best_idx],
    )
}

pub fn surface_based_cape_cin(
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    dewpoint_c: &[f64],
) -> (f64, f64) {
    let (pressure, temperature, dewpoint) =
        remove_nan_profile(pressure_hpa, temperature_c, dewpoint_c);
    if pressure.len() < 2 {
        return (0.0, 0.0);
    }
    let (pressure_aug, temperature_aug, dewpoint_aug, parcel_profile) =
        metpy_parcel_profile_with_lcl(&pressure, &temperature, &dewpoint);
    metpy_cape_cin_from_profile(
        &pressure_aug,
        &temperature_aug,
        &dewpoint_aug,
        &parcel_profile,
    )
}

pub fn mixed_layer_cape_cin(
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    dewpoint_c: &[f64],
    depth_hpa: f64,
) -> (f64, f64) {
    let (pressure, temperature, dewpoint) =
        remove_nan_profile(pressure_hpa, temperature_c, dewpoint_c);
    if pressure.len() < 2 {
        return (0.0, 0.0);
    }
    let (parcel_pressure, parcel_temperature, parcel_dewpoint) =
        get_mixed_layer_parcel(&pressure, &temperature, &dewpoint, depth_hpa);
    let top_pressure = pressure[0] - depth_hpa;
    let mut pressure_prof = vec![parcel_pressure];
    let mut temperature_prof = vec![parcel_temperature];
    let mut dewpoint_prof = vec![parcel_dewpoint];
    for i in 0..pressure.len() {
        if pressure[i] < top_pressure {
            pressure_prof.push(pressure[i]);
            temperature_prof.push(temperature[i]);
            dewpoint_prof.push(dewpoint[i]);
        }
    }
    let (pressure_aug, temperature_aug, dewpoint_aug, parcel_profile) =
        metpy_parcel_profile_with_lcl(&pressure_prof, &temperature_prof, &dewpoint_prof);
    metpy_cape_cin_from_profile(
        &pressure_aug,
        &temperature_aug,
        &dewpoint_aug,
        &parcel_profile,
    )
}

pub fn most_unstable_cape_cin(
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    dewpoint_c: &[f64],
) -> (f64, f64) {
    let (pressure, temperature, dewpoint) =
        remove_nan_profile(pressure_hpa, temperature_c, dewpoint_c);
    if pressure.len() < 2 {
        return (0.0, 0.0);
    }
    let (p_start, _, _) = get_most_unstable_parcel(&pressure, &temperature, &dewpoint, 300.0);
    let start_idx = pressure
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (*a - p_start)
                .abs()
                .partial_cmp(&(*b - p_start).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    let (pressure_aug, temperature_aug, dewpoint_aug, parcel_profile) =
        metpy_parcel_profile_with_lcl(
            &pressure[start_idx..],
            &temperature[start_idx..],
            &dewpoint[start_idx..],
        );
    metpy_cape_cin_from_profile(
        &pressure_aug,
        &temperature_aug,
        &dewpoint_aug,
        &parcel_profile,
    )
}

pub fn cape_cin(
    pressure_profile_hpa: &[f64],
    temperature_profile_c: &[f64],
    dewpoint_profile_c: &[f64],
    height_agl_m: &[f64],
    surface_pressure_hpa: f64,
    temperature_2m_c: f64,
    dewpoint_2m_c: f64,
    parcel_type: &str,
    mixed_layer_depth_hpa: f64,
    most_unstable_depth_hpa: f64,
    top_m: Option<f64>,
) -> (f64, f64, f64, f64) {
    let mut pressure = pressure_profile_hpa.to_vec();
    let mut temperature = temperature_profile_c.to_vec();
    let mut dewpoint = dewpoint_profile_c.to_vec();
    let mut psfc = surface_pressure_hpa;
    let mut t2m = temperature_2m_c;
    let mut td2m = dewpoint_2m_c.min(temperature_2m_c);

    if psfc > 2000.0 {
        for p in &mut pressure {
            *p /= 100.0;
        }
        psfc /= 100.0;
    }
    if t2m > 150.0 {
        for t in &mut temperature {
            *t -= ZEROCNK;
        }
        for td in &mut dewpoint {
            *td -= ZEROCNK;
        }
        t2m -= ZEROCNK;
        td2m -= ZEROCNK;
    }

    let n = pressure.len().min(temperature.len()).min(dewpoint.len());
    let mut p_prof = Vec::with_capacity(n + 1);
    let mut t_prof = Vec::with_capacity(n + 1);
    let mut td_prof = Vec::with_capacity(n + 1);
    let mut z_prof = Vec::with_capacity(n + 1);
    p_prof.push(psfc);
    t_prof.push(t2m);
    td_prof.push(td2m.min(t2m));
    z_prof.push(0.0);
    for i in 0..n {
        p_prof.push(pressure[i]);
        t_prof.push(temperature[i]);
        td_prof.push(dewpoint[i].min(temperature[i]));
        z_prof.push(*height_agl_m.get(i).unwrap_or(&0.0));
    }

    let (p_start, t_start, td_start) = match parcel_type {
        "ml" | "mixed_layer" => {
            get_mixed_layer_parcel(&p_prof, &t_prof, &td_prof, mixed_layer_depth_hpa)
        }
        "mu" | "most_unstable" => {
            get_most_unstable_parcel(&p_prof, &t_prof, &td_prof, most_unstable_depth_hpa)
        }
        _ => (psfc, t2m, td2m),
    };

    let (p_lcl, t_lcl) = drylift(p_start, t_start, td_start);
    let h_lcl = get_height_at_pres(p_lcl, &p_prof, &z_prof);
    let theta_start_k = (t_lcl + ZEROCNK) * (1000.0 / p_lcl).powf(ROCP);
    let theta_start_c = theta_start_k - ZEROCNK;
    let thetam = theta_start_c - wobf(theta_start_c) + wobf(t_lcl);

    let mut el_p = p_lcl;
    let mut lfc_p = p_lcl;
    let mut found_positive = false;
    let mut in_positive = false;

    let mut start_idx = 0;
    for (i, p) in p_prof.iter().enumerate() {
        if *p <= p_lcl {
            start_idx = i;
            break;
        }
    }

    for i in start_idx..p_prof.len() {
        let p_curr = p_prof[i];
        let tv_env = virtual_temperature(t_prof[i], p_curr, td_prof[i]);
        let t_parcel = satlift(p_curr, thetam);
        let tv_parcel = virtual_temperature(t_parcel, p_curr, t_parcel);
        let buoyancy = tv_parcel - tv_env;

        if buoyancy > 0.0 {
            if !in_positive {
                in_positive = true;
                let bottom = if i > 0 {
                    let p_prev = p_prof[i - 1];
                    let tv_env_prev = virtual_temperature(t_prof[i - 1], p_prev, td_prof[i - 1]);
                    let t_parcel_prev = satlift(p_prev, thetam);
                    let tv_parcel_prev = virtual_temperature(t_parcel_prev, p_prev, t_parcel_prev);
                    let buoy_prev = tv_parcel_prev - tv_env_prev;
                    if (buoyancy - buoy_prev).abs() > 1e-12 {
                        let frac = (0.0 - buoy_prev) / (buoyancy - buoy_prev);
                        p_prev + frac * (p_curr - p_prev)
                    } else {
                        p_curr
                    }
                } else {
                    p_curr
                };
                lfc_p = bottom;
                el_p = *p_prof.last().unwrap_or(&p_lcl);
                found_positive = true;
            }
        } else if in_positive {
            in_positive = false;
            let p_prev = p_prof[i - 1];
            let tv_env_prev = virtual_temperature(t_prof[i - 1], p_prev, td_prof[i - 1]);
            let t_parcel_prev = satlift(p_prev, thetam);
            let tv_parcel_prev = virtual_temperature(t_parcel_prev, p_prev, t_parcel_prev);
            let buoy_prev = tv_parcel_prev - tv_env_prev;
            el_p = if (buoyancy - buoy_prev).abs() > 1e-12 {
                let frac = (0.0 - buoy_prev) / (buoyancy - buoy_prev);
                p_prev + frac * (p_curr - p_prev)
            } else {
                p_curr
            };
        }
    }

    if in_positive {
        el_p = *p_prof.last().unwrap_or(&p_lcl);
    }
    if !found_positive {
        return (0.0, 0.0, h_lcl, f64::NAN);
    }
    if lfc_p.is_nan() || lfc_p > p_lcl {
        lfc_p = p_lcl;
    }
    let h_lfc = get_height_at_pres(lfc_p, &p_prof, &z_prof);

    let mut p_top_limit = el_p;
    if let Some(top_cap_m) = top_m {
        let n_top = p_prof.len();
        let h_rev: Vec<f64> = z_prof.iter().copied().rev().collect();
        let p_rev: Vec<f64> = p_prof.iter().copied().rev().collect();
        let p_top_m = get_height_at_pres(top_cap_m, &h_rev, &p_rev);
        if p_top_m >= p_top_limit {
            p_top_limit = p_top_m.max(p_prof[n_top - 1]);
        }
    }

    let theta_dry_k = (t_start + ZEROCNK) * ((1000.0 / p_start).powf(ROCP));
    let r_parcel = mixing_ratio(p_start, td_start);
    let w_kgkg = r_parcel / 1000.0;

    let mut p_moist = vec![p_lcl];
    for &pi in &p_prof {
        if pi < p_lcl && pi > 0.0 {
            p_moist.push(pi);
        }
    }
    let moist_temps = if p_moist.len() > 1 {
        moist_lapse(&p_moist, t_lcl)
    } else {
        vec![t_lcl]
    };

    let n_prof = p_prof.len();
    let mut tv_parcel = vec![f64::NAN; n_prof];
    let mut tv_env = vec![0.0; n_prof];
    for i in 0..n_prof {
        if p_prof[i] <= 0.0 {
            continue;
        }
        tv_env[i] = virtual_temperature(t_prof[i], p_prof[i], td_prof[i]);
        if p_prof[i] >= p_lcl {
            let t_parc_k = theta_dry_k * (p_prof[i] / 1000.0).powf(ROCP);
            let t_parc = t_parc_k - ZEROCNK;
            tv_parcel[i] = (t_parc + ZEROCNK) * (1.0 + w_kgkg / EPSILON) / (1.0 + w_kgkg) - ZEROCNK;
        } else {
            let t_parc = interp_log_p(p_prof[i], &p_moist, &moist_temps);
            tv_parcel[i] = virtual_temperature(t_parc, p_prof[i], t_parc);
        }
    }

    let mut z_calc = vec![0.0; n_prof];
    for i in 1..n_prof {
        if p_prof[i] <= 0.0 || p_prof[i - 1] <= 0.0 {
            z_calc[i] = z_calc[i - 1];
            continue;
        }
        let tv_mean_k = (tv_env[i - 1] + tv_env[i]) / 2.0 + ZEROCNK;
        z_calc[i] = z_calc[i - 1] + (RD * tv_mean_k / G) * (p_prof[i - 1] / p_prof[i]).ln();
    }
    let z_use = if z_prof.iter().any(|&height| height > 0.0) {
        z_prof
    } else {
        z_calc
    };

    let p_top_actual = if p_top_limit > 0.0 {
        p_top_limit
    } else {
        p_prof[n_prof - 1]
    };
    let mut cape = 0.0;
    let mut cin = 0.0;
    for i in 1..n_prof {
        if p_prof[i] <= 0.0 || tv_parcel[i].is_nan() || tv_parcel[i - 1].is_nan() {
            continue;
        }
        if p_prof[i] < p_top_actual {
            continue;
        }
        let tv_e_lo = tv_env[i - 1] + ZEROCNK;
        let tv_e_hi = tv_env[i] + ZEROCNK;
        let tv_p_lo = tv_parcel[i - 1] + ZEROCNK;
        let tv_p_hi = tv_parcel[i] + ZEROCNK;
        let dz = z_use[i] - z_use[i - 1];
        if dz.abs() < 1e-6 || tv_e_lo <= 0.0 || tv_e_hi <= 0.0 {
            continue;
        }
        let buoy_lo = (tv_p_lo - tv_e_lo) / tv_e_lo;
        let buoy_hi = (tv_p_hi - tv_e_hi) / tv_e_hi;
        let val = G * (buoy_lo + buoy_hi) / 2.0 * dz;
        if val > 0.0 {
            cape += val;
        } else {
            cin += val;
        }
    }

    (cape, cin, h_lcl, h_lfc)
}

pub fn isentropic_interpolation(
    theta_levels_k: &[f64],
    pressure_3d_hpa: &[f64],
    temperature_3d_k: &[f64],
    fields: &[&[f64]],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Vec<Vec<f64>> {
    let n2d = nx * ny;
    let n_theta = theta_levels_k.len();
    let n_fields = fields.len();
    let total_output = 2 + n_fields;
    let mut output: Vec<Vec<f64>> = (0..total_output)
        .map(|_| vec![f64::NAN; n_theta * n2d])
        .collect();

    let mut theta_3d = vec![0.0; nz * n2d];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx3 = k * n2d + j * nx + i;
                let t_k = temperature_3d_k[idx3];
                let p_hpa = pressure_3d_hpa[idx3];
                if p_hpa > 0.0 && t_k > 0.0 {
                    theta_3d[idx3] = t_k * (1000.0 / p_hpa).powf(ROCP);
                }
            }
        }
    }

    for j in 0..ny {
        for i in 0..nx {
            let idx2 = j * nx + i;
            let mut col_theta = Vec::with_capacity(nz);
            let mut col_p = Vec::with_capacity(nz);
            let mut col_t = Vec::with_capacity(nz);
            let mut col_fields: Vec<Vec<f64>> =
                (0..n_fields).map(|_| Vec::with_capacity(nz)).collect();

            for k in 0..nz {
                let idx3 = k * n2d + idx2;
                col_theta.push(theta_3d[idx3]);
                col_p.push(pressure_3d_hpa[idx3]);
                col_t.push(temperature_3d_k[idx3]);
                for f in 0..n_fields {
                    col_fields[f].push(fields[f][idx3]);
                }
            }

            let mut sort_idx: Vec<usize> = (0..nz).collect();
            sort_idx.sort_by(|&a, &b| {
                col_theta[a]
                    .partial_cmp(&col_theta[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for (ti, &target_theta) in theta_levels_k.iter().enumerate() {
                let out_idx = ti * n2d + idx2;
                let mut found = false;
                for k in 0..nz.saturating_sub(1) {
                    let th_lo = col_theta[k];
                    let th_hi = col_theta[k + 1];
                    if (th_lo <= target_theta && th_hi >= target_theta)
                        || (th_lo >= target_theta && th_hi <= target_theta)
                    {
                        let dth = th_hi - th_lo;
                        if dth.abs() < 1e-10 {
                            continue;
                        }

                        let ln_p_lo = col_p[k].ln();
                        let ln_p_hi = col_p[k + 1].ln();
                        let d_ln_p = ln_p_hi - ln_p_lo;
                        if d_ln_p.abs() < 1e-10 {
                            continue;
                        }

                        let a = (col_t[k + 1] - col_t[k]) / d_ln_p;
                        let b = col_t[k] - a * ln_p_lo;
                        let pok = 1000.0_f64.powf(ROCP);

                        let mut ln_p = (ln_p_lo + ln_p_hi) / 2.0;
                        for _ in 0..50 {
                            let exner = pok * (-ROCP * ln_p).exp();
                            let t = a * ln_p + b;
                            let f = target_theta - t * exner;
                            let fp = exner * (ROCP * t - a);
                            if fp.abs() < 1e-30 {
                                break;
                            }
                            let delta = f / fp;
                            ln_p -= delta;
                            if delta.abs() < 1e-10 {
                                break;
                            }
                        }

                        output[0][out_idx] = ln_p.exp();
                        output[1][out_idx] = a * ln_p + b;
                        found = true;
                        break;
                    }
                }

                if !found {
                    continue;
                }

                for sk in 0..nz.saturating_sub(1) {
                    let i_lo = sort_idx[sk];
                    let i_hi = sort_idx[sk + 1];
                    let th_lo = col_theta[i_lo];
                    let th_hi = col_theta[i_hi];
                    if th_lo <= target_theta
                        && th_hi >= target_theta
                        && (th_hi - th_lo).abs() > 1e-10
                    {
                        let frac = (target_theta - th_lo) / (th_hi - th_lo);
                        for f in 0..n_fields {
                            output[2 + f][out_idx] = col_fields[f][i_lo]
                                + frac * (col_fields[f][i_hi] - col_fields[f][i_lo]);
                        }
                        break;
                    }
                }
            }
        }
    }

    output
}

pub fn mixed_parcel(
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    dewpoint_c: &[f64],
    depth_hpa: f64,
) -> (f64, f64, f64) {
    get_mixed_layer_parcel(pressure_hpa, temperature_c, dewpoint_c, depth_hpa)
}

pub fn most_unstable_parcel(
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    dewpoint_c: &[f64],
    depth_hpa: f64,
) -> (f64, f64, f64) {
    get_most_unstable_parcel(pressure_hpa, temperature_c, dewpoint_c, depth_hpa)
}

pub fn psychrometric_vapor_pressure_wet(
    dry_bulb_c: f64,
    wet_bulb_c: f64,
    pressure_hpa: f64,
) -> f64 {
    psychrometric_vapor_pressure(dry_bulb_c, wet_bulb_c, pressure_hpa)
}

pub fn density(pressure_hpa: f64, temperature_c: f64, mixing_ratio_gkg: f64) -> f64 {
    let pressure_pa = pressure_hpa * 100.0;
    let temperature_k = temperature_c + ZEROCNK;
    let mixing_ratio_kgkg = mixing_ratio_gkg / 1000.0;
    let virtual_temperature_k = temperature_k * (1.0 + 0.61 * mixing_ratio_kgkg);
    pressure_pa / (RD * virtual_temperature_k)
}

pub fn dry_static_energy(height_m: f64, temperature_k: f64) -> f64 {
    CP_D * temperature_k + G * height_m
}

pub fn moist_static_energy(height_m: f64, temperature_k: f64, specific_humidity_kgkg: f64) -> f64 {
    CP_D * temperature_k + G * height_m + LV * specific_humidity_kgkg
}

pub fn scale_height(temperature_k: f64) -> f64 {
    RD * temperature_k / G
}

pub fn geopotential_to_height(geopotential_m2s2: f64) -> f64 {
    geopotential_m2s2 / G
}

pub fn height_to_geopotential(height_m: f64) -> f64 {
    G * height_m
}

pub fn pressure_to_height_std(pressure_hpa: f64) -> f64 {
    (T0_STD / LAPSE_STD) * (1.0 - (pressure_hpa / P0_STD).powf(1.0 / BARO_EXP))
}

pub fn height_to_pressure_std(height_m: f64) -> f64 {
    P0_STD * (1.0 - LAPSE_STD * height_m / T0_STD).powf(BARO_EXP)
}

pub fn altimeter_to_station_pressure(altimeter_hpa: f64, elevation_m: f64) -> f64 {
    let n = 1.0 / BARO_EXP;
    (altimeter_hpa.powf(n) - P0_STD.powf(n) * LAPSE_STD * elevation_m / T0_STD).powf(1.0 / n) + 0.3
}

pub fn station_to_altimeter_pressure(station_hpa: f64, elevation_m: f64) -> f64 {
    let n = 1.0 / BARO_EXP;
    ((station_hpa - 0.3).powf(n) + P0_STD.powf(n) * LAPSE_STD * elevation_m / T0_STD).powf(1.0 / n)
}

pub fn altimeter_to_sea_level_pressure(
    altimeter_hpa: f64,
    elevation_m: f64,
    temperature_c: f64,
) -> f64 {
    let station_pressure_hpa = altimeter_to_station_pressure(altimeter_hpa, elevation_m);
    let surface_temperature_k = temperature_c + ZEROCNK;
    let mean_temperature_k = surface_temperature_k + 0.5 * LAPSE_STD * elevation_m;
    station_pressure_hpa * (G * elevation_m / (RD * mean_temperature_k)).exp()
}

pub fn sigma_to_pressure(sigma: f64, surface_pressure_hpa: f64, top_pressure_hpa: f64) -> f64 {
    sigma * (surface_pressure_hpa - top_pressure_hpa) + top_pressure_hpa
}

pub fn heat_index(temperature_c: f64, relative_humidity_pct: f64) -> f64 {
    let temperature_f = celsius_to_fahrenheit(temperature_c);
    if temperature_f < 80.0 {
        let simple_f = 0.5
            * (temperature_f + 61.0 + (temperature_f - 68.0) * 1.2 + relative_humidity_pct * 0.094);
        return fahrenheit_to_celsius(simple_f);
    }

    let mut heat_index_f =
        -42.379 + 2.049_015_23 * temperature_f + 10.143_331_27 * relative_humidity_pct
            - 0.224_755_41 * temperature_f * relative_humidity_pct
            - 0.006_837_83 * temperature_f * temperature_f
            - 0.054_817_17 * relative_humidity_pct * relative_humidity_pct
            + 0.001_228_74 * temperature_f * temperature_f * relative_humidity_pct
            + 0.000_852_82 * temperature_f * relative_humidity_pct * relative_humidity_pct
            - 0.000_001_99
                * temperature_f
                * temperature_f
                * relative_humidity_pct
                * relative_humidity_pct;

    if relative_humidity_pct < 13.0 && (80.0..=112.0).contains(&temperature_f) {
        heat_index_f -= ((13.0 - relative_humidity_pct) / 4.0)
            * ((17.0 - (temperature_f - 95.0).abs()) / 17.0).sqrt();
    }

    if relative_humidity_pct > 85.0 && (80.0..=87.0).contains(&temperature_f) {
        heat_index_f += ((relative_humidity_pct - 85.0) / 10.0) * ((87.0 - temperature_f) / 5.0);
    }

    fahrenheit_to_celsius(heat_index_f)
}

pub fn windchill(temperature_c: f64, wind_speed_ms: f64) -> f64 {
    let wind_kmh = wind_speed_ms * 3.6;
    let speed_factor = wind_kmh.powf(0.16);
    (0.6215 + 0.3965 * speed_factor) * temperature_c - 11.37 * speed_factor + 13.12
}

pub fn apparent_temperature(
    temperature_c: f64,
    relative_humidity_pct: f64,
    wind_speed_ms: f64,
) -> f64 {
    let temperature_f = celsius_to_fahrenheit(temperature_c);
    let wind_mph = wind_speed_ms * 2.23694;

    if temperature_f >= 80.0 {
        heat_index(temperature_c, relative_humidity_pct)
    } else if temperature_f <= 50.0 && wind_mph > 3.0 {
        windchill(temperature_c, wind_speed_ms)
    } else {
        temperature_c
    }
}

pub fn moist_air_gas_constant(mixing_ratio_kgkg: f64) -> f64 {
    RD * (1.0 + mixing_ratio_kgkg / 0.622) / (1.0 + mixing_ratio_kgkg)
}

pub fn moist_air_specific_heat_pressure(mixing_ratio_kgkg: f64) -> f64 {
    CP_D * (1.0 + (CP_V / CP_D) * mixing_ratio_kgkg) / (1.0 + mixing_ratio_kgkg)
}

pub fn moist_air_poisson_exponent(mixing_ratio_kgkg: f64) -> f64 {
    moist_air_gas_constant(mixing_ratio_kgkg) / moist_air_specific_heat_pressure(mixing_ratio_kgkg)
}

pub fn water_latent_heat_vaporization(temperature_c: f64) -> f64 {
    2.501e6 - 2370.0 * temperature_c
}

pub fn water_latent_heat_melting(temperature_c: f64) -> f64 {
    3.34e5 + 2106.0 * temperature_c
}

pub fn water_latent_heat_sublimation(temperature_c: f64) -> f64 {
    water_latent_heat_vaporization(temperature_c) + water_latent_heat_melting(temperature_c)
}

pub fn relative_humidity_wet_psychrometric(
    dry_bulb_c: f64,
    wet_bulb_c: f64,
    pressure_hpa: f64,
) -> f64 {
    const A: f64 = 0.000799;
    let es_tw = saturation_vapor_pressure(wet_bulb_c);
    let es_t = saturation_vapor_pressure(dry_bulb_c);
    if es_t <= 0.0 {
        return 0.0;
    }
    let vapor_pressure_hpa = es_tw - A * pressure_hpa * (dry_bulb_c - wet_bulb_c);
    (100.0 * vapor_pressure_hpa / es_t).clamp(0.0, 100.0)
}

pub fn weighted_continuous_average(values: &[f64], weights: &[f64]) -> f64 {
    let n = values.len().min(weights.len());
    if n < 2 {
        return 0.0;
    }
    let span = weights[n - 1] - weights[0];
    if span.abs() < 1e-30 {
        return 0.0;
    }
    let mut integral = 0.0;
    for i in 0..(n - 1) {
        let dw = weights[i + 1] - weights[i];
        let value_avg = 0.5 * (values[i] + values[i + 1]);
        integral += value_avg * dw;
    }
    integral / span
}

pub fn find_intersections(x: &[f64], y1: &[f64], y2: &[f64]) -> Vec<(f64, f64)> {
    let n = x.len().min(y1.len()).min(y2.len());
    if n < 2 {
        return vec![];
    }
    let mut crossings = Vec::new();
    for i in 1..n {
        let d_prev = y1[i - 1] - y2[i - 1];
        let d_curr = y1[i] - y2[i];
        if d_prev * d_curr < 0.0 {
            let frac = d_prev.abs() / (d_prev.abs() + d_curr.abs());
            let x_cross = x[i - 1] + frac * (x[i] - x[i - 1]);
            let y_cross = y1[i - 1] + frac * (y1[i] - y1[i - 1]);
            crossings.push((x_cross, y_cross));
        } else if d_curr.abs() < 1e-15 && d_prev.abs() > 1e-15 {
            crossings.push((x[i], y1[i]));
        }
    }
    crossings
}

pub fn get_layer(
    pressure_hpa: &[f64],
    values: &[f64],
    pressure_bottom_hpa: f64,
    pressure_top_hpa: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = pressure_hpa.len().min(values.len());
    let mut pressure_out = Vec::new();
    let mut values_out = Vec::new();

    for i in 0..n {
        if pressure_hpa[i] <= pressure_bottom_hpa && pressure_hpa[i] >= pressure_top_hpa {
            if pressure_out.is_empty() && i > 0 && pressure_hpa[i - 1] > pressure_bottom_hpa {
                let frac = (pressure_bottom_hpa.ln() - pressure_hpa[i - 1].ln())
                    / (pressure_hpa[i].ln() - pressure_hpa[i - 1].ln());
                let interp = values[i - 1] + frac * (values[i] - values[i - 1]);
                pressure_out.push(pressure_bottom_hpa);
                values_out.push(interp);
            }
            pressure_out.push(pressure_hpa[i]);
            values_out.push(values[i]);
        } else if pressure_hpa[i] < pressure_top_hpa && !pressure_out.is_empty() {
            if i > 0 && pressure_hpa[i - 1] >= pressure_top_hpa {
                let frac = (pressure_top_hpa.ln() - pressure_hpa[i - 1].ln())
                    / (pressure_hpa[i].ln() - pressure_hpa[i - 1].ln());
                let interp = values[i - 1] + frac * (values[i] - values[i - 1]);
                pressure_out.push(pressure_top_hpa);
                values_out.push(interp);
            }
            break;
        }
    }
    (pressure_out, values_out)
}

fn get_layer_interpolated(
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    dewpoint_c: &[f64],
    pressure_bottom_hpa: f64,
    depth_hpa: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let pressure_top_hpa = pressure_bottom_hpa - depth_hpa;
    let (layer_p, layer_t) = get_layer(
        pressure_hpa,
        temperature_c,
        pressure_bottom_hpa,
        pressure_top_hpa,
    );
    let (_, layer_td) = get_layer(
        pressure_hpa,
        dewpoint_c,
        pressure_bottom_hpa,
        pressure_top_hpa,
    );
    (layer_p, layer_t, layer_td)
}

pub fn get_layer_heights(
    pressure_hpa: &[f64],
    height_m: &[f64],
    pressure_bottom_hpa: f64,
    pressure_top_hpa: f64,
) -> (Vec<f64>, Vec<f64>) {
    get_layer(
        pressure_hpa,
        height_m,
        pressure_bottom_hpa,
        pressure_top_hpa,
    )
}

pub fn get_perturbation(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return vec![];
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|value| value - mean).collect()
}

pub fn downdraft_cape(pressure_hpa: &[f64], temperature_c: &[f64], dewpoint_c: &[f64]) -> f64 {
    let n = pressure_hpa
        .len()
        .min(temperature_c.len())
        .min(dewpoint_c.len());
    if n < 3 {
        return 0.0;
    }

    let (layer_p, layer_t, layer_td) =
        get_layer_interpolated(pressure_hpa, temperature_c, dewpoint_c, 700.0, 200.0);
    if layer_p.len() < 2 {
        return 0.0;
    }

    let mut min_idx = 0usize;
    let mut min_theta_e = f64::INFINITY;
    for i in 0..layer_p.len() {
        let theta_e = equivalent_potential_temperature(layer_p[i], layer_t[i], layer_td[i]);
        if theta_e < min_theta_e {
            min_theta_e = theta_e;
            min_idx = i;
        }
    }
    let parcel_start_p = layer_p[min_idx];
    let parcel_start_wb = wet_bulb_temperature(parcel_start_p, layer_t[min_idx], layer_td[min_idx]);

    let mut down_pressure = Vec::new();
    let mut env_temperature = Vec::new();
    let mut env_dewpoint = Vec::new();
    for i in 0..n {
        if pressure_hpa[i] >= parcel_start_p {
            down_pressure.push(pressure_hpa[i]);
            env_temperature.push(temperature_c[i]);
            env_dewpoint.push(dewpoint_c[i]);
        }
    }
    if down_pressure.len() < 2 {
        return 0.0;
    }

    let down_parcel_trace =
        moist_lapse_from_reference(&down_pressure, parcel_start_wb, parcel_start_p);
    let mut integral = 0.0;
    for i in 0..down_pressure.len() - 1 {
        let tv_parcel_lo =
            virtual_temperature(down_parcel_trace[i], down_pressure[i], down_parcel_trace[i]);
        let tv_parcel_hi = virtual_temperature(
            down_parcel_trace[i + 1],
            down_pressure[i + 1],
            down_parcel_trace[i + 1],
        );
        let tv_env_lo = virtual_temperature(env_temperature[i], down_pressure[i], env_dewpoint[i]);
        let tv_env_hi = virtual_temperature(
            env_temperature[i + 1],
            down_pressure[i + 1],
            env_dewpoint[i + 1],
        );
        let diff_lo = tv_env_lo - tv_parcel_lo;
        let diff_hi = tv_env_hi - tv_parcel_hi;
        integral += 0.5 * (diff_lo + diff_hi) * (down_pressure[i + 1].ln() - down_pressure[i].ln());
    }
    -(RD * integral)
}

pub fn reduce_point_density(
    latitudes_deg: &[f64],
    longitudes_deg: &[f64],
    radius_deg: f64,
) -> Vec<bool> {
    let n = latitudes_deg.len().min(longitudes_deg.len());
    let mut keep = vec![true; n];
    let r2 = radius_deg * radius_deg;
    for i in 0..n {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..n {
            if !keep[j] {
                continue;
            }
            let dlat = latitudes_deg[j] - latitudes_deg[i];
            let dlon = longitudes_deg[j] - longitudes_deg[i];
            if dlat * dlat + dlon * dlon < r2 {
                keep[j] = false;
            }
        }
    }
    keep
}

pub fn thickness_hydrostatic(p_bottom_hpa: f64, p_top_hpa: f64, mean_temperature_k: f64) -> f64 {
    (RD * mean_temperature_k / G) * (p_bottom_hpa / p_top_hpa).ln()
}

pub fn precipitable_water(pressure_hpa: &[f64], dewpoint_c: &[f64]) -> f64 {
    let n = pressure_hpa.len().min(dewpoint_c.len());
    if n < 2 {
        return 0.0;
    }
    let mixing_ratio_kgkg: Vec<f64> = pressure_hpa
        .iter()
        .zip(dewpoint_c.iter())
        .take(n)
        .map(|(&pressure, &dewpoint)| {
            let vapor_pressure_hpa = saturation_vapor_pressure(dewpoint);
            (EPSILON * vapor_pressure_hpa / (pressure - vapor_pressure_hpa)).max(0.0)
        })
        .collect();

    let mut pw = 0.0;
    for i in 0..(n - 1) {
        let dp_pa = (pressure_hpa[i] - pressure_hpa[i + 1]) * 100.0;
        let w_avg = 0.5 * (mixing_ratio_kgkg[i] + mixing_ratio_kgkg[i + 1]);
        pw += w_avg * dp_pa;
    }
    pw / G
}

pub fn thickness_hydrostatic_from_relative_humidity(
    pressure_hpa: &[f64],
    temperature_c: &[f64],
    relative_humidity_pct: &[f64],
) -> f64 {
    let n = pressure_hpa
        .len()
        .min(temperature_c.len())
        .min(relative_humidity_pct.len());
    if n < 2 {
        return 0.0;
    }

    let virtual_temperatures_k: Vec<f64> = (0..n)
        .map(|i| {
            let mixing_ratio_gkg = mixing_ratio_from_relative_humidity(
                pressure_hpa[i],
                temperature_c[i],
                relative_humidity_pct[i],
            );
            let mixing_ratio_kgkg = mixing_ratio_gkg / 1000.0;
            let temperature_k = temperature_c[i] + ZEROCNK;
            temperature_k * (1.0 + mixing_ratio_kgkg / EPSILON) / (1.0 + mixing_ratio_kgkg)
        })
        .collect();

    let mut integral = 0.0;
    for i in 0..(n - 1) {
        let dlnp = (pressure_hpa[i + 1] * 100.0).ln() - (pressure_hpa[i] * 100.0).ln();
        let tv_avg = 0.5 * (virtual_temperatures_k[i] + virtual_temperatures_k[i + 1]);
        integral += tv_avg * dlnp;
    }

    -(RD / G) * integral
}

pub fn add_height_to_pressure(pressure_hpa: f64, delta_height_m: f64) -> f64 {
    let height_m = pressure_to_height_std(pressure_hpa);
    height_to_pressure_std(height_m + delta_height_m)
}

pub fn add_pressure_to_height(height_m: f64, delta_pressure_hpa: f64) -> f64 {
    let pressure_hpa = height_to_pressure_std(height_m);
    pressure_to_height_std(pressure_hpa + delta_pressure_hpa)
}

pub fn frost_point(temperature_c: f64, relative_humidity_pct: f64) -> f64 {
    let es_water = saturation_vapor_pressure(temperature_c);
    let vapor_pressure_hpa = (relative_humidity_pct / 100.0) * es_water;
    let ln_ratio = (vapor_pressure_hpa / 6.112).ln();
    272.62 * ln_ratio / (22.46 - ln_ratio)
}

pub fn psychrometric_vapor_pressure(dry_bulb_c: f64, wet_bulb_c: f64, pressure_hpa: f64) -> f64 {
    let es_tw = saturation_vapor_pressure(wet_bulb_c);
    es_tw - 6.6e-4 * pressure_hpa * (dry_bulb_c - wet_bulb_c)
}

pub fn brunt_vaisala_frequency(height_m: &[f64], potential_temperature_k: &[f64]) -> Vec<f64> {
    let n = height_m.len().min(potential_temperature_k.len());
    if n < 2 {
        return vec![0.0; n];
    }

    let mut result = vec![0.0; n];
    for i in 0..n {
        let (dtheta, dz) = centered_difference_pair(height_m, potential_temperature_k, i);
        if dz.abs() < 1e-10 || potential_temperature_k[i].abs() < 1e-10 {
            result[i] = 0.0;
            continue;
        }
        let n_sq = (G / potential_temperature_k[i]) * (dtheta / dz);
        result[i] = if n_sq > 0.0 { n_sq.sqrt() } else { 0.0 };
    }
    result
}

pub fn brunt_vaisala_frequency_squared(
    height_m: &[f64],
    potential_temperature_k: &[f64],
) -> Vec<f64> {
    let n = height_m.len().min(potential_temperature_k.len());
    if n < 2 {
        return vec![0.0; n];
    }

    let mut result = vec![0.0; n];
    for i in 0..n {
        let (dtheta, dz) = centered_difference_pair(height_m, potential_temperature_k, i);
        if dz.abs() < 1e-10 || potential_temperature_k[i].abs() < 1e-10 {
            result[i] = 0.0;
            continue;
        }
        result[i] = (G / potential_temperature_k[i]) * (dtheta / dz);
    }
    result
}

pub fn brunt_vaisala_period(height_m: &[f64], potential_temperature_k: &[f64]) -> Vec<f64> {
    brunt_vaisala_frequency(height_m, potential_temperature_k)
        .into_iter()
        .map(|frequency| {
            if frequency <= 0.0 {
                f64::INFINITY
            } else {
                2.0 * std::f64::consts::PI / frequency
            }
        })
        .collect()
}

pub fn mean_pressure_weighted(pressure_hpa: &[f64], values: &[f64]) -> f64 {
    let n = pressure_hpa.len().min(values.len());
    if n < 2 {
        return values.first().copied().unwrap_or(0.0);
    }

    let mut sum_val = 0.0;
    let mut sum_dp = 0.0;
    for i in 0..(n - 1) {
        let dp = (pressure_hpa[i] - pressure_hpa[i + 1]).abs();
        let avg_val = 0.5 * (values[i] + values[i + 1]);
        sum_val += avg_val * dp;
        sum_dp += dp;
    }
    if sum_dp <= 0.0 {
        values[0]
    } else {
        sum_val / sum_dp
    }
}

pub fn exner_function(pressure_hpa: f64) -> f64 {
    (pressure_hpa / 1000.0).powf(ROCP)
}

pub fn vertical_velocity_pressure(w_ms: f64, pressure_hpa: f64, temperature_c: f64) -> f64 {
    let temperature_k = temperature_c + ZEROCNK;
    let pressure_pa = pressure_hpa * 100.0;
    let density = pressure_pa / (RD * temperature_k);
    -density * G * w_ms
}

pub fn vertical_velocity(omega_pas: f64, pressure_hpa: f64, temperature_c: f64) -> f64 {
    let temperature_k = temperature_c + ZEROCNK;
    let pressure_pa = pressure_hpa * 100.0;
    let density = pressure_pa / (RD * temperature_k);
    -omega_pas / (density * G)
}

pub fn montgomery_streamfunction(
    theta_k: f64,
    pressure_hpa: f64,
    temperature_k: f64,
    height_m: f64,
) -> f64 {
    let _ = theta_k;
    let _ = pressure_hpa;
    CP_D * temperature_k + G * height_m
}

pub fn wind_speed(u: f64, v: f64) -> f64 {
    (u * u + v * v).sqrt()
}

pub fn wind_direction(u: f64, v: f64) -> f64 {
    let direction = 270.0 - v.atan2(u).to_degrees();
    direction.rem_euclid(360.0)
}

pub fn wind_components(speed: f64, direction_deg: f64) -> (f64, f64) {
    let meteorological = direction_deg.to_radians();
    let u = -speed * meteorological.sin();
    let v = -speed * meteorological.cos();
    (u, v)
}

pub fn bulk_shear(
    u_profile_ms: &[f64],
    v_profile_ms: &[f64],
    height_profile_m: &[f64],
    bottom_m: f64,
    top_m: f64,
) -> (f64, f64) {
    assert_eq!(u_profile_ms.len(), v_profile_ms.len());
    assert_eq!(u_profile_ms.len(), height_profile_m.len());
    assert!(u_profile_ms.len() >= 2, "need at least 2 levels");

    let u_bot = interp_at_height(u_profile_ms, height_profile_m, bottom_m).unwrap();
    let v_bot = interp_at_height(v_profile_ms, height_profile_m, bottom_m).unwrap();
    let u_top = interp_at_height(u_profile_ms, height_profile_m, top_m).unwrap();
    let v_top = interp_at_height(v_profile_ms, height_profile_m, top_m).unwrap();
    (u_top - u_bot, v_top - v_bot)
}

pub fn bulk_shear_pressure(
    pressure_profile_hpa: &[f64],
    u_profile_ms: &[f64],
    v_profile_ms: &[f64],
    height_profile_m: &[f64],
    bottom_m: f64,
    top_m: f64,
) -> (f64, f64) {
    assert_eq!(pressure_profile_hpa.len(), u_profile_ms.len());
    assert_eq!(pressure_profile_hpa.len(), v_profile_ms.len());
    assert_eq!(pressure_profile_hpa.len(), height_profile_m.len());
    assert!(pressure_profile_hpa.len() >= 2, "need at least 2 levels");

    let (u_layer, _) = layer_by_height_pressure(
        u_profile_ms,
        pressure_profile_hpa,
        height_profile_m,
        bottom_m,
        top_m,
    )
    .unwrap();
    let (v_layer, _) = layer_by_height_pressure(
        v_profile_ms,
        pressure_profile_hpa,
        height_profile_m,
        bottom_m,
        top_m,
    )
    .unwrap();
    let u_bot = u_layer[0];
    let v_bot = v_layer[0];
    let u_top = *u_layer.last().unwrap();
    let v_top = *v_layer.last().unwrap();
    (u_top - u_bot, v_top - v_bot)
}

pub fn storm_relative_helicity(
    u_profile_ms: &[f64],
    v_profile_ms: &[f64],
    height_profile_m: &[f64],
    depth_m: f64,
    storm_u_ms: f64,
    storm_v_ms: f64,
) -> (f64, f64, f64) {
    let n = u_profile_ms.len();
    assert_eq!(n, v_profile_ms.len());
    assert_eq!(n, height_profile_m.len());
    assert!(n >= 2, "need at least 2 levels");

    let mut heights = Vec::new();
    let mut us = Vec::new();
    let mut vs = Vec::new();
    let h_start = height_profile_m[0];
    let h_end = h_start + depth_m;

    for i in 0..n {
        if height_profile_m[i] >= h_start && height_profile_m[i] <= h_end {
            heights.push(height_profile_m[i]);
            us.push(u_profile_ms[i]);
            vs.push(v_profile_ms[i]);
        }
    }
    if let Some(&last_h) = heights.last() {
        if last_h < h_end {
            if let (Some(u_top), Some(v_top)) = (
                interp_at_height(u_profile_ms, height_profile_m, h_end),
                interp_at_height(v_profile_ms, height_profile_m, h_end),
            ) {
                heights.push(h_end);
                us.push(u_top);
                vs.push(v_top);
            }
        }
    }
    let m = heights.len();
    if m < 2 {
        return (0.0, 0.0, 0.0);
    }

    let mut pos_srh = 0.0;
    let mut neg_srh = 0.0;
    for i in 0..(m - 1) {
        let sru_i = us[i] - storm_u_ms;
        let srv_i = vs[i] - storm_v_ms;
        let sru_ip1 = us[i + 1] - storm_u_ms;
        let srv_ip1 = vs[i + 1] - storm_v_ms;
        let contrib = (sru_ip1 * srv_i) - (sru_i * srv_ip1);
        if contrib > 0.0 {
            pos_srh += contrib;
        } else {
            neg_srh += contrib;
        }
    }
    (pos_srh, neg_srh, pos_srh + neg_srh)
}

pub fn mean_wind(
    u_profile_ms: &[f64],
    v_profile_ms: &[f64],
    height_profile_m: &[f64],
    bottom_m: f64,
    top_m: f64,
) -> (f64, f64) {
    let n = u_profile_ms.len();
    assert_eq!(n, v_profile_ms.len());
    assert_eq!(n, height_profile_m.len());
    assert!(n >= 2, "need at least 2 levels");

    let mut heights = Vec::new();
    let mut us = Vec::new();
    let mut vs = Vec::new();

    heights.push(bottom_m);
    us.push(interp_at_height(u_profile_ms, height_profile_m, bottom_m).unwrap());
    vs.push(interp_at_height(v_profile_ms, height_profile_m, bottom_m).unwrap());
    for i in 0..n {
        if height_profile_m[i] > bottom_m && height_profile_m[i] < top_m {
            heights.push(height_profile_m[i]);
            us.push(u_profile_ms[i]);
            vs.push(v_profile_ms[i]);
        }
    }
    heights.push(top_m);
    us.push(interp_at_height(u_profile_ms, height_profile_m, top_m).unwrap());
    vs.push(interp_at_height(v_profile_ms, height_profile_m, top_m).unwrap());

    let mut sum_u = 0.0;
    let mut sum_v = 0.0;
    let mut total_dz = 0.0;
    for i in 0..(heights.len() - 1) {
        let dz = heights[i + 1] - heights[i];
        sum_u += 0.5 * (us[i] + us[i + 1]) * dz;
        sum_v += 0.5 * (vs[i] + vs[i + 1]) * dz;
        total_dz += dz;
    }
    if total_dz.abs() < 1e-10 {
        (us[0], vs[0])
    } else {
        (sum_u / total_dz, sum_v / total_dz)
    }
}

pub fn bunkers_storm_motion(
    pressure_profile_hpa: &[f64],
    u_profile_ms: &[f64],
    v_profile_ms: &[f64],
    height_profile_m: &[f64],
) -> ((f64, f64), (f64, f64), (f64, f64)) {
    let deviation = 7.5;
    let z_surface = height_profile_m[0];

    let mean_u = pressure_weighted_mean_height(
        u_profile_ms,
        pressure_profile_hpa,
        height_profile_m,
        z_surface,
        z_surface + 6000.0,
    );
    let mean_v = pressure_weighted_mean_height(
        v_profile_ms,
        pressure_profile_hpa,
        height_profile_m,
        z_surface,
        z_surface + 6000.0,
    );

    let u_500m = pressure_weighted_mean_height(
        u_profile_ms,
        pressure_profile_hpa,
        height_profile_m,
        z_surface,
        z_surface + 500.0,
    );
    let v_500m = pressure_weighted_mean_height(
        v_profile_ms,
        pressure_profile_hpa,
        height_profile_m,
        z_surface,
        z_surface + 500.0,
    );
    let u_5500m = pressure_weighted_mean_height(
        u_profile_ms,
        pressure_profile_hpa,
        height_profile_m,
        z_surface + 5500.0,
        z_surface + 6000.0,
    );
    let v_5500m = pressure_weighted_mean_height(
        v_profile_ms,
        pressure_profile_hpa,
        height_profile_m,
        z_surface + 5500.0,
        z_surface + 6000.0,
    );
    let shear_u = u_5500m - u_500m;
    let shear_v = v_5500m - v_500m;
    let shear_mag = (shear_u * shear_u + shear_v * shear_v).sqrt();

    if shear_mag < 1e-10 {
        return ((mean_u, mean_v), (mean_u, mean_v), (mean_u, mean_v));
    }

    let perp_u = shear_v / shear_mag;
    let perp_v = -shear_u / shear_mag;
    let right = (mean_u + deviation * perp_u, mean_v + deviation * perp_v);
    let left = (mean_u - deviation * perp_u, mean_v - deviation * perp_v);
    (right, left, (mean_u, mean_v))
}

pub fn corfidi_storm_motion(
    u_profile_ms: &[f64],
    v_profile_ms: &[f64],
    height_profile_m: &[f64],
    u_850_ms: f64,
    v_850_ms: f64,
) -> ((f64, f64), (f64, f64)) {
    let (mean_u, mean_v) = mean_wind(u_profile_ms, v_profile_ms, height_profile_m, 0.0, 6000.0);
    let prop_u = mean_u - u_850_ms;
    let prop_v = mean_v - v_850_ms;
    let upwind = (prop_u, prop_v);
    let downwind = (mean_u + prop_u, mean_v + prop_v);
    (upwind, downwind)
}

pub fn friction_velocity(u_ms: &[f64], w_ms: &[f64]) -> f64 {
    let n = u_ms.len();
    assert_eq!(n, w_ms.len(), "u and w must have the same length");
    assert!(n >= 2, "need at least 2 samples");

    let n_f = n as f64;
    let mean_u = u_ms.iter().sum::<f64>() / n_f;
    let mean_w = w_ms.iter().sum::<f64>() / n_f;
    let mean_uw = u_ms
        .iter()
        .zip(w_ms.iter())
        .map(|(u, w)| u * w)
        .sum::<f64>()
        / n_f;
    let uw_flux = mean_uw - mean_u * mean_w;
    uw_flux.abs().sqrt()
}

pub fn tke(u_ms: &[f64], v_ms: &[f64], w_ms: &[f64]) -> f64 {
    let n = u_ms.len();
    assert_eq!(n, v_ms.len(), "u and v must have the same length");
    assert_eq!(n, w_ms.len(), "u and w must have the same length");
    assert!(n >= 2, "need at least 2 samples");

    let n_f = n as f64;
    let variance = |arr: &[f64]| {
        let mean = arr.iter().sum::<f64>() / n_f;
        arr.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_f
    };
    0.5 * (variance(u_ms) + variance(v_ms) + variance(w_ms))
}

pub fn gradient_richardson_number(
    height_m: &[f64],
    theta_k: &[f64],
    u_ms: &[f64],
    v_ms: &[f64],
) -> Vec<f64> {
    let n = height_m.len();
    assert_eq!(n, theta_k.len());
    assert_eq!(n, u_ms.len());
    assert_eq!(n, v_ms.len());
    assert!(
        n >= 3,
        "need at least 3 levels for gradient Richardson number"
    );

    let first_deriv = |f: &[f64], x: &[f64]| -> Vec<f64> {
        let m = f.len();
        let mut d = vec![0.0; m];
        let dx_fwd = x[2] - x[0];
        if dx_fwd.abs() > 1e-30 {
            d[0] = (-3.0 * f[0] + 4.0 * f[1] - f[2]) / dx_fwd;
        }
        for i in 1..m - 1 {
            let dx = x[i + 1] - x[i - 1];
            if dx.abs() > 1e-30 {
                d[i] = (f[i + 1] - f[i - 1]) / dx;
            }
        }
        let dx_bwd = x[m - 1] - x[m - 3];
        if dx_bwd.abs() > 1e-30 {
            d[m - 1] = (f[m - 3] - 4.0 * f[m - 2] + 3.0 * f[m - 1]) / dx_bwd;
        }
        d
    };

    let dthetadz = first_deriv(theta_k, height_m);
    let dudz = first_deriv(u_ms, height_m);
    let dvdz = first_deriv(v_ms, height_m);

    let mut ri = vec![0.0; n];
    for i in 0..n {
        let shear_sq = dudz[i].powi(2) + dvdz[i].powi(2);
        if shear_sq.abs() < 1e-30 {
            ri[i] = if dthetadz[i] >= 0.0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
        } else {
            ri[i] = (G / theta_k[i]) * (dthetadz[i] / shear_sq);
        }
    }
    ri
}

pub fn coriolis_parameter(latitude_deg: f64) -> f64 {
    2.0 * OMEGA * latitude_deg.to_radians().sin()
}

pub fn absolute_vorticity(
    u: &Grid2D,
    v: &Grid2D,
    latitudes_deg: &Grid2D,
    dx_m: f64,
    dy_m: f64,
) -> Grid2D {
    assert_eq!(u.nx, v.nx, "u and v grids must share nx");
    assert_eq!(u.ny, v.ny, "u and v grids must share ny");
    assert_eq!(u.nx, latitudes_deg.nx, "u and latitude grids must share nx");
    assert_eq!(u.ny, latitudes_deg.ny, "u and latitude grids must share ny");
    let rel = vorticity_regular(u, v, dx_m, dy_m);
    let mut out = Grid2D::zeros(u.nx, u.ny);
    for y in 0..u.ny {
        for x in 0..u.nx {
            out.set(
                x,
                y,
                rel.get(x, y) + coriolis_parameter(latitudes_deg.get(x, y)),
            );
        }
    }
    out
}

pub fn stretching_deformation(u: &Grid2D, v: &Grid2D, dx_m: f64, dy_m: f64) -> Grid2D {
    assert_eq!(u.nx, v.nx, "u and v grids must share nx");
    assert_eq!(u.ny, v.ny, "u and v grids must share ny");
    let mut out = Grid2D::zeros(u.nx, u.ny);
    for y in 0..u.ny {
        for x in 0..u.nx {
            let dudx = diff_x(u, x, y, dx_m);
            let dvdy = diff_y(v, x, y, dy_m);
            out.set(x, y, dudx - dvdy);
        }
    }
    out
}

pub fn shearing_deformation(u: &Grid2D, v: &Grid2D, dx_m: f64, dy_m: f64) -> Grid2D {
    assert_eq!(u.nx, v.nx, "u and v grids must share nx");
    assert_eq!(u.ny, v.ny, "u and v grids must share ny");
    let mut out = Grid2D::zeros(u.nx, u.ny);
    for y in 0..u.ny {
        for x in 0..u.nx {
            let dvdx = diff_x(v, x, y, dx_m);
            let dudy = diff_y(u, x, y, dy_m);
            out.set(x, y, dvdx + dudy);
        }
    }
    out
}

pub fn total_deformation(u: &Grid2D, v: &Grid2D, dx_m: f64, dy_m: f64) -> Grid2D {
    let st = stretching_deformation(u, v, dx_m, dy_m);
    let sh = shearing_deformation(u, v, dx_m, dy_m);
    let mut out = Grid2D::zeros(u.nx, u.ny);
    for y in 0..u.ny {
        for x in 0..u.nx {
            let s = st.get(x, y);
            let h = sh.get(x, y);
            out.set(x, y, (s * s + h * h).sqrt());
        }
    }
    out
}

pub fn advection(scalar: &Grid2D, u: &Grid2D, v: &Grid2D, dx_m: f64, dy_m: f64) -> Grid2D {
    assert_eq!(scalar.nx, u.nx, "scalar and u grids must share nx");
    assert_eq!(scalar.ny, u.ny, "scalar and u grids must share ny");
    assert_eq!(scalar.nx, v.nx, "scalar and v grids must share nx");
    assert_eq!(scalar.ny, v.ny, "scalar and v grids must share ny");
    let mut out = Grid2D::zeros(scalar.nx, scalar.ny);
    for y in 0..scalar.ny {
        for x in 0..scalar.nx {
            let dsdx = diff_x(scalar, x, y, dx_m);
            let dsdy = diff_y(scalar, x, y, dy_m);
            out.set(x, y, -u.get(x, y) * dsdx - v.get(x, y) * dsdy);
        }
    }
    out
}

pub fn frontogenesis(theta: &Grid2D, u: &Grid2D, v: &Grid2D, dx_m: f64, dy_m: f64) -> Grid2D {
    assert_eq!(theta.nx, u.nx, "theta and u grids must share nx");
    assert_eq!(theta.ny, u.ny, "theta and u grids must share ny");
    assert_eq!(theta.nx, v.nx, "theta and v grids must share nx");
    assert_eq!(theta.ny, v.ny, "theta and v grids must share ny");
    let mut out = Grid2D::zeros(theta.nx, theta.ny);
    for y in 0..theta.ny {
        for x in 0..theta.nx {
            let dtdx = diff_x(theta, x, y, dx_m);
            let dtdy = diff_y(theta, x, y, dy_m);
            let dudx = diff_x(u, x, y, dx_m);
            let dvdy = diff_y(v, x, y, dy_m);
            let dvdx = diff_x(v, x, y, dx_m);
            let dudy = diff_y(u, x, y, dy_m);
            let mag_grad = (dtdx * dtdx + dtdy * dtdy).sqrt();
            let fg = if mag_grad < 1e-20 {
                0.0
            } else {
                -(dtdx * dtdx * dudx + dtdy * dtdy * dvdy + dtdx * dtdy * (dvdx + dudy)) / mag_grad
            };
            out.set(x, y, fg);
        }
    }
    out
}

pub fn geostrophic_wind(
    height_m: &Grid2D,
    latitudes_deg: &Grid2D,
    dx_m: f64,
    dy_m: f64,
) -> (Grid2D, Grid2D) {
    assert_eq!(
        height_m.nx, latitudes_deg.nx,
        "height and latitude grids must share nx"
    );
    assert_eq!(
        height_m.ny, latitudes_deg.ny,
        "height and latitude grids must share ny"
    );
    let mut u_geo = Grid2D::zeros(height_m.nx, height_m.ny);
    let mut v_geo = Grid2D::zeros(height_m.nx, height_m.ny);
    for y in 0..height_m.ny {
        for x in 0..height_m.nx {
            let f = coriolis_parameter(latitudes_deg.get(x, y));
            if f.abs() < 1e-10 {
                u_geo.set(x, y, 0.0);
                v_geo.set(x, y, 0.0);
                continue;
            }
            let dzdx = diff_x(height_m, x, y, dx_m);
            let dzdy = diff_y(height_m, x, y, dy_m);
            let gf = G / f;
            u_geo.set(x, y, -gf * dzdy);
            v_geo.set(x, y, gf * dzdx);
        }
    }
    (u_geo, v_geo)
}

pub fn ageostrophic_wind(
    u: &Grid2D,
    v: &Grid2D,
    u_geo: &Grid2D,
    v_geo: &Grid2D,
) -> (Grid2D, Grid2D) {
    assert_eq!(u.nx, v.nx, "u and v grids must share nx");
    assert_eq!(u.ny, v.ny, "u and v grids must share ny");
    assert_eq!(u.nx, u_geo.nx, "u and u_geo grids must share nx");
    assert_eq!(u.ny, u_geo.ny, "u and u_geo grids must share ny");
    assert_eq!(u.nx, v_geo.nx, "u and v_geo grids must share nx");
    assert_eq!(u.ny, v_geo.ny, "u and v_geo grids must share ny");
    let mut ua = Grid2D::zeros(u.nx, u.ny);
    let mut va = Grid2D::zeros(u.nx, u.ny);
    for y in 0..u.ny {
        for x in 0..u.nx {
            ua.set(x, y, u.get(x, y) - u_geo.get(x, y));
            va.set(x, y, v.get(x, y) - v_geo.get(x, y));
        }
    }
    (ua, va)
}

pub fn q_vector(
    temperature_k: &Grid2D,
    u_geo: &Grid2D,
    v_geo: &Grid2D,
    pressure_hpa: f64,
    dx_m: f64,
    dy_m: f64,
) -> (Grid2D, Grid2D) {
    assert_eq!(
        temperature_k.nx, u_geo.nx,
        "temperature and u_geo grids must share nx"
    );
    assert_eq!(
        temperature_k.ny, u_geo.ny,
        "temperature and u_geo grids must share ny"
    );
    assert_eq!(
        temperature_k.nx, v_geo.nx,
        "temperature and v_geo grids must share nx"
    );
    assert_eq!(
        temperature_k.ny, v_geo.ny,
        "temperature and v_geo grids must share ny"
    );
    let coeff = -RD / (pressure_hpa * 100.0);
    let mut q1 = Grid2D::zeros(temperature_k.nx, temperature_k.ny);
    let mut q2 = Grid2D::zeros(temperature_k.nx, temperature_k.ny);
    for y in 0..temperature_k.ny {
        for x in 0..temperature_k.nx {
            let dtdx = diff_x(temperature_k, x, y, dx_m);
            let dtdy = diff_y(temperature_k, x, y, dy_m);
            let dugdx = diff_x(u_geo, x, y, dx_m);
            let dugdy = diff_y(u_geo, x, y, dy_m);
            let dvgdx = diff_x(v_geo, x, y, dx_m);
            let dvgdy = diff_y(v_geo, x, y, dy_m);
            q1.set(x, y, coeff * (dugdx * dtdx + dvgdx * dtdy));
            q2.set(x, y, coeff * (dugdy * dtdx + dvgdy * dtdy));
        }
    }
    (q1, q2)
}

pub fn curvature_vorticity(u: &Grid2D, v: &Grid2D, dx_m: f64, dy_m: f64) -> Grid2D {
    assert_eq!(u.nx, v.nx, "u and v grids must share nx");
    assert_eq!(u.ny, v.ny, "u and v grids must share ny");
    let mut out = Grid2D::zeros(u.nx, u.ny);
    for y in 0..u.ny {
        for x in 0..u.nx {
            let u0 = u.get(x, y);
            let v0 = v.get(x, y);
            let spd2 = u0 * u0 + v0 * v0;
            if spd2 < 1e-20 {
                out.set(x, y, 0.0);
                continue;
            }
            let dudx = diff_x(u, x, y, dx_m);
            let dudy = diff_y(u, x, y, dy_m);
            let dvdx = diff_x(v, x, y, dx_m);
            let dvdy = diff_y(v, x, y, dy_m);
            let dpsidx = (u0 * dvdx - v0 * dudx) / spd2;
            let dpsidy = (u0 * dvdy - v0 * dudy) / spd2;
            out.set(x, y, u0 * dpsidx + v0 * dpsidy);
        }
    }
    out
}

pub fn shear_vorticity(u: &Grid2D, v: &Grid2D, dx_m: f64, dy_m: f64) -> Grid2D {
    let total = vorticity_regular(u, v, dx_m, dy_m);
    let curvature = curvature_vorticity(u, v, dx_m, dy_m);
    let mut out = Grid2D::zeros(u.nx, u.ny);
    for y in 0..u.ny {
        for x in 0..u.nx {
            out.set(x, y, total.get(x, y) - curvature.get(x, y));
        }
    }
    out
}

pub fn inertial_advective_wind(
    u: &Grid2D,
    v: &Grid2D,
    u_geo: &Grid2D,
    v_geo: &Grid2D,
    dx_m: f64,
    dy_m: f64,
) -> (Grid2D, Grid2D) {
    assert_eq!(u.nx, v.nx, "u and v grids must share nx");
    assert_eq!(u.ny, v.ny, "u and v grids must share ny");
    assert_eq!(u.nx, u_geo.nx, "u and u_geo grids must share nx");
    assert_eq!(u.ny, u_geo.ny, "u and u_geo grids must share ny");
    assert_eq!(u.nx, v_geo.nx, "u and v_geo grids must share nx");
    assert_eq!(u.ny, v_geo.ny, "u and v_geo grids must share ny");
    let mut u_ia = Grid2D::zeros(u.nx, u.ny);
    let mut v_ia = Grid2D::zeros(u.nx, u.ny);
    for y in 0..u.ny {
        for x in 0..u.nx {
            let dugdx = diff_x(u_geo, x, y, dx_m);
            let dugdy = diff_y(u_geo, x, y, dy_m);
            let dvgdx = diff_x(v_geo, x, y, dx_m);
            let dvgdy = diff_y(v_geo, x, y, dy_m);
            let u0 = u.get(x, y);
            let v0 = v.get(x, y);
            u_ia.set(x, y, u0 * dugdx + v0 * dugdy);
            v_ia.set(x, y, u0 * dvgdx + v0 * dvgdy);
        }
    }
    (u_ia, v_ia)
}

pub fn absolute_momentum(u: &Grid2D, latitudes_deg: &Grid2D, y_distances_m: &Grid2D) -> Grid2D {
    assert_eq!(u.nx, latitudes_deg.nx, "u and latitude grids must share nx");
    assert_eq!(u.ny, latitudes_deg.ny, "u and latitude grids must share ny");
    assert_eq!(
        u.nx, y_distances_m.nx,
        "u and y-distance grids must share nx"
    );
    assert_eq!(
        u.ny, y_distances_m.ny,
        "u and y-distance grids must share ny"
    );
    let mut out = Grid2D::zeros(u.nx, u.ny);
    for y in 0..u.ny {
        for x in 0..u.nx {
            let f = coriolis_parameter(latitudes_deg.get(x, y));
            out.set(x, y, u.get(x, y) - f * y_distances_m.get(x, y));
        }
    }
    out
}

pub fn gradient(values: &Grid2D, dx_m: f64, dy_m: f64) -> (Grid2D, Grid2D) {
    (gradient_x(values, dx_m), gradient_y(values, dy_m))
}

pub fn gradient_x(values: &Grid2D, dx_m: f64) -> Grid2D {
    assert!(dx_m > 0.0, "dx_m must be positive");
    let mut out = Grid2D::zeros(values.nx, values.ny);
    for y in 0..values.ny {
        for x in 0..values.nx {
            let value = if values.nx < 2 {
                0.0
            } else if values.nx == 2 {
                (values.get(1, y) - values.get(0, y)) / dx_m
            } else if x == 0 {
                (-3.0 * values.get(0, y) + 4.0 * values.get(1, y) - values.get(2, y)) / (2.0 * dx_m)
            } else if x == values.nx - 1 {
                (3.0 * values.get(values.nx - 1, y) - 4.0 * values.get(values.nx - 2, y)
                    + values.get(values.nx - 3, y))
                    / (2.0 * dx_m)
            } else {
                (values.get(x + 1, y) - values.get(x - 1, y)) / (2.0 * dx_m)
            };
            out.set(x, y, value);
        }
    }
    out
}

pub fn gradient_y(values: &Grid2D, dy_m: f64) -> Grid2D {
    assert!(dy_m > 0.0, "dy_m must be positive");
    let mut out = Grid2D::zeros(values.nx, values.ny);
    for y in 0..values.ny {
        for x in 0..values.nx {
            let value = if values.ny < 2 {
                0.0
            } else if values.ny == 2 {
                (values.get(x, 1) - values.get(x, 0)) / dy_m
            } else if y == 0 {
                (-3.0 * values.get(x, 0) + 4.0 * values.get(x, 1) - values.get(x, 2)) / (2.0 * dy_m)
            } else if y == values.ny - 1 {
                (3.0 * values.get(x, values.ny - 1) - 4.0 * values.get(x, values.ny - 2)
                    + values.get(x, values.ny - 3))
                    / (2.0 * dy_m)
            } else {
                (values.get(x, y + 1) - values.get(x, y - 1)) / (2.0 * dy_m)
            };
            out.set(x, y, value);
        }
    }
    out
}

pub fn first_derivative(values: &Grid2D, axis_spacing_m: f64, axis: usize) -> Grid2D {
    match axis {
        0 => gradient_x(values, axis_spacing_m),
        1 => gradient_y(values, axis_spacing_m),
        _ => panic!("axis must be 0 (x) or 1 (y)"),
    }
}

pub fn second_derivative(values: &Grid2D, axis_spacing_m: f64, axis: usize) -> Grid2D {
    assert!(axis_spacing_m > 0.0, "axis_spacing_m must be positive");
    let mut out = Grid2D::zeros(values.nx, values.ny);
    match axis {
        0 => {
            for y in 0..values.ny {
                for x in 0..values.nx {
                    let value = if values.nx < 3 {
                        0.0
                    } else if x == 0 {
                        (values.get(2, y) - 2.0 * values.get(1, y) + values.get(0, y))
                            / (axis_spacing_m * axis_spacing_m)
                    } else if x == values.nx - 1 {
                        (values.get(values.nx - 1, y) - 2.0 * values.get(values.nx - 2, y)
                            + values.get(values.nx - 3, y))
                            / (axis_spacing_m * axis_spacing_m)
                    } else {
                        (values.get(x + 1, y) - 2.0 * values.get(x, y) + values.get(x - 1, y))
                            / (axis_spacing_m * axis_spacing_m)
                    };
                    out.set(x, y, value);
                }
            }
        }
        1 => {
            for y in 0..values.ny {
                for x in 0..values.nx {
                    let value = if values.ny < 3 {
                        0.0
                    } else if y == 0 {
                        (values.get(x, 2) - 2.0 * values.get(x, 1) + values.get(x, 0))
                            / (axis_spacing_m * axis_spacing_m)
                    } else if y == values.ny - 1 {
                        (values.get(x, values.ny - 1) - 2.0 * values.get(x, values.ny - 2)
                            + values.get(x, values.ny - 3))
                            / (axis_spacing_m * axis_spacing_m)
                    } else {
                        (values.get(x, y + 1) - 2.0 * values.get(x, y) + values.get(x, y - 1))
                            / (axis_spacing_m * axis_spacing_m)
                    };
                    out.set(x, y, value);
                }
            }
        }
        _ => panic!("axis must be 0 (x) or 1 (y)"),
    }
    out
}

pub fn laplacian(values: &Grid2D, dx_m: f64, dy_m: f64) -> Grid2D {
    let d2x = second_derivative(values, dx_m, 0);
    let d2y = second_derivative(values, dy_m, 1);
    let mut out = Grid2D::zeros(values.nx, values.ny);
    for y in 0..values.ny {
        for x in 0..values.nx {
            out.set(x, y, d2x.get(x, y) + d2y.get(x, y));
        }
    }
    out
}

pub fn smooth_gaussian(values: &Grid2D, sigma: f64) -> Grid2D {
    Grid2D::new(
        values.nx,
        values.ny,
        smooth_gaussian_raw(&values.values, values.nx, values.ny, sigma),
    )
}

pub fn smooth_rectangular(values: &Grid2D, size: usize, passes: usize) -> Grid2D {
    Grid2D::new(
        values.nx,
        values.ny,
        smooth_rectangular_raw(&values.values, values.nx, values.ny, size, passes),
    )
}

pub fn smooth_circular(values: &Grid2D, radius: f64, passes: usize) -> Grid2D {
    Grid2D::new(
        values.nx,
        values.ny,
        smooth_circular_raw(&values.values, values.nx, values.ny, radius, passes),
    )
}

pub fn smooth_n_point(values: &Grid2D, n: usize, passes: usize) -> Grid2D {
    let window = if n == 9 {
        Grid2D::new(
            3,
            3,
            vec![
                0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625,
            ],
        )
    } else if n == 5 {
        Grid2D::new(
            3,
            3,
            vec![0.0, 0.125, 0.0, 0.125, 0.5, 0.125, 0.0, 0.125, 0.0],
        )
    } else {
        panic!("n must be 5 or 9, got {n}");
    };
    smooth_window(values, &window, passes, false)
}

pub fn smooth_window(
    values: &Grid2D,
    window: &Grid2D,
    passes: usize,
    normalize_weights: bool,
) -> Grid2D {
    Grid2D::new(
        values.nx,
        values.ny,
        smooth_window_raw(
            &values.values,
            values.nx,
            values.ny,
            &window.values,
            window.nx,
            window.ny,
            passes,
            normalize_weights,
        ),
    )
}

pub fn vector_derivative(
    u: &Grid2D,
    v: &Grid2D,
    dx_m: f64,
    dy_m: f64,
) -> (Grid2D, Grid2D, Grid2D, Grid2D) {
    assert_eq!(u.nx, v.nx, "u and v grids must share nx");
    assert_eq!(u.ny, v.ny, "u and v grids must share ny");
    (
        gradient_x(u, dx_m),
        gradient_y(u, dy_m),
        gradient_x(v, dx_m),
        gradient_y(v, dy_m),
    )
}

pub fn cross_section_components(
    u: &[f64],
    v: &[f64],
    start: (f64, f64),
    end: (f64, f64),
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(u.len(), v.len(), "u and v must have the same length");
    let ((te, tn), (ne, nn)) = unit_vectors_from_cross_section(start, end);
    let mut parallel = Vec::with_capacity(u.len());
    let mut perpendicular = Vec::with_capacity(u.len());
    for i in 0..u.len() {
        parallel.push(u[i] * te + v[i] * tn);
        perpendicular.push(u[i] * ne + v[i] * nn);
    }
    (parallel, perpendicular)
}

pub fn unit_vectors_from_cross_section(
    start: (f64, f64),
    end: (f64, f64),
) -> ((f64, f64), (f64, f64)) {
    let dlat = (end.0 - start.0).to_radians();
    let dlon = (end.1 - start.1).to_radians();
    let mean_lat = ((start.0 + end.0) / 2.0).to_radians();
    let de = dlon * mean_lat.cos();
    let dn = dlat;
    let magnitude = (de * de + dn * dn).sqrt();
    if magnitude < 1e-30 {
        return ((0.0, 0.0), (0.0, 0.0));
    }
    let te = de / magnitude;
    let tn = dn / magnitude;
    ((te, tn), (-tn, te))
}

pub fn tangential_component(u: &[f64], v: &[f64], start: (f64, f64), end: (f64, f64)) -> Vec<f64> {
    assert_eq!(u.len(), v.len(), "u and v must have the same length");
    let ((te, tn), _) = unit_vectors_from_cross_section(start, end);
    u.iter()
        .zip(v.iter())
        .map(|(&ui, &vi)| ui * te + vi * tn)
        .collect()
}

pub fn normal_component(u: &[f64], v: &[f64], start: (f64, f64), end: (f64, f64)) -> Vec<f64> {
    assert_eq!(u.len(), v.len(), "u and v must have the same length");
    let (_, (ne, nn)) = unit_vectors_from_cross_section(start, end);
    u.iter()
        .zip(v.iter())
        .map(|(&ui, &vi)| ui * ne + vi * nn)
        .collect()
}

pub fn advection_3d(
    scalar: &[f64],
    u: &[f64],
    v: &[f64],
    w: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    dx_m: f64,
    dy_m: f64,
    dz_m: f64,
) -> Vec<f64> {
    assert!(dx_m > 0.0, "dx_m must be positive");
    assert!(dy_m > 0.0, "dy_m must be positive");
    assert!(dz_m > 0.0, "dz_m must be positive");
    let nxy = nx * ny;
    let total = nxy * nz;
    assert_eq!(scalar.len(), total, "scalar length mismatch");
    assert_eq!(u.len(), total, "u length mismatch");
    assert_eq!(v.len(), total, "v length mismatch");
    assert_eq!(w.len(), total, "w length mismatch");

    let mut out = vec![0.0; total];
    for k in 0..nz {
        let offset = k * nxy;
        let scalar_slab = Grid2D::new(nx, ny, scalar[offset..offset + nxy].to_vec());
        let dsdx = gradient_x(&scalar_slab, dx_m);
        let dsdy = gradient_y(&scalar_slab, dy_m);
        for ij in 0..nxy {
            let idx = offset + ij;
            let mut tendency = -u[idx] * dsdx.values[ij] - v[idx] * dsdy.values[ij];
            let dsdz = if nz < 2 {
                0.0
            } else if k == 0 {
                (scalar[(k + 1) * nxy + ij] - scalar[k * nxy + ij]) / dz_m
            } else if k == nz - 1 {
                (scalar[k * nxy + ij] - scalar[(k - 1) * nxy + ij]) / dz_m
            } else {
                (scalar[(k + 1) * nxy + ij] - scalar[(k - 1) * nxy + ij]) / (2.0 * dz_m)
            };
            tendency -= w[idx] * dsdz;
            out[idx] = tendency;
        }
    }
    out
}

pub fn kinematic_flux(velocity_component: &Grid2D, scalar: &Grid2D) -> Grid2D {
    assert_eq!(
        velocity_component.nx, scalar.nx,
        "velocity and scalar grids must share nx"
    );
    assert_eq!(
        velocity_component.ny, scalar.ny,
        "velocity and scalar grids must share ny"
    );
    let mut out = Grid2D::zeros(velocity_component.nx, velocity_component.ny);
    for y in 0..velocity_component.ny {
        for x in 0..velocity_component.nx {
            out.set(x, y, velocity_component.get(x, y) * scalar.get(x, y));
        }
    }
    out
}

pub fn geospatial_gradient(
    values: &Grid2D,
    latitudes_deg: &Grid2D,
    longitudes_deg: &Grid2D,
) -> (Grid2D, Grid2D) {
    assert_eq!(
        values.nx, latitudes_deg.nx,
        "values and latitude grids must share nx"
    );
    assert_eq!(
        values.ny, latitudes_deg.ny,
        "values and latitude grids must share ny"
    );
    assert_eq!(
        values.nx, longitudes_deg.nx,
        "values and longitude grids must share nx"
    );
    assert_eq!(
        values.ny, longitudes_deg.ny,
        "values and longitude grids must share ny"
    );

    let (dx_local, dy_local) = lat_lon_grid_deltas(latitudes_deg, longitudes_deg);
    let mut dfdx = Grid2D::zeros(values.nx, values.ny);
    let mut dfdy = Grid2D::zeros(values.nx, values.ny);

    for y in 0..values.ny {
        for x in 0..values.nx {
            let dx = dx_local.get(x, y);
            let dy = dy_local.get(x, y);
            let dfdx_val = if values.nx < 2 || dx.abs() < 1e-10 {
                0.0
            } else if values.nx == 2 {
                (values.get(1, y) - values.get(0, y)) / dx
            } else if x == 0 {
                (-3.0 * values.get(0, y) + 4.0 * values.get(1, y) - values.get(2, y)) / (2.0 * dx)
            } else if x == values.nx - 1 {
                (3.0 * values.get(values.nx - 1, y) - 4.0 * values.get(values.nx - 2, y)
                    + values.get(values.nx - 3, y))
                    / (2.0 * dx)
            } else {
                (values.get(x + 1, y) - values.get(x - 1, y)) / (2.0 * dx)
            };
            let dfdy_val = if values.ny < 2 || dy.abs() < 1e-10 {
                0.0
            } else if values.ny == 2 {
                (values.get(x, 1) - values.get(x, 0)) / dy
            } else if y == 0 {
                (-3.0 * values.get(x, 0) + 4.0 * values.get(x, 1) - values.get(x, 2)) / (2.0 * dy)
            } else if y == values.ny - 1 {
                (3.0 * values.get(x, values.ny - 1) - 4.0 * values.get(x, values.ny - 2)
                    + values.get(x, values.ny - 3))
                    / (2.0 * dy)
            } else {
                (values.get(x, y + 1) - values.get(x, y - 1)) / (2.0 * dy)
            };
            dfdx.set(x, y, dfdx_val);
            dfdy.set(x, y, dfdy_val);
        }
    }

    (dfdx, dfdy)
}

pub fn geospatial_laplacian(
    values: &Grid2D,
    latitudes_deg: &Grid2D,
    longitudes_deg: &Grid2D,
) -> Grid2D {
    assert_eq!(
        values.nx, latitudes_deg.nx,
        "values and latitude grids must share nx"
    );
    assert_eq!(
        values.ny, latitudes_deg.ny,
        "values and latitude grids must share ny"
    );
    assert_eq!(
        values.nx, longitudes_deg.nx,
        "values and longitude grids must share nx"
    );
    assert_eq!(
        values.ny, longitudes_deg.ny,
        "values and longitude grids must share ny"
    );

    let (dx_local, dy_local) = lat_lon_grid_deltas(latitudes_deg, longitudes_deg);
    let mut out = Grid2D::zeros(values.nx, values.ny);
    for y in 0..values.ny {
        for x in 0..values.nx {
            let dx = dx_local.get(x, y);
            let dy = dy_local.get(x, y);
            let d2x = if values.nx < 3 || dx.abs() < 1e-10 {
                0.0
            } else if x == 0 {
                (values.get(2, y) - 2.0 * values.get(1, y) + values.get(0, y)) / (dx * dx)
            } else if x == values.nx - 1 {
                (values.get(values.nx - 1, y) - 2.0 * values.get(values.nx - 2, y)
                    + values.get(values.nx - 3, y))
                    / (dx * dx)
            } else {
                (values.get(x + 1, y) - 2.0 * values.get(x, y) + values.get(x - 1, y)) / (dx * dx)
            };
            let d2y = if values.ny < 3 || dy.abs() < 1e-10 {
                0.0
            } else if y == 0 {
                (values.get(x, 2) - 2.0 * values.get(x, 1) + values.get(x, 0)) / (dy * dy)
            } else if y == values.ny - 1 {
                (values.get(x, values.ny - 1) - 2.0 * values.get(x, values.ny - 2)
                    + values.get(x, values.ny - 3))
                    / (dy * dy)
            } else {
                (values.get(x, y + 1) - 2.0 * values.get(x, y) + values.get(x, y - 1)) / (dy * dy)
            };
            out.set(x, y, d2x + d2y);
        }
    }
    out
}

pub fn divergence(u: &Grid2D, v: &Grid2D, dx_m: f64, dy_m: f64) -> Grid2D {
    assert_eq!(u.nx, v.nx, "u and v grids must share nx");
    assert_eq!(u.ny, v.ny, "u and v grids must share ny");
    let mut out = Grid2D::zeros(u.nx, u.ny);
    for y in 0..u.ny {
        for x in 0..u.nx {
            let dudx = diff_x(u, x, y, dx_m);
            let dvdy = diff_y(v, x, y, dy_m);
            out.set(x, y, dudx + dvdy);
        }
    }
    out
}

pub fn vorticity_regular(u: &Grid2D, v: &Grid2D, dx_m: f64, dy_m: f64) -> Grid2D {
    assert_eq!(u.nx, v.nx, "u and v grids must share nx");
    assert_eq!(u.ny, v.ny, "u and v grids must share ny");
    let mut out = Grid2D::zeros(u.nx, u.ny);
    for y in 0..u.ny {
        for x in 0..u.nx {
            let dvdx = diff_x(v, x, y, dx_m);
            let dudy = diff_y(u, x, y, dy_m);
            out.set(x, y, dvdx - dudy);
        }
    }
    out
}

pub fn divergence_regular(u: &Grid2D, v: &Grid2D, dx_m: f64, dy_m: f64) -> Grid2D {
    divergence(u, v, dx_m, dy_m)
}

pub fn vorticity(u: &Grid2D, v: &Grid2D, dx_m: f64, dy_m: f64) -> Grid2D {
    vorticity_regular(u, v, dx_m, dy_m)
}

pub fn wind_direction_from_uv(u: f64, v: f64) -> f64 {
    wind_direction(u, v)
}

pub fn equivalent_potential_temperature_bolton(
    temperature_c: f64,
    dewpoint_c: f64,
    pressure_hpa: f64,
) -> f64 {
    equivalent_potential_temperature(pressure_hpa, temperature_c, dewpoint_c)
}

pub fn saturation_vapor_pressure_bolton(temperature_c: f64) -> f64 {
    saturation_vapor_pressure(temperature_c)
}

fn wobf(temperature_c: f64) -> f64 {
    let t = temperature_c - 20.0;
    if t <= 0.0 {
        let npol = 1.0
            + t * (-8.841_660_5e-3
                + t * (1.471_414_3e-4
                    + t * (-9.671_989e-7 + t * (-3.260_721_7e-8 + t * -3.859_807_3e-10))));
        15.13 / npol.powi(4)
    } else {
        let ppol = t
            * (4.961_892_2e-7
                + t * (-6.105_936_5e-9
                    + t * (3.940_155_1e-11 + t * (-1.258_812_9e-13 + t * 1.668_828_0e-16))));
        let ppol = 1.0 + t * (3.618_298_9e-3 + t * (-1.360_327_3e-5 + ppol));
        (29.93 / ppol.powi(4)) + (0.96 * t) - 14.8
    }
}

fn satlift(pressure_hpa: f64, thetam_c: f64) -> f64 {
    if pressure_hpa >= 1000.0 {
        return thetam_c;
    }
    let pressure_weight = (pressure_hpa / 1000.0).powf(ROCP);
    let mut temperature_1 = (thetam_c + ZEROCNK) * pressure_weight - ZEROCNK;
    let mut error_1 = wobf(temperature_1) - wobf(thetam_c);
    let mut rate = 1.0;

    for _ in 0..7 {
        if error_1.abs() < 0.001 {
            break;
        }
        let temperature_2 = temperature_1 - error_1 * rate;
        let mut error_2 = (temperature_2 + ZEROCNK) / pressure_weight - ZEROCNK;
        error_2 += wobf(temperature_2) - wobf(error_2) - thetam_c;
        rate = (temperature_2 - temperature_1) / (error_2 - error_1);
        temperature_1 = temperature_2;
        error_1 = error_2;
    }

    temperature_1
}

fn drylift(pressure_hpa: f64, temperature_c: f64, dewpoint_c: f64) -> (f64, f64) {
    let temperature_k = temperature_c + ZEROCNK;
    let specific_humidity = specific_humidity_from_dewpoint(pressure_hpa, dewpoint_c);
    let moist_heat_ratio = moist_air_specific_heat_pressure_specific_humidity(specific_humidity)
        / moist_air_gas_constant_specific_humidity(specific_humidity);
    let spec_heat_diff = CP_L - CP_V_METPY;
    let a = moist_heat_ratio + spec_heat_diff / RV_METPY;
    let b = -(LV + spec_heat_diff * SVP_T0) / (RV_METPY * temperature_k);
    let c = b / a;
    let rh = relative_humidity_from_dewpoint(temperature_c, dewpoint_c) / 100.0;
    let w_minus_one = lambert_w_minus_one(rh.powf(1.0 / a) * c * c.exp());
    let temperature_lcl_k = c / w_minus_one * temperature_k;
    let pressure_lcl_hpa =
        pressure_hpa * (temperature_lcl_k / temperature_k).powf(moist_heat_ratio);
    (pressure_lcl_hpa, temperature_lcl_k - ZEROCNK)
}

fn interp_linear(x: f64, x1: f64, x2: f64, y1: f64, y2: f64) -> f64 {
    if x2 == x1 {
        y1
    } else {
        y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    }
}

fn get_env_at_pres(
    target_pressure_hpa: f64,
    pressure_profile_hpa: &[f64],
    temperature_profile_c: &[f64],
    dewpoint_profile_c: &[f64],
) -> (f64, f64) {
    for i in 0..pressure_profile_hpa.len().saturating_sub(1) {
        if pressure_profile_hpa[i] >= target_pressure_hpa
            && target_pressure_hpa >= pressure_profile_hpa[i + 1]
        {
            let log_p = target_pressure_hpa.ln();
            let log_p1 = pressure_profile_hpa[i].ln();
            let log_p2 = pressure_profile_hpa[i + 1].ln();
            let temperature_interp = interp_linear(
                log_p,
                log_p1,
                log_p2,
                temperature_profile_c[i],
                temperature_profile_c[i + 1],
            );
            let dewpoint_interp = interp_linear(
                log_p,
                log_p1,
                log_p2,
                dewpoint_profile_c[i],
                dewpoint_profile_c[i + 1],
            );
            return (temperature_interp, dewpoint_interp);
        }
    }
    (
        *temperature_profile_c.last().unwrap_or(&0.0),
        *dewpoint_profile_c.last().unwrap_or(&0.0),
    )
}

fn get_height_at_pres(
    target_pressure_hpa: f64,
    pressure_profile_hpa: &[f64],
    height_profile_m: &[f64],
) -> f64 {
    let n = pressure_profile_hpa.len().min(height_profile_m.len());
    if n == 0 {
        return 0.0;
    }
    if target_pressure_hpa >= pressure_profile_hpa[0] {
        return height_profile_m[0];
    }
    if target_pressure_hpa <= pressure_profile_hpa[n - 1] {
        return height_profile_m[n - 1];
    }
    for i in 1..n {
        if pressure_profile_hpa[i] <= target_pressure_hpa {
            let log_p = target_pressure_hpa.ln();
            let log_p1 = pressure_profile_hpa[i - 1].ln();
            let log_p2 = pressure_profile_hpa[i].ln();
            return interp_linear(
                log_p,
                log_p1,
                log_p2,
                height_profile_m[i - 1],
                height_profile_m[i],
            );
        }
    }
    height_profile_m[n - 1]
}

fn lift_parcel_profile(
    pressure_profile_hpa: &[f64],
    temperature_profile_c: &[f64],
    dewpoint_profile_c: &[f64],
) -> Option<(f64, f64, Vec<f64>)> {
    if pressure_profile_hpa.is_empty()
        || temperature_profile_c.is_empty()
        || dewpoint_profile_c.is_empty()
    {
        return None;
    }

    let surface_pressure_hpa = pressure_profile_hpa[0];
    let surface_temperature_c = temperature_profile_c[0];
    let surface_dewpoint_c = dewpoint_profile_c[0];
    let (pressure_lcl_hpa, temperature_lcl_c) = drylift(
        surface_pressure_hpa,
        surface_temperature_c,
        surface_dewpoint_c,
    );
    let theta_k = (temperature_lcl_c + ZEROCNK) * (1000.0 / pressure_lcl_hpa).powf(ROCP);
    let theta_c = theta_k - ZEROCNK;
    let thetam = theta_c - wobf(theta_c) + wobf(temperature_lcl_c);
    let theta_dry_k =
        (surface_temperature_c + ZEROCNK) * (1000.0 / surface_pressure_hpa).powf(ROCP);
    let parcel_mixing_ratio_gkg = mixing_ratio(surface_pressure_hpa, surface_dewpoint_c);

    let mut parcel_tv = Vec::with_capacity(pressure_profile_hpa.len());
    for &pressure in pressure_profile_hpa {
        if pressure > pressure_lcl_hpa {
            let parcel_temperature_k = theta_dry_k * (pressure / 1000.0).powf(ROCP);
            let parcel_temperature_c = parcel_temperature_k - ZEROCNK;
            let tv = (parcel_temperature_c + ZEROCNK)
                * (1.0 + 0.61 * (parcel_mixing_ratio_gkg / 1000.0))
                - ZEROCNK;
            parcel_tv.push(tv);
        } else {
            let parcel_temperature_c = satlift(pressure, thetam);
            parcel_tv.push(virtual_temperature(
                parcel_temperature_c,
                pressure,
                parcel_temperature_c,
            ));
        }
    }

    Some((pressure_lcl_hpa, temperature_lcl_c, parcel_tv))
}

fn moist_lapse_rate(pressure_hpa: f64, temperature_c: f64) -> f64 {
    let temperature_k = temperature_c + ZEROCNK;
    let es = saturation_vapor_pressure(temperature_c);
    let rs = EPSILON * es / (pressure_hpa - es);
    let numerator = (RD * temperature_k + LV * rs) / pressure_hpa;
    let denominator = CP_D + (LV * LV * rs * EPSILON) / (RD * temperature_k * temperature_k);
    numerator / denominator
}

fn interp_log_p(target_pressure_hpa: f64, pressure_profile_hpa: &[f64], values: &[f64]) -> f64 {
    let n = pressure_profile_hpa.len().min(values.len());
    if n == 0 {
        return 0.0;
    }
    if target_pressure_hpa >= pressure_profile_hpa[0] {
        return values[0];
    }
    if target_pressure_hpa <= pressure_profile_hpa[n - 1] {
        return values[n - 1];
    }
    for i in 1..n {
        if pressure_profile_hpa[i] <= target_pressure_hpa {
            let log_p0 = pressure_profile_hpa[i - 1].ln();
            let log_p1 = pressure_profile_hpa[i].ln();
            let log_pt = target_pressure_hpa.ln();
            let frac = (log_pt - log_p0) / (log_p1 - log_p0);
            return values[i - 1] + frac * (values[i] - values[i - 1]);
        }
    }
    values[n - 1]
}

fn interp_at_height(profile: &[f64], heights_m: &[f64], target_h_m: f64) -> Option<f64> {
    debug_assert_eq!(profile.len(), heights_m.len());
    let n = heights_m.len();
    if n == 0 {
        return None;
    }
    if target_h_m <= heights_m[0] {
        return Some(profile[0]);
    }
    if target_h_m >= heights_m[n - 1] {
        return Some(profile[n - 1]);
    }
    for i in 1..n {
        if heights_m[i] >= target_h_m {
            let frac = (target_h_m - heights_m[i - 1]) / (heights_m[i] - heights_m[i - 1]);
            return Some(profile[i - 1] + frac * (profile[i] - profile[i - 1]));
        }
    }
    None
}

fn interp_pressure_at_height_linear(
    pressure_hpa: &[f64],
    heights_m: &[f64],
    target_h_m: f64,
) -> Option<f64> {
    debug_assert_eq!(pressure_hpa.len(), heights_m.len());
    let n = heights_m.len();
    if n == 0 {
        return None;
    }
    if target_h_m <= heights_m[0] {
        return Some(pressure_hpa[0]);
    }
    if target_h_m >= heights_m[n - 1] {
        return Some(pressure_hpa[n - 1]);
    }
    for i in 1..n {
        if heights_m[i] >= target_h_m {
            let frac = (target_h_m - heights_m[i - 1]) / (heights_m[i] - heights_m[i - 1]);
            return Some(pressure_hpa[i - 1] + frac * (pressure_hpa[i] - pressure_hpa[i - 1]));
        }
    }
    None
}

fn layer_by_height_pressure(
    values: &[f64],
    pressure_hpa: &[f64],
    heights_m: &[f64],
    bottom_m: f64,
    top_m: f64,
) -> Option<(Vec<f64>, Vec<f64>)> {
    debug_assert_eq!(values.len(), pressure_hpa.len());
    debug_assert_eq!(values.len(), heights_m.len());
    let n = values.len();
    if n == 0 {
        return None;
    }

    let bottom_pressure_hpa = interp_pressure_at_height_linear(pressure_hpa, heights_m, bottom_m)?;
    let top_pressure_hpa = interp_pressure_at_height_linear(pressure_hpa, heights_m, top_m)?;
    let mut layer_pressure = pressure_hpa
        .iter()
        .copied()
        .filter(|&pressure| pressure <= bottom_pressure_hpa && pressure >= top_pressure_hpa)
        .collect::<Vec<_>>();
    if layer_pressure.is_empty() {
        layer_pressure.push(bottom_pressure_hpa);
        layer_pressure.push(top_pressure_hpa);
    } else {
        if layer_pressure
            .first()
            .is_none_or(|pressure| (pressure - bottom_pressure_hpa).abs() > 1.0e-6)
        {
            layer_pressure.push(bottom_pressure_hpa);
        }
        if layer_pressure
            .last()
            .is_none_or(|pressure| (pressure - top_pressure_hpa).abs() > 1.0e-6)
        {
            layer_pressure.push(top_pressure_hpa);
        }
    }
    layer_pressure.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    layer_pressure.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-6);
    let layer_values = layer_pressure
        .iter()
        .map(|&pressure| interp_log_p(pressure, pressure_hpa, values))
        .collect::<Vec<_>>();
    Some((layer_values, layer_pressure))
}

fn pressure_weighted_mean_height(
    component: &[f64],
    pressure_hpa: &[f64],
    heights_m: &[f64],
    bottom_m: f64,
    top_m: f64,
) -> f64 {
    assert_eq!(component.len(), pressure_hpa.len());
    assert_eq!(component.len(), heights_m.len());
    let (layer_component, layer_pressure) =
        layer_by_height_pressure(component, pressure_hpa, heights_m, bottom_m, top_m).unwrap();

    if layer_component.len() < 2 {
        return layer_component[0];
    }

    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for i in 1..layer_component.len() {
        let dp = layer_pressure[i] - layer_pressure[i - 1];
        numerator += 0.5 * (layer_component[i] + layer_component[i - 1]) * dp;
        denominator += dp;
    }
    if denominator.abs() > 1e-10 {
        numerator / denominator
    } else {
        layer_component[0]
    }
}

fn celsius_to_fahrenheit(temperature_c: f64) -> f64 {
    temperature_c * 9.0 / 5.0 + 32.0
}

fn fahrenheit_to_celsius(temperature_f: f64) -> f64 {
    (temperature_f - 32.0) * 5.0 / 9.0
}

fn svp_liquid_pa(temperature_k: f64) -> f64 {
    let latent_heat = LV_0 - (CP_L - CP_V_METPY) * (temperature_k - SVP_T0);
    let heat_power = (CP_L - CP_V_METPY) / RV_METPY;
    let exp_term = (LV_0 / SVP_T0 - latent_heat / temperature_k) / RV_METPY;
    SAT_PRESSURE_0C * (SVP_T0 / temperature_k).powf(heat_power) * exp_term.exp()
}

fn vappres_sharppy(temperature_c: f64) -> f64 {
    let pol = temperature_c * (1.111_201_8e-17 + (temperature_c * -3.099_457_1e-20));
    let pol = temperature_c * (2.187_442_5e-13 + (temperature_c * (-1.789_232e-15 + pol)));
    let pol = temperature_c * (4.388_418_0e-09 + (temperature_c * (-2.988_388e-11 + pol)));
    let pol = temperature_c * (7.873_616_9e-05 + (temperature_c * (-6.111_796e-07 + pol)));
    let pol = 0.999_996_83 + (temperature_c * (-9.082_695e-03 + pol));
    6.1078 / pol.powi(8)
}

fn centered_difference_pair(x: &[f64], y: &[f64], i: usize) -> (f64, f64) {
    let n = x.len().min(y.len());
    if n < 2 {
        return (0.0, 0.0);
    }
    if i == 0 {
        (y[1] - y[0], x[1] - x[0])
    } else if i == n - 1 {
        (y[n - 1] - y[n - 2], x[n - 1] - x[n - 2])
    } else {
        (y[i + 1] - y[i - 1], x[i + 1] - x[i - 1])
    }
}

fn haversine_distance_m(lat1_deg: f64, lon1_deg: f64, lat2_deg: f64, lon2_deg: f64) -> f64 {
    let earth_radius_m = 6_371_000.0;
    let lat1 = lat1_deg.to_radians();
    let lon1 = lon1_deg.to_radians();
    let lat2 = lat2_deg.to_radians();
    let lon2 = lon2_deg.to_radians();
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * earth_radius_m * a.sqrt().asin()
}

fn wrapped_longitude_delta_deg(lon1_deg: f64, lon2_deg: f64) -> f64 {
    let mut delta = lon2_deg - lon1_deg;
    while delta <= -180.0 {
        delta += 360.0;
    }
    while delta > 180.0 {
        delta -= 360.0;
    }
    delta
}

pub fn lat_lon_grid_deltas(latitudes_deg: &Grid2D, longitudes_deg: &Grid2D) -> (Grid2D, Grid2D) {
    assert_eq!(
        latitudes_deg.nx, longitudes_deg.nx,
        "latitude and longitude grids must share nx"
    );
    assert_eq!(
        latitudes_deg.ny, longitudes_deg.ny,
        "latitude and longitude grids must share ny"
    );
    let nx = latitudes_deg.nx;
    let ny = latitudes_deg.ny;
    let mut dx = Grid2D::zeros(nx, ny);
    let mut dy = Grid2D::zeros(nx, ny);
    let mut dx_edges = vec![0.0; ny.saturating_mul(nx.saturating_sub(1))];
    let mut dy_edges = vec![0.0; ny.saturating_sub(1).saturating_mul(nx)];

    if nx >= 2 {
        for y in 0..ny {
            for x in 0..nx - 1 {
                let distance = haversine_distance_m(
                    latitudes_deg.get(x, y),
                    longitudes_deg.get(x, y),
                    latitudes_deg.get(x + 1, y),
                    longitudes_deg.get(x + 1, y),
                );
                let sign = wrapped_longitude_delta_deg(
                    longitudes_deg.get(x, y),
                    longitudes_deg.get(x + 1, y),
                )
                .signum();
                dx_edges[y * (nx - 1) + x] = distance * sign;
            }
        }

        for y in 0..ny {
            for x in 0..nx {
                let dx_val = if x == 0 {
                    dx_edges[y * (nx - 1)]
                } else if x == nx - 1 {
                    dx_edges[y * (nx - 1) + (nx - 2)]
                } else {
                    (dx_edges[y * (nx - 1) + (x - 1)] + dx_edges[y * (nx - 1) + x]) / 2.0
                };
                dx.set(x, y, dx_val);
            }
        }
    }

    if ny >= 2 {
        for y in 0..ny - 1 {
            for x in 0..nx {
                let distance = haversine_distance_m(
                    latitudes_deg.get(x, y),
                    longitudes_deg.get(x, y),
                    latitudes_deg.get(x, y + 1),
                    longitudes_deg.get(x, y + 1),
                );
                let sign = (latitudes_deg.get(x, y + 1) - latitudes_deg.get(x, y)).signum();
                dy_edges[y * nx + x] = distance * sign;
            }
        }

        for y in 0..ny {
            for x in 0..nx {
                let dy_val = if y == 0 {
                    dy_edges[x]
                } else if y == ny - 1 {
                    dy_edges[(ny - 2) * nx + x]
                } else {
                    (dy_edges[(y - 1) * nx + x] + dy_edges[y * nx + x]) / 2.0
                };
                dy.set(x, y, dy_val);
            }
        }
    }

    (dx, dy)
}

fn flat_idx(x: usize, y: usize, nx: usize) -> usize {
    y * nx + x
}

fn smooth_gaussian_raw(data: &[f64], nx: usize, ny: usize, sigma: f64) -> Vec<f64> {
    let total = nx * ny;
    assert_eq!(data.len(), total, "data length must equal nx * ny");
    assert!(sigma > 0.0, "sigma must be positive, got {sigma}");

    let half = (4.0 * sigma).ceil() as usize;
    let kernel_size = 2 * half + 1;
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut kernel = vec![0.0; kernel_size];
    for k in 0..kernel_size {
        let delta = k as f64 - half as f64;
        kernel[k] = (-delta * delta / two_sigma_sq).exp();
    }

    let mut temp = vec![f64::NAN; total];
    for y in 0..ny {
        for x in 0..nx {
            let mut weight_sum = 0.0;
            let mut value_sum = 0.0;
            for k in 0..kernel_size {
                let xi = x as isize + k as isize - half as isize;
                if !(0..nx as isize).contains(&xi) {
                    continue;
                }
                let value = data[flat_idx(xi as usize, y, nx)];
                if value.is_nan() {
                    continue;
                }
                weight_sum += kernel[k];
                value_sum += kernel[k] * value;
            }
            temp[flat_idx(x, y, nx)] = if weight_sum > 0.0 {
                value_sum / weight_sum
            } else {
                f64::NAN
            };
        }
    }

    let mut out = vec![f64::NAN; total];
    for y in 0..ny {
        for x in 0..nx {
            let mut weight_sum = 0.0;
            let mut value_sum = 0.0;
            for k in 0..kernel_size {
                let yi = y as isize + k as isize - half as isize;
                if !(0..ny as isize).contains(&yi) {
                    continue;
                }
                let value = temp[flat_idx(x, yi as usize, nx)];
                if value.is_nan() {
                    continue;
                }
                weight_sum += kernel[k];
                value_sum += kernel[k] * value;
            }
            out[flat_idx(x, y, nx)] = if weight_sum > 0.0 {
                value_sum / weight_sum
            } else {
                f64::NAN
            };
        }
    }
    out
}

fn smooth_rectangular_raw(
    data: &[f64],
    nx: usize,
    ny: usize,
    size: usize,
    passes: usize,
) -> Vec<f64> {
    let total = nx * ny;
    assert_eq!(data.len(), total, "data length must equal nx * ny");
    assert!(size > 0, "kernel size must be > 0");

    let half = size / 2;
    let padded_nx = nx + 1;
    let mut current = data.to_vec();
    for _ in 0..passes {
        let mut sat_val = vec![0.0; padded_nx * (ny + 1)];
        let mut sat_nan = vec![0u32; padded_nx * (ny + 1)];
        for y in 0..ny {
            for x in 0..nx {
                let value = current[flat_idx(x, y, nx)];
                let is_nan = value.is_nan();
                let py = y + 1;
                let px = x + 1;
                let idx = py * padded_nx + px;
                sat_val[idx] = (if is_nan { 0.0 } else { value })
                    + sat_val[(py - 1) * padded_nx + px]
                    + sat_val[py * padded_nx + (px - 1)]
                    - sat_val[(py - 1) * padded_nx + (px - 1)];
                sat_nan[idx] = (if is_nan { 1 } else { 0 })
                    + sat_nan[(py - 1) * padded_nx + px]
                    + sat_nan[py * padded_nx + (px - 1)]
                    - sat_nan[(py - 1) * padded_nx + (px - 1)];
            }
        }

        let mut out = current.clone();
        for y in half..ny.saturating_sub(half) {
            for x in half..nx.saturating_sub(half) {
                let y1 = y - half;
                let y2 = y + half;
                let x1 = x - half;
                let x2 = x + half;
                let br = (y2 + 1) * padded_nx + (x2 + 1);
                let tr = y1 * padded_nx + (x2 + 1);
                let bl = (y2 + 1) * padded_nx + x1;
                let tl = y1 * padded_nx + x1;
                let nan_count = sat_nan[br] - sat_nan[tr] - sat_nan[bl] + sat_nan[tl];
                out[flat_idx(x, y, nx)] = if nan_count > 0 {
                    f64::NAN
                } else {
                    let sum = sat_val[br] - sat_val[tr] - sat_val[bl] + sat_val[tl];
                    let count = (y2 - y1 + 1) * (x2 - x1 + 1);
                    sum / count as f64
                };
            }
        }
        current = out;
    }
    current
}

fn smooth_circular_raw(data: &[f64], nx: usize, ny: usize, radius: f64, passes: usize) -> Vec<f64> {
    let total = nx * ny;
    assert_eq!(data.len(), total, "data length must equal nx * ny");
    assert!(radius > 0.0, "radius must be positive, got {radius}");

    let half = radius.ceil() as isize;
    let half_usize = half as usize;
    let radius_sq = radius * radius;
    let mut offsets = Vec::new();
    for dy in -half..=half {
        for dx in -half..=half {
            if (dx * dx + dy * dy) as f64 <= radius_sq {
                offsets.push((dx, dy));
            }
        }
    }

    let mut current = data.to_vec();
    for _ in 0..passes {
        let mut out = current.clone();
        for y in half_usize..ny.saturating_sub(half_usize) {
            for x in half_usize..nx.saturating_sub(half_usize) {
                let mut sum = 0.0;
                let mut count = 0usize;
                let mut has_nan = false;
                for &(dx, dy) in &offsets {
                    let xi = (x as isize + dx) as usize;
                    let yi = (y as isize + dy) as usize;
                    let value = current[flat_idx(xi, yi, nx)];
                    if value.is_nan() {
                        has_nan = true;
                        break;
                    }
                    sum += value;
                    count += 1;
                }
                out[flat_idx(x, y, nx)] = if has_nan || count == 0 {
                    f64::NAN
                } else {
                    sum / count as f64
                };
            }
        }
        current = out;
    }
    current
}

fn smooth_window_raw(
    data: &[f64],
    nx: usize,
    ny: usize,
    window: &[f64],
    window_nx: usize,
    window_ny: usize,
    passes: usize,
    normalize_weights: bool,
) -> Vec<f64> {
    let total = nx * ny;
    assert_eq!(data.len(), total, "data length must equal nx * ny");
    assert_eq!(
        window.len(),
        window_nx * window_ny,
        "window length must equal window_nx * window_ny"
    );
    assert!(window_nx > 0, "window_nx must be > 0");
    assert!(window_ny > 0, "window_ny must be > 0");

    let half_x = window_nx / 2;
    let half_y = window_ny / 2;
    let weights = if normalize_weights {
        let weight_sum: f64 = window.iter().sum();
        if weight_sum.abs() > 1e-30 {
            window.iter().map(|&w| w / weight_sum).collect::<Vec<_>>()
        } else {
            window.to_vec()
        }
    } else {
        window.to_vec()
    };

    let mut current = data.to_vec();
    for _ in 0..passes {
        let mut out = current.clone();
        for y in half_y..ny.saturating_sub(half_y) {
            for x in half_x..nx.saturating_sub(half_x) {
                let mut sum = 0.0;
                let mut has_nan = false;
                'kernel: for ky in 0..window_ny {
                    let yi = y + ky - half_y;
                    for kx in 0..window_nx {
                        let xi = x + kx - half_x;
                        let value = current[flat_idx(xi, yi, nx)];
                        if value.is_nan() {
                            has_nan = true;
                            break 'kernel;
                        }
                        sum += weights[flat_idx(kx, ky, window_nx)] * value;
                    }
                }
                out[flat_idx(x, y, nx)] = if has_nan { f64::NAN } else { sum };
            }
        }
        current = out;
    }
    current
}

pub fn angle_to_direction(angle: f64) -> &'static str {
    let normalized = ((angle % 360.0) + 360.0) % 360.0;
    let index = ((normalized + 11.25) / 22.5) as usize % 16;
    DIRECTIONS_16[index]
}

pub fn parse_angle(direction: &str) -> Option<f64> {
    let upper = direction.to_uppercase();
    DIRECTIONS_16
        .iter()
        .position(|&candidate| candidate == upper)
        .map(|index| index as f64 * 22.5)
}

pub fn find_bounding_indices(values: &[f64], target: f64) -> Option<(usize, usize)> {
    if values.len() < 2 {
        return None;
    }
    for i in 0..values.len() - 1 {
        let low = values[i].min(values[i + 1]);
        let high = values[i].max(values[i + 1]);
        if (low..=high).contains(&target) {
            return Some((i, i + 1));
        }
    }
    None
}

pub fn nearest_intersection_idx(x: &[f64], y1: &[f64], y2: &[f64]) -> Option<usize> {
    let n = x.len().min(y1.len()).min(y2.len());
    if n < 2 {
        return None;
    }
    let differences = (0..n).map(|i| y1[i] - y2[i]).collect::<Vec<_>>();
    let mut best = None;
    let mut best_abs = f64::INFINITY;
    for i in 0..n - 1 {
        if differences[i] * differences[i + 1] <= 0.0 {
            let (candidate_idx, abs_value) = if differences[i].abs() <= differences[i + 1].abs() {
                (i, differences[i].abs())
            } else {
                (i + 1, differences[i + 1].abs())
            };
            if abs_value < best_abs {
                best_abs = abs_value;
                best = Some(candidate_idx);
            }
        }
    }
    best
}

pub fn resample_nn_1d(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    let count = xp.len().min(fp.len());
    x.iter()
        .map(|&target| {
            if count == 0 {
                return f64::NAN;
            }
            let mut best_idx = 0usize;
            let mut best_distance = (target - xp[0]).abs();
            for i in 1..count {
                let distance = (target - xp[i]).abs();
                if distance < best_distance {
                    best_distance = distance;
                    best_idx = i;
                }
            }
            fp[best_idx]
        })
        .collect()
}

pub fn find_peaks(data: &[f64], maxima: bool, iqr_ratio: f64) -> Vec<usize> {
    if data.len() < 3 {
        return Vec::new();
    }
    let mut candidates = Vec::new();
    for i in 1..data.len() - 1 {
        let is_extremum = if maxima {
            data[i] > data[i - 1] && data[i] > data[i + 1]
        } else {
            data[i] < data[i - 1] && data[i] < data[i + 1]
        };
        if is_extremum {
            candidates.push(i);
        }
    }
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut sorted = data
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if sorted.is_empty() {
        return Vec::new();
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let len = sorted.len();
    let median = if len % 2 == 0 {
        (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
    } else {
        sorted[len / 2]
    };
    let q1 = sorted[len / 4];
    let q3 = sorted[3 * len / 4];
    let threshold = if maxima {
        median + iqr_ratio * (q3 - q1)
    } else {
        median - iqr_ratio * (q3 - q1)
    };

    candidates
        .into_iter()
        .filter(|&i| {
            if maxima {
                data[i] >= threshold
            } else {
                data[i] <= threshold
            }
        })
        .collect()
}

pub fn peak_persistence(data: &[f64], maxima: bool) -> Vec<(usize, f64)> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    let values = if maxima {
        data.to_vec()
    } else {
        data.iter().map(|&value| -value).collect::<Vec<_>>()
    };
    let mut order = (0..n).collect::<Vec<_>>();
    order.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    fn find(parent: &mut [usize], mut idx: usize) -> usize {
        while parent[idx] != idx {
            parent[idx] = parent[parent[idx]];
            idx = parent[idx];
        }
        idx
    }

    let mut parent = vec![usize::MAX; n];
    let mut birth = vec![0.0; n];
    let mut processed = vec![false; n];
    let mut result = Vec::new();

    for &idx in &order {
        parent[idx] = idx;
        birth[idx] = values[idx];
        processed[idx] = true;

        let mut roots = Vec::new();
        if idx > 0 && processed[idx - 1] {
            roots.push(find(&mut parent, idx - 1));
        }
        if idx + 1 < n && processed[idx + 1] {
            roots.push(find(&mut parent, idx + 1));
        }
        roots.sort_unstable();
        roots.dedup();
        if roots.is_empty() {
            continue;
        }

        roots.push(idx);
        roots.sort_unstable();
        roots.dedup();
        let mut components = roots
            .iter()
            .map(|&root| find(&mut parent, root))
            .collect::<Vec<_>>();
        components.sort_unstable();
        components.dedup();
        if components.len() <= 1 {
            parent[idx] = components[0];
            continue;
        }

        let oldest = *components
            .iter()
            .max_by(|&&a, &&b| {
                birth[a]
                    .partial_cmp(&birth[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        for &component in &components {
            if component != oldest {
                result.push((component, birth[component] - values[idx]));
                parent[component] = oldest;
            }
        }
    }

    let global_root = find(&mut parent, order[0]);
    let persistence = birth[global_root] - values[*order.last().unwrap()];
    result.push((global_root, persistence));
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

pub fn azimuth_range_to_lat_lon(
    azimuths_deg: &[f64],
    ranges_m: &[f64],
    center_lat_deg: f64,
    center_lon_deg: f64,
) -> (Vec<f64>, Vec<f64>) {
    const EARTH_RADIUS_M: f64 = 6_371_228.0;
    let lat0 = center_lat_deg.to_radians();
    let lon0 = center_lon_deg.to_radians();
    let mut lats = Vec::with_capacity(azimuths_deg.len() * ranges_m.len());
    let mut lons = Vec::with_capacity(azimuths_deg.len() * ranges_m.len());
    for &azimuth_deg in azimuths_deg {
        let azimuth = azimuth_deg.to_radians();
        for &range_m in ranges_m {
            let angular_distance = range_m / EARTH_RADIUS_M;
            let lat = (lat0.sin() * angular_distance.cos()
                + lat0.cos() * angular_distance.sin() * azimuth.cos())
            .asin();
            let lon = lon0
                + (azimuth.sin() * angular_distance.sin() * lat0.cos())
                    .atan2(angular_distance.cos() - lat0.sin() * lat.sin());
            lats.push(lat.to_degrees());
            lons.push(lon.to_degrees());
        }
    }
    (lats, lons)
}

pub fn interpolate_1d(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    assert_eq!(xp.len(), fp.len(), "xp and fp must have the same length");
    assert!(!xp.is_empty(), "xp must not be empty");
    let last = xp.len() - 1;
    x.iter()
        .map(|&target| {
            if target < xp[0] || target > xp[last] {
                return f64::NAN;
            }
            if target == xp[0] {
                return fp[0];
            }
            if target == xp[last] {
                return fp[last];
            }
            let mut low = 0usize;
            let mut high = last;
            while high - low > 1 {
                let middle = (low + high) / 2;
                if xp[middle] <= target {
                    low = middle;
                } else {
                    high = middle;
                }
            }
            let fraction = (target - xp[low]) / (xp[high] - xp[low]);
            fp[low] + fraction * (fp[high] - fp[low])
        })
        .collect()
}

pub fn log_interpolate_1d(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    assert_eq!(xp.len(), fp.len(), "xp and fp must have the same length");
    assert!(!xp.is_empty(), "xp must not be empty");
    let log_x = x.iter().map(|&value| value.ln()).collect::<Vec<_>>();
    let (log_xp, sorted_fp) = if xp.len() >= 2 && xp[0] > xp[xp.len() - 1] {
        (
            xp.iter().rev().map(|&value| value.ln()).collect::<Vec<_>>(),
            fp.iter().rev().copied().collect::<Vec<_>>(),
        )
    } else {
        (
            xp.iter().map(|&value| value.ln()).collect::<Vec<_>>(),
            fp.to_vec(),
        )
    };
    interpolate_1d(&log_x, &log_xp, &sorted_fp)
}

pub fn interpolate_nans_1d(values: &mut [f64]) {
    if values.is_empty() {
        return;
    }
    let valid = (0..values.len())
        .filter(|&index| !values[index].is_nan())
        .collect::<Vec<_>>();
    if valid.is_empty() {
        return;
    }
    let first_valid = valid[0];
    let first_value = values[first_valid];
    for value in values.iter_mut().take(first_valid) {
        *value = first_value;
    }
    let last_valid = *valid.last().unwrap();
    let last_value = values[last_valid];
    for value in values.iter_mut().skip(last_valid + 1) {
        *value = last_value;
    }
    for window in valid.windows(2) {
        let low = window[0];
        let high = window[1];
        if high - low > 1 {
            let low_value = values[low];
            let high_value = values[high];
            for index in low + 1..high {
                let fraction = (index - low) as f64 / (high - low) as f64;
                values[index] = low_value + fraction * (high_value - low_value);
            }
        }
    }
}

pub fn interpolate_to_isosurface(
    values_3d: &[f64],
    surface_values: &[f64],
    target: f64,
    levels: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Vec<f64> {
    let nxy = nx * ny;
    assert_eq!(values_3d.len(), nxy * nz, "values_3d length mismatch");
    assert_eq!(
        surface_values.len(),
        nxy * nz,
        "surface_values length mismatch"
    );
    assert_eq!(levels.len(), nz, "levels length mismatch");
    let _ = levels;

    let mut out = vec![f64::NAN; nxy];
    if nz < 2 {
        return out;
    }
    for y in 0..ny {
        for x in 0..nx {
            let ij = flat_idx(x, y, nx);
            for k in 0..nz - 1 {
                let lower = k * nxy + ij;
                let upper = (k + 1) * nxy + ij;
                let surface_lower = surface_values[lower];
                let surface_upper = surface_values[upper];
                if (surface_lower - target) * (surface_upper - target) <= 0.0
                    && (surface_upper - surface_lower).abs() > 1e-30
                {
                    let fraction = (target - surface_lower) / (surface_upper - surface_lower);
                    out[ij] = values_3d[lower] + fraction * (values_3d[upper] - values_3d[lower]);
                    break;
                }
            }
        }
    }
    out
}

pub fn inverse_distance_to_points(
    obs_x: &[f64],
    obs_y: &[f64],
    obs_values: &[f64],
    grid_x: &[f64],
    grid_y: &[f64],
    radius: f64,
    min_neighbors: usize,
    kind: u8,
    kappa: f64,
    gamma: f64,
) -> Vec<f64> {
    assert_eq!(
        obs_x.len(),
        obs_values.len(),
        "obs_x length must match obs_values"
    );
    assert_eq!(
        obs_y.len(),
        obs_values.len(),
        "obs_y length must match obs_values"
    );
    assert_eq!(
        grid_x.len(),
        grid_y.len(),
        "grid_x and grid_y must have the same length"
    );
    let radius_sq = radius * radius;
    grid_x
        .iter()
        .zip(grid_y.iter())
        .map(|(&gx, &gy)| {
            let mut weight_sum = 0.0;
            let mut weighted_value_sum = 0.0;
            let mut count = 0usize;
            for i in 0..obs_values.len() {
                let dx = gx - obs_x[i];
                let dy = gy - obs_y[i];
                let distance_sq = dx * dx + dy * dy;
                if distance_sq > radius_sq {
                    continue;
                }
                if distance_sq < 1e-30 {
                    return obs_values[i];
                }
                let weight = match kind {
                    1 => (-distance_sq / (kappa * gamma)).exp(),
                    2 => (radius_sq - distance_sq) / (radius_sq + distance_sq),
                    _ => 1.0 / distance_sq,
                };
                weight_sum += weight;
                weighted_value_sum += weight * obs_values[i];
                count += 1;
            }
            if count < min_neighbors {
                f64::NAN
            } else {
                weighted_value_sum / weight_sum
            }
        })
        .collect()
}

fn haversine_distance_deg(lat1_deg: f64, lon1_deg: f64, lat2_deg: f64, lon2_deg: f64) -> f64 {
    let dlat = (lat2_deg - lat1_deg).to_radians();
    let dlon = (lon2_deg - lon1_deg).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1_deg.to_radians().cos() * lat2_deg.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    2.0 * a.sqrt().asin().to_degrees()
}

fn natural_neighbor_point_value(
    target_lat: f64,
    target_lon: f64,
    source_lats: &[f64],
    source_lons: &[f64],
    source_values: &[f64],
) -> f64 {
    if source_values.is_empty() {
        return f64::NAN;
    }
    let neighbor_count = 12.min(source_values.len());
    let mut distances = (0..source_values.len())
        .map(|i| {
            (
                haversine_distance_deg(target_lat, target_lon, source_lats[i], source_lons[i]),
                i,
            )
        })
        .collect::<Vec<_>>();
    distances.select_nth_unstable_by(neighbor_count - 1, |a, b| {
        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    let nearest = distances[..neighbor_count]
        .iter()
        .map(|&(distance, _)| distance)
        .fold(f64::INFINITY, f64::min);
    if nearest < 1e-15 {
        let (_, idx) = distances[..neighbor_count]
            .iter()
            .find(|&&(distance, _)| distance < 1e-15)
            .copied()
            .unwrap();
        return source_values[idx];
    }
    let mut weight_sum = 0.0;
    let mut weighted_value_sum = 0.0;
    for &(distance, idx) in &distances[..neighbor_count] {
        let weight = 1.0 / (distance * distance);
        weight_sum += weight;
        weighted_value_sum += weight * source_values[idx];
    }
    weighted_value_sum / weight_sum
}

pub fn natural_neighbor_to_points(
    source_lats: &[f64],
    source_lons: &[f64],
    source_values: &[f64],
    target_lats: &[f64],
    target_lons: &[f64],
) -> Vec<f64> {
    assert_eq!(
        source_lats.len(),
        source_values.len(),
        "source_lats length must match source_values"
    );
    assert_eq!(
        source_lons.len(),
        source_values.len(),
        "source_lons length must match source_values"
    );
    assert_eq!(
        target_lats.len(),
        target_lons.len(),
        "target_lats and target_lons must have the same length"
    );
    target_lats
        .iter()
        .zip(target_lons.iter())
        .map(|(&lat, &lon)| {
            natural_neighbor_point_value(lat, lon, source_lats, source_lons, source_values)
        })
        .collect()
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
    let fraction = (target - grid[low]) / (grid[high] - grid[low]);
    low as f64 + fraction
}

fn bilinear_at(
    data: &[f64],
    offset: usize,
    nx: usize,
    ny: usize,
    fractional_x: f64,
    fractional_y: f64,
) -> f64 {
    let x0 = (fractional_x.floor() as usize).min(nx.saturating_sub(2));
    let y0 = (fractional_y.floor() as usize).min(ny.saturating_sub(2));
    let x1 = (x0 + 1).min(nx - 1);
    let y1 = (y0 + 1).min(ny - 1);
    let dx = fractional_x - x0 as f64;
    let dy = fractional_y - y0 as f64;
    let v00 = data[offset + flat_idx(x0, y0, nx)];
    let v10 = data[offset + flat_idx(x1, y0, nx)];
    let v01 = data[offset + flat_idx(x0, y1, nx)];
    let v11 = data[offset + flat_idx(x1, y1, nx)];
    let top = v00 * (1.0 - dx) + v10 * dx;
    let bottom = v01 * (1.0 - dx) + v11 * dx;
    top * (1.0 - dy) + bottom * dy
}

pub fn interpolate_to_slice(
    values_3d: &[f64],
    levels: &[f64],
    lat_slice: &[f64],
    lon_slice: &[f64],
    source_lats: &[f64],
    source_lons: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Vec<Vec<f64>> {
    assert_eq!(values_3d.len(), nx * ny * nz, "values_3d length mismatch");
    assert_eq!(levels.len(), nz, "levels length mismatch");
    assert_eq!(source_lats.len(), ny, "source_lats length mismatch");
    assert_eq!(source_lons.len(), nx, "source_lons length mismatch");
    assert_eq!(
        lat_slice.len(),
        lon_slice.len(),
        "lat/lon slice length mismatch"
    );

    let _ = levels;
    let nxy = nx * ny;
    let mut result = Vec::with_capacity(lat_slice.len());
    for i in 0..lat_slice.len() {
        let fractional_x = fractional_index(lon_slice[i], source_lons);
        let fractional_y = fractional_index(lat_slice[i], source_lats);
        let mut column = Vec::with_capacity(nz);
        for k in 0..nz {
            column.push(bilinear_at(
                values_3d,
                k * nxy,
                nx,
                ny,
                fractional_x,
                fractional_y,
            ));
        }
        result.push(column);
    }
    result
}

pub fn remove_nan_observations(
    lats: &[f64],
    lons: &[f64],
    values: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    assert_eq!(lats.len(), values.len(), "lats length must match values");
    assert_eq!(lons.len(), values.len(), "lons length must match values");
    let mut kept_lats = Vec::with_capacity(values.len());
    let mut kept_lons = Vec::with_capacity(values.len());
    let mut kept_values = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        if !values[i].is_nan() {
            kept_lats.push(lats[i]);
            kept_lons.push(lons[i]);
            kept_values.push(values[i]);
        }
    }
    (kept_lats, kept_lons, kept_values)
}

pub fn remove_observations_below_value(
    lats: &[f64],
    lons: &[f64],
    values: &[f64],
    threshold: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    assert_eq!(lats.len(), values.len(), "lats length must match values");
    assert_eq!(lons.len(), values.len(), "lons length must match values");
    let mut kept_lats = Vec::with_capacity(values.len());
    let mut kept_lons = Vec::with_capacity(values.len());
    let mut kept_values = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        if values[i] >= threshold {
            kept_lats.push(lats[i]);
            kept_lons.push(lons[i]);
            kept_values.push(values[i]);
        }
    }
    (kept_lats, kept_lons, kept_values)
}

pub fn interpolate_to_points(
    source_lats: &[f64],
    source_lons: &[f64],
    source_values: &[f64],
    target_lats: &[f64],
    target_lons: &[f64],
    interpolation_type: &str,
) -> Vec<f64> {
    match interpolation_type {
        "natural_neighbor" | "nn" | "natural" => natural_neighbor_to_points(
            source_lats,
            source_lons,
            source_values,
            target_lats,
            target_lons,
        ),
        _ => inverse_distance_to_points(
            source_lons,
            source_lats,
            source_values,
            target_lons,
            target_lats,
            10.0,
            1,
            0,
            100_000.0,
            0.2,
        ),
    }
}

pub fn remove_repeat_coordinates(
    lats: &[f64],
    lons: &[f64],
    values: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    use std::collections::HashSet;

    assert_eq!(lats.len(), values.len(), "lats length must match values");
    assert_eq!(lons.len(), values.len(), "lons length must match values");
    let mut seen = HashSet::new();
    let mut kept_lats = Vec::with_capacity(values.len());
    let mut kept_lons = Vec::with_capacity(values.len());
    let mut kept_values = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        let key = (lats[i].to_bits(), lons[i].to_bits());
        if seen.insert(key) {
            kept_lats.push(lats[i]);
            kept_lons.push(lons[i]);
            kept_values.push(values[i]);
        }
    }
    (kept_lats, kept_lons, kept_values)
}

pub fn geodesic(start: (f64, f64), end: (f64, f64), n_points: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(n_points >= 2, "n_points must be at least 2");
    let start_lat = start.0.to_radians();
    let start_lon = start.1.to_radians();
    let end_lat = end.0.to_radians();
    let end_lon = end.1.to_radians();
    let dlat = end_lat - start_lat;
    let dlon = end_lon - start_lon;
    let a =
        (dlat / 2.0).sin().powi(2) + start_lat.cos() * end_lat.cos() * (dlon / 2.0).sin().powi(2);
    let central_angle = 2.0 * a.sqrt().asin();

    let mut lats = Vec::with_capacity(n_points);
    let mut lons = Vec::with_capacity(n_points);
    if central_angle.abs() < 1e-15 {
        for _ in 0..n_points {
            lats.push(start.0);
            lons.push(start.1);
        }
        return (lats, lons);
    }

    for idx in 0..n_points {
        let fraction = idx as f64 / (n_points - 1) as f64;
        let a_coeff = ((1.0 - fraction) * central_angle).sin() / central_angle.sin();
        let b_coeff = (fraction * central_angle).sin() / central_angle.sin();
        let x =
            a_coeff * start_lat.cos() * start_lon.cos() + b_coeff * end_lat.cos() * end_lon.cos();
        let y =
            a_coeff * start_lat.cos() * start_lon.sin() + b_coeff * end_lat.cos() * end_lon.sin();
        let z = a_coeff * start_lat.sin() + b_coeff * end_lat.sin();
        lats.push(z.atan2((x * x + y * y).sqrt()).to_degrees());
        lons.push(y.atan2(x).to_degrees());
    }
    (lats, lons)
}

fn diff_x(grid: &Grid2D, x: usize, y: usize, dx_m: f64) -> f64 {
    if grid.nx == 1 {
        return 0.0;
    }
    if x == 0 {
        (grid.get(x + 1, y) - grid.get(x, y)) / dx_m
    } else if x == grid.nx - 1 {
        (grid.get(x, y) - grid.get(x - 1, y)) / dx_m
    } else {
        (grid.get(x + 1, y) - grid.get(x - 1, y)) / (2.0 * dx_m)
    }
}

fn diff_y(grid: &Grid2D, x: usize, y: usize, dy_m: f64) -> f64 {
    if grid.ny == 1 {
        return 0.0;
    }
    if y == 0 {
        (grid.get(x, y + 1) - grid.get(x, y)) / dy_m
    } else if y == grid.ny - 1 {
        (grid.get(x, y) - grid.get(x, y - 1)) / dy_m
    } else {
        (grid.get(x, y + 1) - grid.get(x, y - 1)) / (2.0 * dy_m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    fn make_unstable_sounding() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            vec![1000.0, 925.0, 850.0, 700.0, 500.0, 400.0, 300.0, 200.0],
            vec![30.0, 22.0, 16.0, 4.0, -15.0, -28.0, -44.0, -60.0],
            vec![22.0, 18.0, 12.0, -2.0, -25.0, -38.0, -54.0, -70.0],
        )
    }

    #[test]
    fn computes_potential_temperature() {
        let theta = potential_temperature(850.0, 20.0);
        assert!(approx(theta, 307.14, 0.3));
    }

    #[test]
    fn computes_relative_humidity_and_dewpoint_roundtrip() {
        let rh = relative_humidity_from_dewpoint(20.0, 10.0);
        let dewpoint = dewpoint_from_relative_humidity(20.0, rh);
        assert!(approx(rh, 52.5, 1.0));
        assert!(approx(dewpoint, 10.0, 0.5));
    }

    #[test]
    fn humidity_conversions_roundtrip() {
        let q = specific_humidity_from_mixing_ratio(0.012);
        let w = mixing_ratio_from_specific_humidity(q);
        assert!(approx(q, 0.011_857_707_5, 1e-8));
        assert!(approx(w, 12.0, 1e-9));
    }

    #[test]
    fn mixing_ratio_and_specific_humidity_paths_agree() {
        let w = mixing_ratio_from_relative_humidity(1000.0, 20.0, 50.0);
        let rh = relative_humidity_from_mixing_ratio(1000.0, 20.0, w);
        let q = specific_humidity_from_dewpoint(1000.0, 10.0);
        let dewpoint = dewpoint_from_specific_humidity(1000.0, q);
        assert!(approx(rh, 50.0, 0.2));
        assert!(approx(dewpoint, 10.0, 0.5));
    }

    #[test]
    fn computes_static_energy_and_density() {
        let dse = dry_static_energy(1500.0, 290.0);
        let mse = moist_static_energy(1500.0, 290.0, 0.010);
        let rho = density(900.0, 15.0, 8.0);
        assert!(mse > dse);
        assert!(rho > 0.9 && rho < 1.3);
    }

    #[test]
    fn computes_standard_atmosphere_roundtrip() {
        let p = height_to_pressure_std(1500.0);
        let h = pressure_to_height_std(p);
        assert!(approx(h, 1500.0, 25.0));
    }

    #[test]
    fn computes_altimeter_roundtrip() {
        let alt = 1013.25;
        let station = altimeter_to_station_pressure(alt, 300.0);
        let alt_back = station_to_altimeter_pressure(station, 300.0);
        assert!(approx(alt_back, alt, 0.5));
    }

    #[test]
    fn computes_apparent_temperature_indices() {
        assert!(heat_index(35.0, 70.0) > 35.0);
        assert!(windchill(-5.0, 10.0) < -5.0);
        assert!(apparent_temperature(5.0, 50.0, 0.5) == 5.0);
    }

    #[test]
    fn computes_moist_air_properties_and_latent_heats() {
        assert!(approx(moist_air_gas_constant(0.0), RD, 1e-6));
        assert!(approx(moist_air_specific_heat_pressure(0.0), CP_D, 1e-6));
        assert!(approx(moist_air_poisson_exponent(0.0), RD / CP_D, 1e-6));
        assert!(approx(water_latent_heat_vaporization(0.0), 2.501e6, 1.0));
        assert!(approx(water_latent_heat_melting(0.0), 3.34e5, 1.0));
        assert!(approx(
            water_latent_heat_sublimation(0.0),
            2.501e6 + 3.34e5,
            2.0
        ));
    }

    #[test]
    fn computes_layer_utilities() {
        let dz = thickness_hydrostatic(1000.0, 500.0, 260.0);
        let dz_rh = thickness_hydrostatic_from_relative_humidity(
            &[1000.0, 900.0, 800.0, 700.0, 600.0, 500.0],
            &[25.0, 18.0, 10.0, 2.0, -8.0, -18.0],
            &[80.0, 70.0, 60.0, 50.0, 40.0, 30.0],
        );
        let avg = weighted_continuous_average(&[5.0, 5.0, 5.0], &[0.0, 1.0, 2.0]);
        let pert = get_perturbation(&[1.0, 2.0, 3.0]);
        assert!(dz > 5000.0 && dz < 6000.0);
        assert!(dz_rh > 5500.0 && dz_rh < 5700.0);
        assert!(approx(avg, 5.0, 1e-12));
        assert_eq!(pert, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn computes_sounding_basics() {
        let (p, t, td) = make_unstable_sounding();
        let (p_lcl, t_lcl) = lcl(1000.0, 25.0, 10.0);
        let tw = wet_bulb_temperature(1000.0, 30.0, 15.0);
        let li = lifted_index(&p, &t, &td);
        assert!(p_lcl < 1000.0 && p_lcl > 500.0);
        assert!(t_lcl < 25.0);
        assert!(tw >= 15.0 && tw <= 30.0);
        assert!(li < 5.0);
    }

    #[test]
    fn computes_profile_levels_and_layer_accessors() {
        let (p, t, td) = make_unstable_sounding();
        if let Some((p_lfc, _)) = lfc(&p, &t, &td) {
            assert!(p_lfc < 1000.0 && p_lfc > 100.0);
        }
        if let Some((p_el, _)) = el(&p, &t, &td) {
            assert!(p_el < 1000.0 && p_el > 100.0);
        }
        if let Some((p_ccl, t_ccl)) = ccl(&p, &t, &td) {
            assert!(p_ccl < 1000.0 && p_ccl > 200.0);
            assert!(t_ccl < 30.0);
        }
        let (p_layer, t_layer) = get_layer(&p, &t, 925.0, 700.0);
        let (p_height, z_layer) = get_layer_heights(
            &p,
            &[0.0, 750.0, 1500.0, 3000.0, 5500.0, 7000.0, 9000.0, 12000.0],
            925.0,
            700.0,
        );
        assert!(!p_layer.is_empty());
        assert_eq!(p_layer.len(), t_layer.len());
        assert_eq!(p_height.len(), z_layer.len());
    }

    #[test]
    fn computes_lapse_profiles_and_parcel_profile() {
        let dry = dry_lapse(&[1000.0, 850.0, 700.0, 500.0], 20.0);
        let moist = moist_lapse(&[1000.0, 900.0, 800.0, 700.0, 600.0, 500.0], 25.0);
        let parcel = parcel_profile(&[1000.0, 925.0, 850.0, 700.0, 500.0, 300.0], 25.0, 18.0);
        assert_eq!(dry.len(), 4);
        assert_eq!(moist.len(), 6);
        assert_eq!(parcel.len(), 6);
        assert!(dry[1] < dry[0]);
        assert!((moist[1] - 21.47).abs() < 0.2);
        for i in 1..parcel.len() {
            assert!(parcel[i] < parcel[i - 1]);
        }
    }

    #[test]
    fn computes_remaining_thermo_wrappers_and_aliases() {
        let theta_w = wet_bulb_potential_temperature(1000.0, 25.0, 15.0);
        let theta_es = saturation_equivalent_potential_temperature(900.0, 20.0);
        let sigma = static_stability(
            &[1000.0, 850.0, 700.0, 500.0],
            &[293.0, 283.0, 270.0, 252.0],
        );
        let ml = mixed_layer(
            &[1000.0, 975.0, 950.0, 925.0, 900.0, 850.0, 700.0],
            &[20.0, 19.0, 18.0, 17.0, 16.0, 14.0, 4.0],
            100.0,
        );
        let (p_ml, t_ml, td_ml) = get_mixed_layer_parcel(
            &[1000.0, 975.0, 950.0, 925.0, 900.0, 850.0, 700.0],
            &[20.0, 19.0, 18.0, 17.0, 16.0, 14.0, 4.0],
            &[16.0, 15.0, 14.0, 13.0, 12.0, 8.0, -2.0],
            100.0,
        );
        let (p_mu, _, _) = get_most_unstable_parcel(
            &[1000.0, 925.0, 850.0, 700.0, 500.0],
            &[30.0, 24.0, 18.0, 6.0, -10.0],
            &[22.0, 20.0, 16.0, -2.0, -20.0],
            300.0,
        );
        let (cape_sb, cin_sb) = surface_based_cape_cin(
            &[1000.0, 925.0, 850.0, 700.0, 500.0, 300.0],
            &[30.0, 25.0, 20.0, 8.0, -10.0, -35.0],
            &[22.0, 18.0, 14.0, 0.0, -20.0, -45.0],
        );
        let (cape_ml, cin_ml) = mixed_layer_cape_cin(
            &[1000.0, 925.0, 850.0, 700.0, 500.0, 300.0],
            &[30.0, 25.0, 20.0, 8.0, -10.0, -35.0],
            &[22.0, 18.0, 14.0, 0.0, -20.0, -45.0],
            100.0,
        );
        let (cape_mu, cin_mu) = most_unstable_cape_cin(
            &[1000.0, 925.0, 850.0, 700.0, 500.0, 300.0],
            &[30.0, 25.0, 20.0, 8.0, -10.0, -35.0],
            &[22.0, 18.0, 14.0, 0.0, -20.0, -45.0],
        );
        let mu_pressures = [1000.0, 925.0, 850.0, 700.0, 500.0, 300.0];
        let mu_temperatures = [30.0, 25.0, 20.0, 8.0, -10.0, -35.0];
        let mu_dewpoints = [22.0, 18.0, 14.0, 0.0, -20.0, -45.0];
        let mu_start_idx = mu_pressures
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (*a - p_mu)
                    .abs()
                    .partial_cmp(&(*b - p_mu).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let (mu_p_aug, mu_t_aug, mu_td_aug, mu_profile) = metpy_parcel_profile_with_lcl(
            &mu_pressures[mu_start_idx..],
            &mu_temperatures[mu_start_idx..],
            &mu_dewpoints[mu_start_idx..],
        );
        let (cape_mu_ref, cin_mu_ref) =
            metpy_cape_cin_from_profile(&mu_p_aug, &mu_t_aug, &mu_td_aug, &mu_profile);
        let (cape_core, cin_core, h_lcl, h_lfc) = cape_cin(
            &[925.0, 850.0, 700.0, 500.0, 300.0],
            &[25.0, 20.0, 8.0, -10.0, -35.0],
            &[18.0, 14.0, 0.0, -20.0, -45.0],
            &[700.0, 1500.0, 3000.0, 5600.0, 9000.0],
            1000.0,
            30.0,
            22.0,
            "sb",
            100.0,
            300.0,
            Some(6000.0),
        );
        let isentropic = isentropic_interpolation(
            &[305.0],
            &[1000.0, 900.0],
            &[300.0, 302.0],
            &[&[10.0, 20.0]],
            1,
            1,
            2,
        );
        let (p_aug, t_aug) =
            parcel_profile_with_lcl(&[1000.0, 900.0, 800.0, 700.0, 500.0], 25.0, 15.0);
        let e_wet = psychrometric_vapor_pressure_wet(20.0, 20.0, 1000.0);
        let e_base = psychrometric_vapor_pressure(20.0, 20.0, 1000.0);
        let (p_alias, _, _) = mixed_parcel(
            &[1000.0, 975.0, 950.0, 925.0, 900.0, 850.0, 700.0],
            &[20.0, 19.0, 18.0, 17.0, 16.0, 14.0, 4.0],
            &[16.0, 15.0, 14.0, 13.0, 12.0, 8.0, -2.0],
            100.0,
        );
        let (p_alias_mu, _, _) = most_unstable_parcel(
            &[1000.0, 925.0, 850.0, 700.0, 500.0],
            &[30.0, 24.0, 18.0, 6.0, -10.0],
            &[22.0, 20.0, 16.0, -2.0, -20.0],
            300.0,
        );

        assert!(theta_w > 270.0 && theta_w < 310.0);
        assert!(theta_es > 300.0);
        assert_eq!(sigma.len(), 4);
        assert!(ml > 15.0 && ml < 20.0);
        assert!(p_ml > 900.0 && p_ml <= 1000.0);
        assert!(t_ml.is_finite() && td_ml.is_finite());
        assert!(p_mu <= 1000.0 && p_mu >= 700.0);
        assert!(cape_sb >= 0.0 && cin_sb <= 0.0);
        assert!(cape_ml >= 0.0 && cin_ml <= 0.0);
        assert!(cape_mu >= 0.0 && cin_mu <= 0.0);
        assert!((cape_mu - cape_mu_ref).abs() < 1.0e-6);
        assert!((cin_mu - cin_mu_ref).abs() < 1.0e-6);
        assert!(cape_core >= 0.0 && cin_core <= 0.0);
        assert!(h_lcl.is_finite() && h_lcl >= 0.0);
        assert!(h_lfc.is_finite() || h_lfc.is_nan());
        assert_eq!(isentropic.len(), 3);
        assert!(isentropic[0][0].is_finite());
        assert!(isentropic[1][0].is_finite());
        assert!(isentropic[2][0].is_finite());
        assert!(p_aug.len() >= 5 && p_aug.len() == t_aug.len());
        assert!(approx(e_wet, e_base, 1e-12));
        assert!(approx(p_alias, p_ml, 1e-9));
        assert!(approx(p_alias_mu, p_mu, 1e-9));
    }

    #[test]
    fn computes_instability_indices() {
        let (p, t, td) = make_unstable_sounding();
        let si = showalter_index(&p, &t, &td);
        let ki = k_index(20.0, 14.0, 10.0, 2.0, -10.0);
        let vt = vertical_totals(20.0, -10.0);
        let ct = cross_totals(14.0, -10.0);
        let tt = total_totals(20.0, 14.0, -10.0);
        let sweat = sweat_index(20.0, 14.0, -10.0, 200.0, 250.0, 25.0, 40.0);
        assert!(si.is_finite());
        assert!(approx(ki, 36.0, 1e-10));
        assert!(approx(vt, 30.0, 1e-10));
        assert!(approx(ct, 24.0, 1e-10));
        assert!(approx(tt, 54.0, 1e-10));
        assert!(sweat > 0.0);
    }

    #[test]
    fn handles_surface_parcel_lcl_near_existing_pressure_level() {
        let pressure = [
            995.52975, 975.0, 950.0, 925.0, 900.0, 850.0, 800.0, 750.0, 700.0, 650.0, 600.0, 550.0,
            500.0, 450.0, 400.0, 350.0, 300.0, 250.0, 200.0, 150.0, 100.0,
        ];
        let temperature = [
            -1.7726806640624773,
            -3.0645117187499977,
            -5.070781249999982,
            -7.114765624999961,
            -9.190761718749968,
            -12.822714843749964,
            -14.152714843749948,
            -16.872695312499957,
            -20.180292968749967,
            -24.033320312499967,
            -27.776582031249973,
            -30.480605468749985,
            -31.95001953124998,
            -37.22660156249998,
            -38.59134765624998,
            -38.581484374999974,
            -40.84871093749996,
            -41.58328124999997,
            -44.55666015624996,
            -47.953749999999985,
            -51.446425781249985,
        ];
        let dewpoint = [
            -8.188975641396661,
            -9.476593719635849,
            -10.01752160374095,
            -10.443761579165505,
            -10.870974959228421,
            -13.488477583874483,
            -16.98910810544761,
            -21.507594863376998,
            -24.66680994817949,
            -26.703386568821514,
            -32.45082502992388,
            -45.47301474426294,
            -42.72699317592646,
            -44.25431165441229,
            -59.75310683290039,
            -68.65219309292296,
            -69.7988824246584,
            -74.81189583656328,
            -80.84056643889447,
            -83.10349278095686,
            -85.44179616530126,
        ];

        let (pressure_aug, parcel_profile) =
            parcel_profile_with_lcl(&pressure, temperature[0], dewpoint[0]);

        assert_eq!(pressure_aug.len(), pressure.len());
        assert_eq!(parcel_profile.len(), pressure_aug.len());
        assert!(approx(parcel_profile[4], -9.482_848_625_622_27, 1.0e-6));
        assert!(approx(parcel_profile[5], -12.807_787_532_874_032, 1.0e-6));

        let (cape, cin) = surface_based_cape_cin(&pressure, &temperature, &dewpoint);
        assert!(cape >= 0.0 && cape < 0.1);
        assert!(cin < -8.9 && cin > -9.1);
    }

    #[test]
    fn computes_intersections_and_dcape() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y1 = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y2 = vec![4.0, 3.0, 2.0, 1.0, 0.0];
        let crossings = find_intersections(&x, &y1, &y2);
        let dcape = downdraft_cape(
            &[1000.0, 925.0, 850.0, 700.0, 500.0, 400.0, 300.0],
            &[30.0, 22.0, 16.0, 4.0, -15.0, -28.0, -44.0],
            &[22.0, 18.0, 12.0, -2.0, -25.0, -38.0, -54.0],
        );
        assert_eq!(crossings.len(), 1);
        assert!((crossings[0].0 - 2.0).abs() < 0.01);
        assert!((crossings[0].1 - 2.0).abs() < 0.01);
        assert!(dcape >= 0.0);
    }

    #[test]
    fn computes_column_integrals_and_pressure_weighting() {
        let pw = precipitable_water(
            &[1000.0, 925.0, 850.0, 700.0, 500.0],
            &[20.0, 15.0, 10.0, 0.0, -20.0],
        );
        let mean = mean_pressure_weighted(&[1000.0, 900.0, 800.0], &[10.0, 8.0, 6.0]);
        assert!(pw > 0.0);
        assert!(approx(mean, 8.0, 1e-9));
    }

    #[test]
    fn computes_vertical_motion_helpers() {
        let omega = vertical_velocity_pressure(1.0, 700.0, 0.0);
        let w = vertical_velocity(omega, 700.0, 0.0);
        let psi = montgomery_streamfunction(320.0, 700.0, 275.0, 3000.0);
        assert!(omega < 0.0);
        assert!(approx(w, 1.0, 1e-9));
        assert!(psi > 0.0);
        assert!(approx(exner_function(1000.0), 1.0, 1e-12));
    }

    #[test]
    fn computes_brunt_vaisala_metrics() {
        let z = [0.0, 1000.0, 2000.0, 3000.0];
        let theta = [300.0, 303.0, 306.0, 309.0];
        let n = brunt_vaisala_frequency(&z, &theta);
        let n2 = brunt_vaisala_frequency_squared(&z, &theta);
        let period = brunt_vaisala_period(&z, &theta);
        assert!(n[1] > 0.0);
        assert!(n2[1] > 0.0);
        assert!(period[1].is_finite());
    }

    #[test]
    fn computes_wind_metrics() {
        assert!(approx(wind_speed(3.0, 4.0), 5.0, 1e-9));
        assert!(approx(
            wind_direction(0.0, -10.0).rem_euclid(360.0),
            0.0,
            1e-9
        ));
        let (u, v) = wind_components(10.0, 180.0);
        assert!(u.abs() < 1e-9);
        assert!(approx(v, 10.0, 1e-9));
    }

    fn linear_profile() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            vec![0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
            vec![0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0],
            vec![0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0],
            vec![1000.0, 900.0, 800.0, 700.0, 600.0, 500.0, 400.0],
        )
    }

    #[test]
    fn computes_wind_profile_diagnostics() {
        let (us, vs, hgts, ps) = linear_profile();

        let (du_full, dv_full) = bulk_shear(&us, &vs, &hgts, 0.0, 6000.0);
        let (du_sub, dv_sub) = bulk_shear(&us, &vs, &hgts, 0.0, 1000.0);
        let (du_interp, dv_interp) = bulk_shear(&us, &vs, &hgts, 250.0, 750.0);
        assert!(approx(du_full, 30.0, 1e-10));
        assert!(approx(dv_full, 15.0, 1e-10));
        assert!(approx(du_sub, 5.0, 1e-10));
        assert!(approx(dv_sub, 2.5, 1e-10));
        assert!(approx(du_interp, 2.5, 1e-10));
        assert!(approx(dv_interp, 1.25, 1e-10));

        let (mean_u, mean_v) = mean_wind(&us, &vs, &hgts, 0.0, 6000.0);
        let (mean_u_sub, mean_v_sub) = mean_wind(&us, &vs, &hgts, 0.0, 1000.0);
        assert!(approx(mean_u, 15.0, 1e-10));
        assert!(approx(mean_v, 7.5, 1e-10));
        assert!(approx(mean_u_sub, 2.5, 1e-10));
        assert!(approx(mean_v_sub, 1.25, 1e-10));

        let heights = vec![0.0, 1000.0, 2000.0, 3000.0];
        let unidir_u = vec![0.0, 10.0, 20.0, 30.0];
        let unidir_v = vec![0.0, 0.0, 0.0, 0.0];
        let (pos0, neg0, total0) =
            storm_relative_helicity(&unidir_u, &unidir_v, &heights, 3000.0, 15.0, 0.0);
        assert!(approx(pos0, 0.0, 1e-10));
        assert!(approx(neg0, 0.0, 1e-10));
        assert!(approx(total0, 0.0, 1e-10));

        let cw_u = vec![0.0, 10.0, 10.0, 0.0];
        let cw_v = vec![0.0, 0.0, 10.0, 10.0];
        let (_pos_cw, neg_cw, total_cw) =
            storm_relative_helicity(&cw_u, &cw_v, &heights, 3000.0, 0.0, 0.0);
        assert!(total_cw < 0.0);
        assert!(neg_cw < 0.0);

        let mixed_heights = vec![0.0, 500.0, 1000.0, 1500.0, 2000.0];
        let mixed_u = vec![0.0, 5.0, 10.0, 5.0, 0.0];
        let mixed_v = vec![0.0, 5.0, 0.0, -5.0, 0.0];
        let (pos_mix, neg_mix, total_mix) =
            storm_relative_helicity(&mixed_u, &mixed_v, &mixed_heights, 2000.0, 3.0, 1.0);
        assert!(approx(pos_mix + neg_mix, total_mix, 1e-10));

        let (right, left, mean_motion) = bunkers_storm_motion(&ps, &us, &vs, &hgts);
        assert!((mean_motion.0 - mean_u).abs() < 2.0);
        assert!((mean_motion.1 - mean_v).abs() < 1.0);
        let right_dev =
            ((right.0 - mean_motion.0).powi(2) + (right.1 - mean_motion.1).powi(2)).sqrt();
        let left_dev = ((left.0 - mean_motion.0).powi(2) + (left.1 - mean_motion.1).powi(2)).sqrt();
        assert!(approx(right_dev, 7.5, 1e-10));
        assert!(approx(left_dev, 7.5, 1e-10));
        assert!(approx((right.0 + left.0) / 2.0, mean_motion.0, 1e-10));
        assert!(approx((right.1 + left.1) / 2.0, mean_motion.1, 1e-10));

        let (upwind, downwind) = corfidi_storm_motion(&us, &vs, &hgts, 5.0, 2.0);
        assert!(approx(upwind.0, mean_u - 5.0, 1e-10));
        assert!(approx(upwind.1, mean_v - 2.0, 1e-10));
        assert!(approx(downwind.0, mean_u + upwind.0, 1e-10));
        assert!(approx(downwind.1, mean_v + upwind.1, 1e-10));

        let u_star = friction_velocity(&[1.0, -1.0, 1.0, -1.0, 1.0], &[0.5, -0.5, 0.5, -0.5, 0.5]);
        let u_star_zero_mean = friction_velocity(&[1.0, -1.0, 2.0, -2.0], &[0.5, -0.5, 1.0, -1.0]);
        let u_star_uncorr = friction_velocity(&[1.0, -1.0, 1.0, -1.0], &[1.0, 1.0, -1.0, -1.0]);
        assert!(approx(u_star, 0.692_820_323_0, 1e-8));
        assert!(approx(u_star_zero_mean, 1.25_f64.sqrt(), 1e-10));
        assert!(approx(u_star_uncorr, 0.0, 1e-10));

        let e_simple = tke(
            &[1.0, -1.0, 1.0, -1.0],
            &[2.0, -2.0, 2.0, -2.0],
            &[0.5, -0.5, 0.5, -0.5],
        );
        let e_zero = tke(
            &[5.0, 5.0, 5.0, 5.0],
            &[3.0, 3.0, 3.0, 3.0],
            &[0.0, 0.0, 0.0, 0.0],
        );
        let e_equal = tke(&[1.0, -1.0], &[1.0, -1.0], &[1.0, -1.0]);
        assert!(approx(e_simple, 2.625, 1e-10));
        assert!(approx(e_zero, 0.0, 1e-10));
        assert!(approx(e_equal, 1.5, 1e-10));

        let ri = gradient_richardson_number(
            &[0.0, 100.0, 200.0, 300.0, 400.0],
            &[300.0, 301.0, 302.5, 304.5, 307.0],
            &[2.0, 5.0, 8.0, 10.0, 12.0],
            &[1.0, 2.0, 3.5, 5.0, 6.0],
        );
        let expected = [
            0.256_383_01,
            0.385_564_88,
            0.667_443_36,
            1.302_704_38,
            1.925_360_76,
        ];
        assert_eq!(ri.len(), expected.len());
        for (actual, want) in ri.iter().zip(expected.iter()) {
            assert!(
                approx(*actual, *want, 1e-6),
                "Ri = {actual}, expected {want}"
            );
        }

        let keep = reduce_point_density(&[0.0, 0.05, 1.0], &[0.0, 0.04, 1.0], 0.1);
        assert_eq!(keep, vec![true, false, true]);
    }

    #[test]
    fn computes_regular_grid_kinematics_suite() {
        let f_45 = coriolis_parameter(45.0);
        let expected_f_45 = 2.0 * 7.292_115_9e-5 * 45.0_f64.to_radians().sin();
        assert!(approx(f_45, expected_f_45, 1e-12));
        assert!(coriolis_parameter(0.0).abs() < 1e-15);

        let nx = 5;
        let ny = 5;
        let mut u_rot = Grid2D::zeros(nx, ny);
        let mut v_rot = Grid2D::zeros(nx, ny);
        let lats = Grid2D::new(nx, ny, vec![45.0; nx * ny]);
        for y in 0..ny {
            for x in 0..nx {
                u_rot.set(x, y, -(y as f64));
                v_rot.set(x, y, x as f64);
            }
        }
        let avor = absolute_vorticity(&u_rot, &v_rot, &lats, 1.0, 1.0);
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                assert!(approx(avor.get(x, y), 2.0 + f_45, 1e-10));
            }
        }

        let mut u_def = Grid2D::zeros(nx, ny);
        let mut v_def = Grid2D::zeros(nx, ny);
        for y in 0..ny {
            for x in 0..nx {
                u_def.set(x, y, x as f64);
                v_def.set(x, y, -(y as f64));
            }
        }
        let st = stretching_deformation(&u_def, &v_def, 1.0, 1.0);
        let sh = shearing_deformation(&u_def, &v_def, 1.0, 1.0);
        let td = total_deformation(&u_def, &v_def, 1.0, 1.0);
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                assert!(approx(st.get(x, y), 2.0, 1e-10));
                assert!(approx(sh.get(x, y), 0.0, 1e-10));
                assert!(approx(td.get(x, y), 2.0, 1e-10));
            }
        }

        let mut scalar = Grid2D::zeros(nx, 3);
        let u_adv = Grid2D::new(nx, 3, vec![1.0; nx * 3]);
        let v_adv = Grid2D::new(nx, 3, vec![0.0; nx * 3]);
        for y in 0..3 {
            for x in 0..nx {
                scalar.set(x, y, x as f64);
            }
        }
        let adv = advection(&scalar, &u_adv, &v_adv, 1.0, 1.0);
        for y in 0..3 {
            for x in 1..nx - 1 {
                assert!(approx(adv.get(x, y), -1.0, 1e-10));
            }
        }

        let theta = Grid2D::new(nx, ny, vec![300.0; nx * ny]);
        let u_const = Grid2D::new(nx, ny, vec![10.0; nx * ny]);
        let v_const = Grid2D::new(nx, ny, vec![5.0; nx * ny]);
        let fg = frontogenesis(&theta, &u_const, &v_const, 1000.0, 1000.0);
        assert!(fg.values.iter().all(|value| value.abs() < 1e-10));

        let dx = 100_000.0;
        let dy = 100_000.0;
        let mut height = Grid2D::zeros(nx, ny);
        for y in 0..ny {
            for x in 0..nx {
                height.set(x, y, (y as f64) * 10.0);
            }
        }
        let (u_geo, v_geo) = geostrophic_wind(&height, &lats, dx, dy);
        let expected_u_geo = -(G / f_45) * (10.0 / dy);
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                assert!((u_geo.get(x, y) - expected_u_geo).abs() < 1.0);
                assert!(v_geo.get(x, y).abs() < 1e-6);
            }
        }

        let u_obs = Grid2D::new(3, 1, vec![15.0, 20.0, 10.0]);
        let v_obs = Grid2D::new(3, 1, vec![5.0, -3.0, 8.0]);
        let u_geo_small = Grid2D::new(3, 1, vec![12.0, 18.0, 9.0]);
        let v_geo_small = Grid2D::new(3, 1, vec![4.0, -2.0, 7.0]);
        let (ua, va) = ageostrophic_wind(&u_obs, &v_obs, &u_geo_small, &v_geo_small);
        assert_eq!(ua.values, vec![3.0, 2.0, 1.0]);
        assert_eq!(va.values, vec![1.0, -1.0, 1.0]);

        let (q1, q2) = q_vector(&theta, &u_const, &v_const, 500.0, 50_000.0, 50_000.0);
        assert!(q1.values.iter().all(|value| value.abs() < 1e-15));
        assert!(q2.values.iter().all(|value| value.abs() < 1e-15));

        let mut u_mix = Grid2D::zeros(7, 7);
        let mut v_mix = Grid2D::zeros(7, 7);
        for y in 0..7 {
            for x in 0..7 {
                u_mix.set(x, y, (x as f64) * 2.0 - (y as f64));
                v_mix.set(x, y, (y as f64) + (x as f64) * 0.5);
            }
        }
        let vort_total = vorticity_regular(&u_mix, &v_mix, 1.0, 1.0);
        let vort_curv = curvature_vorticity(&u_mix, &v_mix, 1.0, 1.0);
        let vort_shear = shear_vorticity(&u_mix, &v_mix, 1.0, 1.0);
        for idx in 0..u_mix.len() {
            assert!(approx(
                vort_curv.values[idx] + vort_shear.values[idx],
                vort_total.values[idx],
                1e-10
            ));
        }

        let mut u_geo_grad = Grid2D::zeros(nx, ny);
        let v_geo_grad = Grid2D::new(nx, ny, vec![0.0; nx * ny]);
        let u_total = Grid2D::new(nx, ny, vec![1.0; nx * ny]);
        let v_total = Grid2D::new(nx, ny, vec![0.0; nx * ny]);
        for y in 0..ny {
            for x in 0..nx {
                u_geo_grad.set(x, y, x as f64);
            }
        }
        let (u_ia, v_ia) =
            inertial_advective_wind(&u_total, &v_total, &u_geo_grad, &v_geo_grad, 1.0, 1.0);
        for y in 0..ny {
            for x in 1..nx - 1 {
                assert!(approx(u_ia.get(x, y), 1.0, 1e-10));
                assert!(approx(v_ia.get(x, y), 0.0, 1e-10));
            }
        }

        let u_abs = Grid2D::new(2, 1, vec![10.0, 20.0]);
        let lats_abs = Grid2D::new(2, 1, vec![45.0, 45.0]);
        let y_abs = Grid2D::new(2, 1, vec![0.0, 100_000.0]);
        let momentum = absolute_momentum(&u_abs, &lats_abs, &y_abs);
        assert!(approx(momentum.get(0, 0), 10.0, 1e-10));
        assert!((momentum.get(1, 0) - (20.0 - f_45 * 100_000.0)).abs() < 1e-6);

        let flux = kinematic_flux(
            &Grid2D::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
            &Grid2D::new(2, 2, vec![10.0, 20.0, 30.0, 40.0]),
        );
        assert_eq!(flux.values, vec![10.0, 40.0, 90.0, 160.0]);

        let lat_grid = Grid2D::new(
            3,
            3,
            vec![30.0, 30.0, 30.0, 31.0, 31.0, 31.0, 32.0, 32.0, 32.0],
        );
        let lon_grid = Grid2D::new(
            3,
            3,
            vec![
                -100.0, -99.0, -98.0, -100.0, -99.0, -98.0, -100.0, -99.0, -98.0,
            ],
        );
        let (dx_grid, dy_grid) = lat_lon_grid_deltas(&lat_grid, &lon_grid);
        assert!(dx_grid.values.iter().all(|value| *value > 0.0));
        assert!(dy_grid.values.iter().all(|value| *value > 0.0));
        let val_x = Grid2D::new(3, 3, vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
        let (dfdx, dfdy) = geospatial_gradient(&val_x, &lat_grid, &lon_grid);
        assert!(dfdx.values.iter().all(|value| value.is_finite()));
        assert!(dfdy.values.iter().all(|value| value.is_finite()));
        assert!(dfdx.values.iter().any(|value| value.abs() > 0.0));
        let lap = geospatial_laplacian(&val_x, &lat_grid, &lon_grid);
        assert!(lap.values.iter().all(|value| value.is_finite()));

        let lat_desc = Grid2D::new(
            3,
            3,
            vec![32.0, 32.0, 32.0, 31.0, 31.0, 31.0, 30.0, 30.0, 30.0],
        );
        let (_dx_desc, dy_desc) = lat_lon_grid_deltas(&lat_desc, &lon_grid);
        assert!(dy_desc.values.iter().all(|value| *value < 0.0));
        let val_y = Grid2D::new(3, 3, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        let (_dfdx_desc, dfdy_desc) = geospatial_gradient(&val_y, &lat_desc, &lon_grid);
        assert!(dfdy_desc.values.iter().all(|value| value.is_finite()));
        assert!(dfdy_desc.values.iter().all(|value| *value < 0.0));
    }

    #[test]
    fn uniform_flow_has_zero_vorticity() {
        let u = Grid2D::new(3, 3, vec![10.0; 9]);
        let v = Grid2D::new(3, 3, vec![5.0; 9]);
        let vort = vorticity_regular(&u, &v, 1_000.0, 1_000.0);
        assert!(vort.values.iter().all(|value| value.abs() < 1e-12));
    }

    #[test]
    fn linear_outflow_has_positive_divergence() {
        let u = Grid2D::new(3, 3, vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
        let v = Grid2D::new(3, 3, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        let div = divergence(&u, &v, 1.0, 1.0);
        assert!(div.values.iter().all(|value| approx(*value, 2.0, 1e-9)));
    }

    #[test]
    fn computes_smooth_and_derivative_helpers() {
        let linear = Grid2D::new(
            5,
            3,
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0,
            ],
        );
        let gx = gradient_x(&linear, 1.0);
        let gy = gradient_y(&linear, 1.0);
        let (gxx, gyy) = gradient(&linear, 1.0, 1.0);
        let first_x = first_derivative(&linear, 1.0, 0);
        let second_x = second_derivative(&linear, 1.0, 0);
        let lap = laplacian(&linear, 1.0, 1.0);
        for y in 0..linear.ny {
            for x in 0..linear.nx {
                assert!(approx(gx.get(x, y), 1.0, 1e-10));
                assert!(approx(gy.get(x, y), 0.0, 1e-10));
                assert!(approx(gxx.get(x, y), gx.get(x, y), 1e-10));
                assert!(approx(gyy.get(x, y), gy.get(x, y), 1e-10));
                assert!(approx(first_x.get(x, y), 1.0, 1e-10));
                assert!(approx(second_x.get(x, y), 0.0, 1e-10));
                assert!(approx(lap.get(x, y), 0.0, 1e-10));
            }
        }

        let mut spike = Grid2D::zeros(7, 7);
        spike.set(3, 3, 100.0);
        let gaussian = smooth_gaussian(&spike, 1.0);
        let rectangular = smooth_rectangular(&spike, 3, 1);
        let circular = smooth_circular(&spike, 1.5, 1);
        let five_point = smooth_n_point(&spike, 5, 1);
        let custom = smooth_window(&spike, &Grid2D::new(3, 3, vec![1.0; 9]), 1, true);
        assert!(gaussian.get(3, 3) < 100.0 && gaussian.get(3, 3) > 0.0);
        assert!(rectangular.get(3, 3) < 100.0 && rectangular.get(3, 3) > 0.0);
        assert!(circular.get(3, 3) < 100.0 && circular.get(3, 3) > 0.0);
        assert!(five_point.get(3, 3) < 100.0 && five_point.get(3, 3) > 0.0);
        assert!(custom.get(3, 3) < 100.0 && custom.get(3, 3) > 0.0);
    }

    #[test]
    fn computes_cross_section_and_3d_kinematics_helpers() {
        let ((te, tn), (ne, nn)) = unit_vectors_from_cross_section((45.0, -100.0), (45.0, -80.0));
        assert!(approx(te, 1.0, 1e-10));
        assert!(approx(tn, 0.0, 1e-10));
        assert!(approx(ne, 0.0, 1e-10));
        assert!(approx(nn, 1.0, 1e-10));

        let tang = tangential_component(&[10.0, 0.0], &[0.0, 5.0], (45.0, -100.0), (45.0, -80.0));
        let norm = normal_component(&[10.0, 0.0], &[0.0, 5.0], (45.0, -100.0), (45.0, -80.0));
        let (parallel, perpendicular) =
            cross_section_components(&[10.0, 0.0], &[0.0, 5.0], (45.0, -100.0), (45.0, -80.0));
        assert!(approx(tang[0], 10.0, 1e-10));
        assert!(approx(norm[1], 5.0, 1e-10));
        assert_eq!(tang, parallel);
        assert_eq!(norm, perpendicular);

        let u = Grid2D::new(3, 3, vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
        let v = Grid2D::new(
            3,
            3,
            vec![0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -2.0, -2.0, -2.0],
        );
        let (du_dx, du_dy, dv_dx, dv_dy) = vector_derivative(&u, &v, 1.0, 1.0);
        assert!(approx(du_dx.get(1, 1), 1.0, 1e-10));
        assert!(approx(du_dy.get(1, 1), 0.0, 1e-10));
        assert!(approx(dv_dx.get(1, 1), 0.0, 1e-10));
        assert!(approx(dv_dy.get(1, 1), -1.0, 1e-10));

        let scalar = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let advection = advection_3d(
            &scalar, &[1.0; 6], &[0.0; 6], &[0.0; 6], 3, 1, 2, 1.0, 1.0, 1.0,
        );
        assert!(approx(advection[1], -1.0, 1e-10));
        assert!(approx(advection[4], -1.0, 1e-10));
    }

    #[test]
    fn computes_utils_and_interpolation_helpers() {
        assert_eq!(angle_to_direction(225.0), "SW");
        assert_eq!(parse_angle("sse"), Some(157.5));
        assert_eq!(find_bounding_indices(&[10.0, 8.0, 6.0], 7.0), Some((1, 2)));
        assert_eq!(
            nearest_intersection_idx(&[0.0, 1.0, 2.0], &[0.0, 2.0, 0.0], &[1.0, 1.0, 1.0]),
            Some(0)
        );
        assert_eq!(
            resample_nn_1d(&[0.2, 1.8], &[0.0, 1.0, 2.0], &[10.0, 20.0, 30.0]),
            vec![10.0, 30.0]
        );
        assert_eq!(
            find_peaks(&[0.0, 5.0, 0.0, 10.0, 0.0], true, 0.0),
            vec![1, 3]
        );
        assert_eq!(peak_persistence(&[0.0, 10.0, 0.0, 5.0, 0.0], true)[0].0, 1);

        let (radar_lats, radar_lons) = azimuth_range_to_lat_lon(&[0.0], &[0.0], 35.0, -97.0);
        assert!(approx(radar_lats[0], 35.0, 1e-6));
        assert!(approx(radar_lons[0], -97.0, 1e-6));

        let interp = interpolate_1d(&[0.5, 1.5], &[0.0, 1.0, 2.0], &[0.0, 10.0, 20.0]);
        assert!(approx(interp[0], 5.0, 1e-10));
        assert!(approx(interp[1], 15.0, 1e-10));
        let log_interp = log_interpolate_1d(&[950.0], &[1000.0, 900.0], &[300.0, 290.0]);
        assert!(log_interp[0].is_finite());
        let mut nan_series = vec![f64::NAN, 0.0, f64::NAN, 10.0, f64::NAN];
        interpolate_nans_1d(&mut nan_series);
        assert_eq!(nan_series, vec![0.0, 0.0, 5.0, 10.0, 10.0]);

        let iso = interpolate_to_isosurface(
            &[10.0, 20.0, 30.0, 40.0],
            &[1000.0, 1000.0, 900.0, 900.0],
            950.0,
            &[1000.0, 900.0],
            2,
            1,
            2,
        );
        assert!(approx(iso[0], 20.0, 1e-10));
        assert!(approx(iso[1], 30.0, 1e-10));

        let point_idw = inverse_distance_to_points(
            &[0.0, 2.0],
            &[0.0, 0.0],
            &[10.0, 20.0],
            &[1.0],
            &[0.0],
            10.0,
            1,
            0,
            100_000.0,
            0.2,
        );
        assert!(approx(point_idw[0], 15.0, 1e-10));
        let point_nn =
            natural_neighbor_to_points(&[0.0, 0.0], &[0.0, 2.0], &[10.0, 20.0], &[0.0], &[1.0]);
        assert!(point_nn[0].is_finite());
        let point_dispatch = interpolate_to_points(
            &[0.0, 0.0],
            &[0.0, 2.0],
            &[10.0, 20.0],
            &[0.0],
            &[1.0],
            "nn",
        );
        assert!(point_dispatch[0].is_finite());

        let (keep_lats, keep_lons, keep_vals) =
            remove_nan_observations(&[0.0, 1.0], &[0.0, 1.0], &[f64::NAN, 2.0]);
        assert_eq!(keep_lats, vec![1.0]);
        assert_eq!(keep_lons, vec![1.0]);
        assert_eq!(keep_vals, vec![2.0]);
        let (_, _, above_vals) =
            remove_observations_below_value(&[0.0, 1.0], &[0.0, 1.0], &[1.0, 3.0], 2.0);
        assert_eq!(above_vals, vec![3.0]);
        let (_, _, dedup_vals) =
            remove_repeat_coordinates(&[0.0, 0.0, 1.0], &[0.0, 0.0, 1.0], &[1.0, 2.0, 3.0]);
        assert_eq!(dedup_vals, vec![1.0, 3.0]);

        let slice = interpolate_to_slice(
            &[10.0, 20.0, 30.0, 40.0],
            &[1000.0],
            &[0.0, 1.0],
            &[0.0, 1.0],
            &[0.0, 1.0],
            &[0.0, 1.0],
            2,
            2,
            1,
        );
        assert!(approx(slice[0][0], 10.0, 1e-10));
        assert!(approx(slice[1][0], 40.0, 1e-10));

        let (geo_lats, geo_lons) = geodesic((35.0, -100.0), (35.0, -90.0), 3);
        assert!(approx(geo_lats[0], 35.0, 1e-10));
        assert!(approx(geo_lats[2], 35.0, 1e-10));
        assert!(approx(geo_lons[0], -100.0, 1e-10));
        assert!(approx(geo_lons[2], -90.0, 1e-10));
    }

    #[test]
    fn calc_inventory_summary_is_consistent() {
        let summary = calc_port_summary();
        assert!(summary.total >= summary.ported);
        assert_eq!(summary.total, summary.ported + summary.missing);
        assert!(summary.ported >= 10);
    }
}
