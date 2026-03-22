use std::fs;
use std::path::PathBuf;

use clap::Parser;
use serde_json::json;
use wx_calc::{
    dewpoint_from_relative_humidity, downdraft_cape, k_index, lcl, lifted_index, parcel_profile,
    precipitable_water, showalter_index, total_totals, wet_bulb_potential_temperature,
    wet_bulb_temperature,
};

#[derive(Debug, Parser)]
#[command(name = "verify-thermo-profiles")]
#[command(about = "Emit fixed thermo/profile regression cases as JSON")]
struct Cli {
    #[arg(long, default_value = "examples/thermo_profile_verify")]
    output_dir: PathBuf,
}

struct SoundingCase {
    name: &'static str,
    pressure_hpa: Vec<f64>,
    temperature_c: Vec<f64>,
    dewpoint_c: Vec<f64>,
    surface_pressure_hpa: f64,
    surface_temperature_c: f64,
    surface_dewpoint_c: f64,
}

const DOC_PRESSURE_HPA: &[f64] = &[
    1008.0, 1000.0, 950.0, 900.0, 850.0, 800.0, 750.0, 700.0, 650.0, 600.0, 550.0, 500.0, 450.0,
    400.0, 350.0, 300.0, 250.0, 200.0, 175.0, 150.0, 125.0, 100.0, 80.0, 70.0, 60.0, 50.0, 40.0,
    30.0, 25.0, 20.0,
];
const DOC_TEMPERATURE_C: &[f64] = &[
    29.3, 28.1, 23.5, 20.9, 18.4, 15.9, 13.1, 10.1, 6.7, 3.1, -0.5, -4.5, -9.0, -14.8, -21.5,
    -29.7, -40.0, -52.4, -59.2, -66.5, -74.1, -78.5, -76.0, -71.6, -66.7, -61.3, -56.3, -51.7,
    -50.7, -47.5,
];
const DOC_RH_FRAC: &[f64] = &[
    0.85, 0.65, 0.36, 0.39, 0.82, 0.72, 0.75, 0.86, 0.65, 0.22, 0.52, 0.66, 0.64, 0.20, 0.05, 0.75,
    0.76, 0.45, 0.25, 0.48, 0.76, 0.88, 0.56, 0.88, 0.39, 0.67, 0.15, 0.04, 0.94, 0.35,
];
const PLAINS_PRESSURE_HPA: &[f64] = &[1000.0, 925.0, 850.0, 700.0, 500.0, 300.0];
const PLAINS_TEMPERATURE_C: &[f64] = &[25.0, 20.0, 8.0, -10.0, -35.0, -52.0];
const PLAINS_DEWPOINT_C: &[f64] = &[18.0, 14.0, 0.0, -20.0, -45.0, -60.0];
const TROPICAL_PRESSURE_HPA: &[f64] = &[1008.0, 1000.0, 950.0, 900.0, 850.0, 800.0, 700.0, 500.0];
const TROPICAL_TEMPERATURE_C: &[f64] = &[29.0, 28.4, 26.2, 23.0, 20.1, 17.8, 10.2, -4.2];
const TROPICAL_DEWPOINT_C: &[f64] = &[24.5, 24.0, 22.0, 19.5, 15.8, 10.8, -2.0, -18.0];

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    fs::create_dir_all(&cli.output_dir)
        .map_err(|err| format!("failed to create '{}': {err}", cli.output_dir.display()))?;

    let doc_dewpoint_c = DOC_TEMPERATURE_C
        .iter()
        .zip(DOC_RH_FRAC.iter())
        .map(|(&temperature_c, &rh_frac)| {
            dewpoint_from_relative_humidity(temperature_c, rh_frac * 100.0)
        })
        .collect::<Vec<_>>();

    let cases = vec![
        SoundingCase {
            name: "metpy_doc_profile",
            pressure_hpa: DOC_PRESSURE_HPA.to_vec(),
            temperature_c: DOC_TEMPERATURE_C.to_vec(),
            dewpoint_c: doc_dewpoint_c.clone(),
            surface_pressure_hpa: DOC_PRESSURE_HPA[0],
            surface_temperature_c: DOC_TEMPERATURE_C[0],
            surface_dewpoint_c: doc_dewpoint_c[0],
        },
        SoundingCase {
            name: "plains_profile",
            pressure_hpa: PLAINS_PRESSURE_HPA.to_vec(),
            temperature_c: PLAINS_TEMPERATURE_C.to_vec(),
            dewpoint_c: PLAINS_DEWPOINT_C.to_vec(),
            surface_pressure_hpa: PLAINS_PRESSURE_HPA[0],
            surface_temperature_c: PLAINS_TEMPERATURE_C[0],
            surface_dewpoint_c: PLAINS_DEWPOINT_C[0],
        },
        SoundingCase {
            name: "tropical_profile",
            pressure_hpa: TROPICAL_PRESSURE_HPA.to_vec(),
            temperature_c: TROPICAL_TEMPERATURE_C.to_vec(),
            dewpoint_c: TROPICAL_DEWPOINT_C.to_vec(),
            surface_pressure_hpa: TROPICAL_PRESSURE_HPA[0],
            surface_temperature_c: TROPICAL_TEMPERATURE_C[0],
            surface_dewpoint_c: TROPICAL_DEWPOINT_C[0],
        },
    ];

    let mut output_cases = Vec::with_capacity(cases.len());
    for case in cases {
        let (lcl_pressure_hpa, lcl_temperature_c) = lcl(
            case.surface_pressure_hpa,
            case.surface_temperature_c,
            case.surface_dewpoint_c,
        );
        let parcel = parcel_profile(
            &case.pressure_hpa,
            case.surface_temperature_c,
            case.surface_dewpoint_c,
        );
        let parcel_500_c = interp_log_pressure(500.0, &case.pressure_hpa, &parcel);
        output_cases.push(json!({
            "name": case.name,
            "surface": {
                "pressure_hpa": case.surface_pressure_hpa,
                "temperature_c": case.surface_temperature_c,
                "dewpoint_c": case.surface_dewpoint_c,
            },
            "profile": {
                "pressure_hpa": case.pressure_hpa,
                "temperature_c": case.temperature_c,
                "dewpoint_c": case.dewpoint_c,
            },
            "products": {
                "lcl_pressure_hpa": lcl_pressure_hpa,
                "lcl_temperature_c": lcl_temperature_c,
                "wet_bulb_temperature_c": wet_bulb_temperature(
                    case.surface_pressure_hpa,
                    case.surface_temperature_c,
                    case.surface_dewpoint_c,
                ),
                "wet_bulb_potential_temperature_k": wet_bulb_potential_temperature(
                    case.surface_pressure_hpa,
                    case.surface_temperature_c,
                    case.surface_dewpoint_c,
                ),
                "lifted_index_c": lifted_index(
                    &case.pressure_hpa,
                    &case.temperature_c,
                    &case.dewpoint_c,
                ),
                "showalter_index_c": showalter_index(
                    &case.pressure_hpa,
                    &case.temperature_c,
                    &case.dewpoint_c,
                ),
                "k_index_c": k_index_at_levels(&case.pressure_hpa, &case.temperature_c, &case.dewpoint_c),
                "total_totals_c": total_totals_at_levels(&case.pressure_hpa, &case.temperature_c, &case.dewpoint_c),
                "precipitable_water_mm": precipitable_water(&case.pressure_hpa, &case.dewpoint_c),
                "downdraft_cape_jkg": downdraft_cape(&case.pressure_hpa, &case.temperature_c, &case.dewpoint_c),
                "parcel_profile_500_c": parcel_500_c,
            }
        }));
    }

    let report = json!({ "cases": output_cases });
    fs::write(
        cli.output_dir.join("wxtrain_thermo_cases.json"),
        serde_json::to_string_pretty(&report)
            .map_err(|err| format!("failed to serialize thermo cases: {err}"))?,
    )
    .map_err(|err| format!("failed to write thermo cases: {err}"))?;

    println!("wrote {}", cli.output_dir.display());
    Ok(())
}

fn interp_log_pressure(target_hpa: f64, pressure_hpa: &[f64], values: &[f64]) -> f64 {
    let n = pressure_hpa.len().min(values.len());
    if n == 0 {
        return f64::NAN;
    }
    for i in 1..n {
        let p0 = pressure_hpa[i - 1];
        let p1 = pressure_hpa[i];
        if (p0 >= target_hpa && p1 <= target_hpa) || (p0 <= target_hpa && p1 >= target_hpa) {
            let frac = (target_hpa.ln() - p0.ln()) / (p1.ln() - p0.ln());
            return values[i - 1] + frac * (values[i] - values[i - 1]);
        }
    }
    f64::NAN
}

fn k_index_at_levels(pressure_hpa: &[f64], temperature_c: &[f64], dewpoint_c: &[f64]) -> f64 {
    let t850 = interp_log_pressure(850.0, pressure_hpa, temperature_c);
    let td850 = interp_log_pressure(850.0, pressure_hpa, dewpoint_c);
    let t700 = interp_log_pressure(700.0, pressure_hpa, temperature_c);
    let td700 = interp_log_pressure(700.0, pressure_hpa, dewpoint_c);
    let t500 = interp_log_pressure(500.0, pressure_hpa, temperature_c);
    k_index(t850, td850, t700, td700, t500)
}

fn total_totals_at_levels(pressure_hpa: &[f64], temperature_c: &[f64], dewpoint_c: &[f64]) -> f64 {
    let t850 = interp_log_pressure(850.0, pressure_hpa, temperature_c);
    let td850 = interp_log_pressure(850.0, pressure_hpa, dewpoint_c);
    let t500 = interp_log_pressure(500.0, pressure_hpa, temperature_c);
    total_totals(t850, td850, t500)
}
