use std::fs;
use std::path::PathBuf;

use clap::Parser;
use serde_json::json;
use wx_export::ExportEngine;
use wx_grib::GribEngine;

#[derive(Debug, Parser)]
#[command(name = "dump-grib-field")]
#[command(about = "Decode one GRIB field and dump the grid as NPY for external comparison")]
struct Cli {
    #[arg(long)]
    file: PathBuf,
    #[arg(long, default_value_t = 1)]
    message: u64,
    #[arg(long)]
    output_dir: PathBuf,
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    fs::create_dir_all(&cli.output_dir)
        .map_err(|err| format!("failed to create '{}': {err}", cli.output_dir.display()))?;

    let grib = GribEngine::new();
    let export = ExportEngine::new();
    let field = grib.decode_file_message(&cli.file, cli.message)?;

    export.write_npy_f32_grid(cli.output_dir.join("field.npy"), &field.grid)?;

    let (min, mean, max) = field
        .min_mean_max()
        .unwrap_or((f64::NAN, f64::NAN, f64::NAN));
    let manifest = json!({
        "file": cli.file,
        "message": cli.message,
        "grid": {
            "nx": field.grid.nx,
            "ny": field.grid.ny,
        },
        "field": {
            "variable": field.descriptor.variable,
            "level": field.descriptor.level,
            "units": field.descriptor.units,
            "min": min,
            "mean": mean,
            "max": max,
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
