use std::fs;
use std::fs::File;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;

use chrono::{DateTime, TimeZone, Utc};
use clap::{Parser, Subcommand, ValueEnum};
use parquet::data_type::{ByteArray, ByteArrayType, Int64Type};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::{SerializedFileWriter, SerializedRowGroupWriter};
use parquet::schema::parser::parse_message_type;
use wx_calc::{
    calc_port_summary, divergence, equivalent_potential_temperature, missing_names,
    potential_temperature, relative_humidity_from_dewpoint, vorticity, wind_direction, wind_speed,
    CalcEngine,
};
use wx_export::{ChannelNormEntry, ChannelNormStats, ChannelStats, ExportEngine, SampleChannelArtifact};
use wx_fetch::{
    resolve_offset_length, supported_sources_for, FetchEngine, FetchPlan, FetchRequest,
    ModelDownloadOptions, ModelKind, ProductKind, SourceKind,
};
use wx_grib::GribEngine;
use wx_radar::{ColorTable, RadarEngine};
use wx_render::{
    render_scalar_grid, write_png_rgba, ColorMap, RenderEngine, RenderJob, RenderTarget,
};
use wx_train::{
    plan_agent_job, AgentJobSpec, DatasetBuildManifest, DatasetSampleRef, DatasetSplit,
    DatasetSplitCounts, GribDatasetBuildRequest, GribSampleInput, LearningTask, ModelArchitecture,
    TrainingPlan,
};
use wx_types::{Grid2D, TrainingChannel};

mod feature_materialize;

#[derive(Debug, Parser)]
#[command(
    name = "wxtrain",
    about = "Rust-first end-to-end weather and ML pipeline workspace"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    About,
    Crates,
    Models,
    Fetch {
        #[command(subcommand)]
        command: FetchCommand,
    },
    PlanFetch {
        #[arg(long, value_enum)]
        model: ModelArg,
        #[arg(long, value_enum)]
        product: ProductArg,
        #[arg(long)]
        forecast_hour: u16,
        #[arg(long)]
        run: Option<String>,
        #[arg(long, value_enum)]
        source: Option<SourceArg>,
        #[arg(long, default_value_t = false)]
        available: bool,
    },
    ParseIdx {
        #[arg(long)]
        file: PathBuf,
        #[arg(long)]
        search: Option<String>,
    },
    ScanGrib {
        #[arg(long)]
        file: PathBuf,
        #[arg(long)]
        search: Option<String>,
    },
    DecodeGrib {
        #[arg(long)]
        file: PathBuf,
        #[arg(long, default_value_t = 1)]
        message: u64,
        #[arg(long, default_value_t = 8)]
        limit: usize,
    },
    Calc {
        #[command(subcommand)]
        command: CalcCommand,
    },
    Train {
        #[command(subcommand)]
        command: TrainCommand,
    },
    Render {
        #[command(subcommand)]
        command: RenderCommand,
    },
    Radar {
        #[command(subcommand)]
        command: RadarCommand,
    },
}

#[derive(Debug, Subcommand)]
enum CalcCommand {
    Parity {
        #[arg(long, default_value_t = 12)]
        limit: usize,
    },
    Thermo {
        #[arg(long)]
        temperature_c: f64,
        #[arg(long)]
        dewpoint_c: f64,
        #[arg(long)]
        pressure_hpa: f64,
    },
    Wind {
        #[arg(long)]
        u: f64,
        #[arg(long)]
        v: f64,
    },
    GridDemo {
        #[arg(long, default_value_t = 5)]
        nx: usize,
        #[arg(long, default_value_t = 5)]
        ny: usize,
        #[arg(long, default_value_t = 1_000.0)]
        dx_m: f64,
        #[arg(long, default_value_t = 1_000.0)]
        dy_m: f64,
    },
}

#[derive(Debug, Subcommand)]
enum TrainCommand {
    Plan {
        #[arg(long, value_enum, default_value_t = TrainPreset::Baseline)]
        preset: TrainPreset,
    },
    JobInit {
        #[arg(long)]
        output: PathBuf,
        #[arg(long, value_enum, default_value_t = ModelArchitectureArg::SwinTransformer)]
        architecture: ModelArchitectureArg,
        #[arg(long, value_enum, default_value_t = LearningTaskArg::Forecasting)]
        task: LearningTaskArg,
        #[arg(long)]
        dataset_name: Option<String>,
        #[arg(long)]
        job_name: Option<String>,
    },
    JobPlan {
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    JobBuild {
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long, value_enum, default_value_t = ColorMapArg::Gray)]
        colormap: ColorMapArg,
        #[arg(long)]
        min: Option<f64>,
        #[arg(long)]
        max: Option<f64>,
    },
    BuildGribSample {
        #[arg(long)]
        file: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long)]
        dataset_name: Option<String>,
        #[arg(long)]
        sample_id: Option<String>,
        #[arg(long, value_delimiter = ',')]
        messages: Vec<u64>,
        #[arg(long, value_enum, default_value_t = ColorMapArg::Gray)]
        colormap: ColorMapArg,
        #[arg(long)]
        min: Option<f64>,
        #[arg(long)]
        max: Option<f64>,
        /// Computed/derived channels to materialize (e.g. wind_speed,relative_humidity,stp).
        /// When specified, the pipeline decodes raw GRIB fields and runs wx-calc computations
        /// to produce the requested channels as NPY arrays.
        #[arg(long, value_delimiter = ',')]
        channels: Vec<String>,
    },
    BuildGribDataset {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long, value_enum, default_value_t = ColorMapArg::Gray)]
        colormap: ColorMapArg,
        #[arg(long)]
        min: Option<f64>,
        #[arg(long)]
        max: Option<f64>,
    },
}

#[derive(Debug, Subcommand)]
enum FetchCommand {
    Download {
        #[arg(long)]
        location: String,
        #[arg(long)]
        output: PathBuf,
    },
    ModelDownload {
        #[arg(long, value_enum)]
        model: ModelArg,
        #[arg(long, value_enum)]
        product: ProductArg,
        #[arg(long)]
        forecast_hour: u16,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        run: Option<String>,
        #[arg(long, value_enum)]
        source: Option<SourceArg>,
        #[arg(long, default_value_t = true)]
        available: bool,
        #[arg(long, value_delimiter = ',')]
        variables: Vec<String>,
        #[arg(long = "pressure-level", value_delimiter = ',')]
        pressure_levels: Vec<String>,
        #[arg(long)]
        area: Option<String>,
        #[arg(long)]
        grid: Option<String>,
    },
    Subset {
        #[arg(long)]
        grib: String,
        #[arg(long)]
        idx: String,
        #[arg(long)]
        search: String,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        limit: Option<usize>,
    },
    ModelSubset {
        #[arg(long, value_enum)]
        model: ModelArg,
        #[arg(long, value_enum)]
        product: ProductArg,
        #[arg(long)]
        forecast_hour: u16,
        #[arg(long)]
        search: String,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        run: Option<String>,
        #[arg(long, value_enum)]
        source: Option<SourceArg>,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long, default_value_t = true)]
        available: bool,
    },
    Batch {
        #[arg(long, value_enum)]
        model: ModelArg,
        #[arg(long, value_enum)]
        product: ProductArg,
        #[arg(long)]
        forecast_hours: String,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long, default_value_t = 4)]
        parallelism: usize,
        #[arg(long, value_enum)]
        source: Option<SourceArg>,
        #[arg(long)]
        run: Option<String>,
    },
}

#[derive(Debug, Subcommand)]
enum RenderCommand {
    Gradient {
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 256)]
        width: usize,
        #[arg(long, default_value_t = 256)]
        height: usize,
        #[arg(long, value_enum, default_value_t = ColorMapArg::Gray)]
        colormap: ColorMapArg,
    },
    Grib {
        #[arg(long)]
        file: PathBuf,
        #[arg(long, default_value_t = 1)]
        message: u64,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        min: Option<f64>,
        #[arg(long)]
        max: Option<f64>,
        #[arg(long, value_enum, default_value_t = ColorMapArg::Gray)]
        colormap: ColorMapArg,
    },
}

#[derive(Debug, Subcommand)]
enum RadarCommand {
    ParsePalette {
        #[arg(long)]
        file: PathBuf,
        #[arg(long)]
        value: i32,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModelArg {
    Hrrr,
    Gfs,
    Nam,
    Rap,
    EcmwfIfs,
    Era5,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ProductArg {
    Surface,
    Pressure,
    Native,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SourceArg {
    Nomads,
    Aws,
    Unidata,
    Ecmwf,
    Cds,
    LocalFilesystem,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TrainPreset {
    Baseline,
    Radar,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModelArchitectureArg {
    ClassicalMl,
    Diffusion,
    SwinTransformer,
    Fgn,
    Custom,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum LearningTaskArg {
    Regression,
    BinaryClassification,
    MulticlassClassification,
    Segmentation,
    Denoising,
    Forecasting,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ColorMapArg {
    Gray,
    Heat,
    Radar,
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();

    match cli.command {
        Command::About => {
            println!("wxtrain is a fresh Rust-first weather and ML pipeline workspace.");
            println!("It is designed to replace repo drift with one canonical crate graph.");
        }
        Command::Crates => {
            let fetch = FetchEngine::new();
            let grib = GribEngine::new();
            let calc = CalcEngine::new();
            let radar = RadarEngine::new();
            let render = RenderEngine::new();
            let export = ExportEngine::new();

            println!("wx-fetch: {}", fetch.design_goal());
            println!("wx-grib: {}", grib.design_goal());
            println!("wx-calc: {} priorities", calc.priorities().len());
            println!(
                "wx-radar: level2={}, detection={}",
                radar.capabilities().supports_level2,
                radar.capabilities().supports_rotation_detection
            );
            println!(
                "wx-render: render_job_valid={}",
                render.validate_job(&RenderJob {
                    name: "smoke".to_string(),
                    width: 512,
                    height: 512,
                    target: RenderTarget::Png,
                })
            );
            println!(
                "wx-export: {} preferred formats",
                export.recommended_formats().len()
            );
        }
        Command::Models => {
            print_model_summary(ModelKind::Hrrr);
            print_model_summary(ModelKind::Rap);
            print_model_summary(ModelKind::Nam);
            print_model_summary(ModelKind::Gfs);
            print_model_summary(ModelKind::EcmwfIfs);
            print_model_summary(ModelKind::Era5);
        }
        Command::Fetch { command } => match command {
            FetchCommand::Download { location, output } => {
                let engine = FetchEngine::new();
                let bytes = engine.download_to_file(&location, &output)?;
                println!("wrote {}", output.display());
                println!("bytes={bytes}");
            }
            FetchCommand::ModelDownload {
                model,
                product,
                forecast_hour,
                output,
                run,
                source,
                available,
                variables,
                pressure_levels,
                area,
                grid,
            } => {
                let engine = FetchEngine::new();
                let now = Utc::now();
                let model = model.into();
                let source = source.map(Into::into);
                let product_kind: ProductKind = product.into();
                let run_time = match run {
                    Some(run) => parse_run_time(&run)?,
                    None if available => engine.latest_available_cycle_for(
                        model,
                        product_kind,
                        source.as_ref().unwrap_or(&model.default_source()),
                        now,
                        forecast_hour,
                        12,
                    )?,
                    None => engine.latest_cycle_for(model, now),
                };
                let request = FetchRequest {
                    model,
                    run_time,
                    product: product_kind,
                    forecast_hour,
                    source,
                };
                let plan = engine.plan(&request)?;
                let options = ModelDownloadOptions {
                    variables,
                    pressure_levels,
                    area: area.as_deref().map(parse_quad).transpose()?,
                    grid: grid.as_deref().map(parse_pair).transpose()?,
                };
                let receipt = engine.execute_plan_to_file(&plan, &output, &options)?;
                println!("wrote {}", output.display());
                println!("bytes={}", receipt.bytes);
                println!("source={:?}", receipt.source);
                println!("location={}", receipt.location);
                if let Some(dataset) = receipt.dataset {
                    println!("dataset={dataset}");
                }
                if let Some(job_id) = receipt.job_id {
                    println!("job_id={job_id}");
                }
            }
            FetchCommand::Subset {
                grib,
                idx,
                search,
                output,
                limit,
            } => {
                fetch_subset_from_locations(&grib, &idx, &search, &output, limit)?;
            }
            FetchCommand::ModelSubset {
                model,
                product,
                forecast_hour,
                search,
                output,
                run,
                source,
                limit,
                available,
            } => {
                let engine = FetchEngine::new();
                let now = Utc::now();
                let model = model.into();
                let source = source.map(Into::into);
                let run_time = match run {
                    Some(run) => parse_run_time(&run)?,
                    None if available => engine.latest_available_cycle_for(
                        model,
                        product.into(),
                        source.as_ref().unwrap_or(&model.default_source()),
                        now,
                        forecast_hour,
                        12,
                    )?,
                    None => engine.latest_cycle_for(model, now),
                };
                let request = FetchRequest {
                    model,
                    run_time,
                    product: product.into(),
                    forecast_hour,
                    source,
                };
                let plan = engine.plan(&request)?;
                fetch_subset_from_locations(
                    &plan.grib_url,
                    &plan.idx_url,
                    &search,
                    &output,
                    limit,
                )?;
                println!("source={:?}", plan.source);
                println!("grib_url={}", plan.grib_url);
                println!("idx_url={}", plan.idx_url);
            }
            FetchCommand::Batch {
                model,
                product,
                forecast_hours,
                output_dir,
                parallelism,
                source,
                run,
            } => {
                let hours = parse_forecast_hours(&forecast_hours)?;
                let engine = FetchEngine::new();
                let now = Utc::now();
                let model_kind: ModelKind = model.into();
                let product_kind: ProductKind = product.into();
                let source_kind: Option<SourceKind> = source.map(Into::into);
                let run_time = match run {
                    Some(run) => parse_run_time(&run)?,
                    None => engine.latest_available_cycle_for(
                        model_kind,
                        product_kind,
                        source_kind.as_ref().unwrap_or(&model_kind.default_source()),
                        now,
                        *hours.first().unwrap_or(&0),
                        12,
                    )?,
                };
                fs::create_dir_all(&output_dir).map_err(|e| {
                    format!("failed to create output dir '{}': {e}", output_dir.display())
                })?;
                // Build plans for all forecast hours
                let mut plans: Vec<(u16, FetchPlan)> = Vec::new();
                for &fhr in &hours {
                    let request = FetchRequest {
                        model: model_kind,
                        run_time,
                        product: product_kind,
                        forecast_hour: fhr,
                        source: source_kind.clone(),
                    };
                    let plan = engine.plan(&request)?;
                    plans.push((fhr, plan));
                }
                let total = plans.len();
                println!(
                    "batch: downloading {} forecast hours for {:?} {:?} run={} parallelism={}",
                    total,
                    model_kind,
                    product_kind,
                    run_time.format("%Y%m%d%Hz"),
                    parallelism,
                );
                // Run async downloads with semaphore-limited concurrency
                let rt = tokio::runtime::Runtime::new()
                    .map_err(|e| format!("failed to create tokio runtime: {e}"))?;
                let results = rt.block_on(async {
                    let semaphore = Arc::new(tokio::sync::Semaphore::new(parallelism));
                    let client = reqwest::Client::builder()
                        .user_agent("wxtrain/0.1")
                        .build()
                        .map_err(|e| format!("failed to create async client: {e}"))?;
                    let completed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
                    let mut handles = Vec::new();
                    for (fhr, plan) in plans {
                        let sem = semaphore.clone();
                        let cl = client.clone();
                        let dir = output_dir.clone();
                        let done = completed.clone();
                        let total_count = total;
                        let url = plan.grib_url.clone();
                        let file_name = format!("{:?}_f{:02}.grib2", model_kind, fhr).to_lowercase();
                        handles.push(tokio::spawn(async move {
                            let _permit = sem.acquire().await.map_err(|e| format!("semaphore error: {e}"))?;
                            let dest = dir.join(&file_name);
                            let response = cl.get(&url)
                                .send()
                                .await
                                .map_err(|e| format!("GET f{fhr:02} failed: {e}"))?
                                .error_for_status()
                                .map_err(|e| format!("f{fhr:02} bad status: {e}"))?;
                            let bytes = response.bytes()
                                .await
                                .map_err(|e| format!("f{fhr:02} body read failed: {e}"))?;
                            fs::write(&dest, &bytes).map_err(|e| {
                                format!("f{fhr:02} write to '{}' failed: {e}", dest.display())
                            })?;
                            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                            println!(
                                "[{}/{}] f{:02} -> {} ({} bytes)",
                                n, total_count, fhr, dest.display(), bytes.len()
                            );
                            Ok::<(u16, u64), String>((fhr, bytes.len() as u64))
                        }));
                    }
                    let mut results = Vec::new();
                    for handle in handles {
                        results.push(handle.await.map_err(|e| format!("task join error: {e}"))?);
                    }
                    Ok::<Vec<Result<(u16, u64), String>>, String>(results)
                })?;
                let mut total_bytes: u64 = 0;
                let mut failures = 0u32;
                for result in &results {
                    match result {
                        Ok((_, bytes)) => total_bytes += bytes,
                        Err(e) => {
                            eprintln!("error: {e}");
                            failures += 1;
                        }
                    }
                }
                println!(
                    "batch complete: {} succeeded, {} failed, {} total bytes",
                    results.len() as u32 - failures,
                    failures,
                    total_bytes,
                );
                if failures > 0 {
                    return Err(format!("{failures} download(s) failed"));
                }
            }
        },
        Command::PlanFetch {
            model,
            product,
            forecast_hour,
            run,
            source,
            available,
        } => {
            let engine = FetchEngine::new();
            let now = Utc::now();
            let model = model.into();
            let source = source.map(Into::into);
            let run_time = match run {
                Some(run) => parse_run_time(&run)?,
                None if available => engine.latest_available_cycle_for(
                    model,
                    product.into(),
                    source.as_ref().unwrap_or(&model.default_source()),
                    now,
                    forecast_hour,
                    12,
                )?,
                None => engine.latest_cycle_for(model, now),
            };
            let request = FetchRequest {
                model,
                run_time,
                product: product.into(),
                forecast_hour,
                source,
            };
            let plan = engine.plan(&request)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&plan)
                    .map_err(|err| format!("failed to serialize fetch plan: {err}"))?
            );
        }
        Command::ParseIdx { file, search } => {
            let text = fs::read_to_string(&file)
                .map_err(|err| format!("failed to read '{}': {err}", file.display()))?;
            let engine = GribEngine::new();
            let inventory = engine.parse_idx_text(&text)?;
            let matches = match search {
                Some(needle) => engine.search(&inventory, &needle),
                None => inventory.messages.iter().collect(),
            };
            println!("messages: {}", inventory.messages.len());
            for message in matches {
                println!(
                    "#{:>3} ({}) off={} len={} {} {}",
                    message.message_no,
                    message.message_id,
                    message.offset_bytes,
                    message.length_bytes,
                    message.variable,
                    message.level
                );
            }
        }
        Command::ScanGrib { file, search } => {
            let engine = GribEngine::new();
            let inventory = engine.scan_file(&file)?;
            let matches = match search {
                Some(needle) => engine.search(&inventory, &needle),
                None => inventory.messages.iter().collect(),
            };
            println!("messages: {}", inventory.messages.len());
            for message in matches {
                println!(
                        "#{:>3} ({}) off={} len={} edition={:?} var={} level={} grid={:?} dims={:?}x{:?} ref_time={}",
                        message.message_no,
                        message.message_id,
                        message.offset_bytes,
                        message.length_bytes,
                        message.edition,
                        message.variable,
                        message.level,
                        message.grid_template,
                        message.nx,
                        message.ny,
                        message.reference_time.as_deref().unwrap_or("unknown"),
                    );
            }
        }
        Command::DecodeGrib {
            file,
            message,
            limit,
        } => {
            let engine = GribEngine::new();
            let field = engine.decode_file_message(&file, message)?;
            let preview = field
                .grid
                .values
                .iter()
                .take(limit)
                .map(|value| {
                    if value.is_nan() {
                        "NaN".to_string()
                    } else {
                        format!("{value:.3}")
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            println!("message={}", field.descriptor.message_no);
            println!("variable={}", field.descriptor.variable);
            println!("level={}", field.descriptor.level);
            println!("grid={}x{}", field.grid.nx, field.grid.ny);
            println!("missing_count={}", field.missing_count);
            if let Some((min, mean, max)) = field.min_mean_max() {
                println!("min={min:.3}");
                println!("mean={mean:.3}");
                println!("max={max:.3}");
            }
            if let Some(axis) = &field.x_axis {
                let axis_preview = axis
                    .values
                    .iter()
                    .take(limit)
                    .map(|value| format!("{value:.3}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                println!("x_axis={} [{}]", axis.name, axis_preview);
            }
            if let Some(axis) = &field.y_axis {
                let axis_preview = axis
                    .values
                    .iter()
                    .take(limit)
                    .map(|value| format!("{value:.3}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                println!("y_axis={} [{}]", axis.name, axis_preview);
            }
            println!("preview=[{}]", preview);
        }
        Command::Calc { command } => match command {
            CalcCommand::Parity { limit } => {
                let summary = calc_port_summary();
                println!("total_targets={}", summary.total);
                println!("ported={}", summary.ported);
                println!("missing={}", summary.missing);
                for category in summary.categories {
                    println!(
                        "{}: {}/{} ported (missing={})",
                        category.category.as_str(),
                        category.ported,
                        category.total,
                        category.missing
                    );
                }
                let sample_missing = missing_names(limit);
                if !sample_missing.is_empty() {
                    println!("next_missing={}", sample_missing.join(", "));
                }
            }
            CalcCommand::Thermo {
                temperature_c,
                dewpoint_c,
                pressure_hpa,
            } => {
                let theta = potential_temperature(pressure_hpa, temperature_c);
                let theta_e =
                    equivalent_potential_temperature(pressure_hpa, temperature_c, dewpoint_c);
                let rh = relative_humidity_from_dewpoint(temperature_c, dewpoint_c);
                println!("theta_k={theta:.3}");
                println!("theta_e_k={theta_e:.3}");
                println!("relative_humidity_pct={rh:.3}");
            }
            CalcCommand::Wind { u, v } => {
                let speed = wind_speed(u, v);
                let direction = wind_direction(u, v);
                println!("wind_speed={speed:.3}");
                println!("wind_direction_deg={direction:.3}");
            }
            CalcCommand::GridDemo { nx, ny, dx_m, dy_m } => {
                let mut u = Grid2D::zeros(nx, ny);
                let mut v = Grid2D::zeros(nx, ny);
                for y in 0..ny {
                    for x in 0..nx {
                        u.set(x, y, x as f64);
                        v.set(x, y, y as f64);
                    }
                }
                let div = divergence(&u, &v, dx_m, dy_m);
                let vort = vorticity(&u, &v, dx_m, dy_m);
                println!("grid_points={}", div.len());
                println!("div_center={:.6}", div.get(nx.min(2) - 1, ny.min(2) - 1));
                println!("vort_center={:.6}", vort.get(nx.min(2) - 1, ny.min(2) - 1));
            }
        },
        Command::Train { command } => match command {
            TrainCommand::Plan { preset } => {
                let plan = match preset {
                    TrainPreset::Baseline => TrainingPlan::baseline(),
                    TrainPreset::Radar => TrainingPlan::radar_supervised(),
                };
                let export = ExportEngine::new();
                let manifest = export.manifest_from_plan(&plan.export);
                println!(
                    "{}",
                    export
                        .to_json_pretty(&manifest)
                        .map_err(|err| format!("failed to print manifest: {err}"))?
                );
            }
            TrainCommand::JobInit {
                output,
                architecture,
                task,
                dataset_name,
                job_name,
            } => {
                let dataset_name = dataset_name.unwrap_or_else(|| {
                    format!(
                        "{}-dataset",
                        sanitize_filename_component(&format!("{architecture:?}"))
                    )
                });
                let job_name = job_name.unwrap_or_else(|| {
                    format!("{}-job", sanitize_filename_component(&dataset_name))
                });
                let spec =
                    AgentJobSpec::starter(job_name, dataset_name, architecture.into(), task.into());
                write_json_file(&output, &spec)?;
                println!("wrote {}", output.display());
            }
            TrainCommand::JobPlan { spec, output } => {
                let spec = read_agent_job_spec(&spec)?;
                let plan = plan_agent_job(&spec);
                if let Some(output) = output {
                    write_json_file(&output, &plan)?;
                    println!("wrote {}", output.display());
                } else {
                    print_json(&plan)?;
                }
            }
            TrainCommand::JobBuild {
                spec,
                output_dir,
                colormap,
                min,
                max,
            } => {
                let spec = read_agent_job_spec(&spec)?;
                let plan = plan_agent_job(&spec);
                write_json_file(&output_dir.join("job_plan.json"), &plan)?;
                write_json_file(&output_dir.join("model_recipe.json"), &plan.model_recipe)?;
                let Some(request) = plan.dataset_build_request else {
                    return Err(
                        "job spec does not resolve to executable local samples; fetch/model-window expansion is still required"
                            .to_string(),
                    );
                };
                write_json_file(&output_dir.join("dataset_request.json"), &request)?;
                build_grib_training_dataset_request_with_channels(
                    &request,
                    &output_dir,
                    colormap,
                    min,
                    max,
                    Some(&plan.training.spec.channels),
                )?;
            }
            TrainCommand::BuildGribSample {
                file,
                output_dir,
                dataset_name,
                sample_id,
                messages,
                colormap,
                min,
                max,
                channels,
            } => {
                if channels.is_empty() {
                    build_grib_training_sample(
                        &file,
                        &output_dir,
                        dataset_name,
                        sample_id,
                        messages,
                        colormap,
                        min,
                        max,
                    )?;
                } else {
                    build_grib_computed_sample(
                        &file,
                        &output_dir,
                        dataset_name,
                        sample_id,
                        messages,
                        colormap,
                        min,
                        max,
                        channels,
                    )?;
                }
            }
            TrainCommand::BuildGribDataset {
                manifest,
                output_dir,
                colormap,
                min,
                max,
            } => {
                build_grib_training_dataset(&manifest, &output_dir, colormap, min, max)?;
            }
        },
        Command::Render { command } => match command {
            RenderCommand::Gradient {
                output,
                width,
                height,
                colormap,
            } => {
                let mut values = Vec::with_capacity(width * height);
                for _y in 0..height {
                    for x in 0..width {
                        let denom = (width.saturating_sub(1)).max(1) as f64;
                        values.push(x as f64 / denom);
                    }
                }
                let grid = Grid2D::new(width, height, values);
                let rgba = render_scalar_grid(&grid, 0.0, 1.0, colormap.into());
                write_png_rgba(&output, width as u32, height as u32, &rgba)?;
                println!("wrote {}", output.display());
            }
            RenderCommand::Grib {
                file,
                message,
                output,
                min,
                max,
                colormap,
            } => {
                let engine = GribEngine::new();
                let field = engine.decode_file_message(&file, message)?;
                let (auto_min, _, auto_max) = field
                    .min_mean_max()
                    .ok_or_else(|| "decoded field contains no finite values".to_string())?;
                let rgba = render_scalar_grid(
                    &field.grid,
                    min.unwrap_or(auto_min),
                    max.unwrap_or(auto_max),
                    colormap.into(),
                );
                write_png_rgba(&output, field.grid.nx as u32, field.grid.ny as u32, &rgba)?;
                println!("wrote {}", output.display());
                println!("variable={}", field.descriptor.variable);
                println!("level={}", field.descriptor.level);
            }
        },
        Command::Radar { command } => match command {
            RadarCommand::ParsePalette { file, value } => {
                let text = fs::read_to_string(&file)
                    .map_err(|err| format!("failed to read '{}': {err}", file.display()))?;
                let table = ColorTable::from_pal_str(
                    file.file_stem()
                        .and_then(|stem| stem.to_str())
                        .unwrap_or("palette"),
                    &text,
                )?;
                let rgba = table.sample(value);
                println!("name={}", table.name);
                println!("stops={}", table.stops.len());
                println!("sample={},{},{},{}", rgba[0], rgba[1], rgba[2], rgba[3]);
            }
        },
    }

    Ok(())
}

fn print_model_summary(model: ModelKind) {
    let sources = supported_sources_for(model)
        .iter()
        .map(|source| format!("{source:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!(
        "{model:?}: cadence={}h default_source={:?} supported_sources=[{}]",
        model.cadence_hours(),
        model.default_source(),
        sources,
    );
}

fn parse_forecast_hours(raw: &str) -> Result<Vec<u16>, String> {
    let raw = raw.trim();
    if raw.contains('-') && !raw.contains(',') {
        let parts: Vec<&str> = raw.splitn(2, '-').collect();
        let start: u16 = parts[0]
            .trim()
            .parse()
            .map_err(|e| format!("invalid range start '{}': {e}", parts[0]))?;
        let end: u16 = parts[1]
            .trim()
            .parse()
            .map_err(|e| format!("invalid range end '{}': {e}", parts[1]))?;
        if end < start {
            return Err(format!("range end {end} is less than start {start}"));
        }
        Ok((start..=end).collect())
    } else {
        raw.split(',')
            .map(|s| {
                s.trim()
                    .parse::<u16>()
                    .map_err(|e| format!("invalid forecast hour '{}': {e}", s.trim()))
            })
            .collect()
    }
}

fn parse_run_time(raw: &str) -> Result<DateTime<Utc>, String> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(raw) {
        return Ok(dt.with_timezone(&Utc));
    }
    if raw.len() == 10 {
        let year = raw[0..4]
            .parse::<i32>()
            .map_err(|err| format!("invalid run year: {err}"))?;
        let month = raw[4..6]
            .parse::<u32>()
            .map_err(|err| format!("invalid run month: {err}"))?;
        let day = raw[6..8]
            .parse::<u32>()
            .map_err(|err| format!("invalid run day: {err}"))?;
        let hour = raw[8..10]
            .parse::<u32>()
            .map_err(|err| format!("invalid run hour: {err}"))?;
        return Utc
            .with_ymd_and_hms(year, month, day, hour, 0, 0)
            .single()
            .ok_or_else(|| format!("invalid UTC run time '{raw}'"));
    }
    Err(format!(
        "unsupported run time '{raw}', use RFC3339 or YYYYMMDDHH"
    ))
}

fn parse_quad(raw: &str) -> Result<[f64; 4], String> {
    let values = raw
        .split(',')
        .map(|item| {
            item.trim()
                .parse::<f64>()
                .map_err(|err| format!("invalid numeric value '{item}' in '{raw}': {err}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() != 4 {
        return Err(format!(
            "expected four comma-separated values for area, got '{}'",
            raw
        ));
    }
    Ok([values[0], values[1], values[2], values[3]])
}

fn parse_pair(raw: &str) -> Result<[f64; 2], String> {
    let values = raw
        .split(',')
        .map(|item| {
            item.trim()
                .parse::<f64>()
                .map_err(|err| format!("invalid numeric value '{item}' in '{raw}': {err}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() != 2 {
        return Err(format!(
            "expected two comma-separated values for grid, got '{}'",
            raw
        ));
    }
    Ok([values[0], values[1]])
}

fn read_agent_job_spec(path: &PathBuf) -> Result<AgentJobSpec, String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read agent job spec '{}': {err}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|err| format!("failed to parse agent job spec '{}': {err}", path.display()))
}

fn print_json(value: &impl serde::Serialize) -> Result<(), String> {
    println!(
        "{}",
        serde_json::to_string_pretty(value)
            .map_err(|err| format!("failed to serialize json output: {err}"))?
    );
    Ok(())
}

fn write_json_file(path: &PathBuf, value: &impl serde::Serialize) -> Result<(), String> {
    let json = serde_json::to_string_pretty(value)
        .map_err(|err| format!("failed to serialize '{}': {err}", path.display()))?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create parent directory for '{}': {err}",
                path.display()
            )
        })?;
    }
    fs::write(path, json).map_err(|err| format!("failed to write '{}': {err}", path.display()))
}

impl From<ModelArg> for ModelKind {
    fn from(value: ModelArg) -> Self {
        match value {
            ModelArg::Hrrr => Self::Hrrr,
            ModelArg::Gfs => Self::Gfs,
            ModelArg::Nam => Self::Nam,
            ModelArg::Rap => Self::Rap,
            ModelArg::EcmwfIfs => Self::EcmwfIfs,
            ModelArg::Era5 => Self::Era5,
        }
    }
}

impl From<ProductArg> for ProductKind {
    fn from(value: ProductArg) -> Self {
        match value {
            ProductArg::Surface => Self::Surface,
            ProductArg::Pressure => Self::Pressure,
            ProductArg::Native => Self::Native,
        }
    }
}

impl From<SourceArg> for SourceKind {
    fn from(value: SourceArg) -> Self {
        match value {
            SourceArg::Nomads => Self::Nomads,
            SourceArg::Aws => Self::Aws,
            SourceArg::Unidata => Self::Unidata,
            SourceArg::Ecmwf => Self::Ecmwf,
            SourceArg::Cds => Self::Cds,
            SourceArg::LocalFilesystem => Self::LocalFilesystem,
        }
    }
}

impl From<ModelArchitectureArg> for ModelArchitecture {
    fn from(value: ModelArchitectureArg) -> Self {
        match value {
            ModelArchitectureArg::ClassicalMl => Self::ClassicalMl,
            ModelArchitectureArg::Diffusion => Self::Diffusion,
            ModelArchitectureArg::SwinTransformer => Self::SwinTransformer,
            ModelArchitectureArg::Fgn => Self::ForecastGraphNetwork,
            ModelArchitectureArg::Custom => Self::Custom,
        }
    }
}

impl From<LearningTaskArg> for LearningTask {
    fn from(value: LearningTaskArg) -> Self {
        match value {
            LearningTaskArg::Regression => Self::Regression,
            LearningTaskArg::BinaryClassification => Self::BinaryClassification,
            LearningTaskArg::MulticlassClassification => Self::MulticlassClassification,
            LearningTaskArg::Segmentation => Self::Segmentation,
            LearningTaskArg::Denoising => Self::Denoising,
            LearningTaskArg::Forecasting => Self::Forecasting,
        }
    }
}

impl From<ColorMapArg> for ColorMap {
    fn from(value: ColorMapArg) -> Self {
        match value {
            ColorMapArg::Gray => Self::Gray,
            ColorMapArg::Heat => Self::Heat,
            ColorMapArg::Radar => Self::Radar,
        }
    }
}

fn build_grib_training_sample(
    file: &PathBuf,
    output_dir: &PathBuf,
    dataset_name: Option<String>,
    sample_id: Option<String>,
    messages: Vec<u64>,
    colormap: ColorMapArg,
    min: Option<f64>,
    max: Option<f64>,
) -> Result<(), String> {
    let sample_manifest = write_grib_training_sample(
        file,
        output_dir,
        dataset_name,
        sample_id,
        messages,
        colormap,
        min,
        max,
    )?;

    println!("wrote {}", output_dir.display());
    println!("channels={}", sample_manifest.channel_count);
    println!("dataset={}", sample_manifest.dataset_name);
    println!("sample_id={}", sample_manifest.sample_id);
    Ok(())
}

fn build_grib_computed_sample(
    file: &PathBuf,
    output_dir: &PathBuf,
    dataset_name: Option<String>,
    sample_id: Option<String>,
    messages: Vec<u64>,
    colormap: ColorMapArg,
    min: Option<f64>,
    max: Option<f64>,
    channels: Vec<String>,
) -> Result<(), String> {
    let dataset_name = dataset_name.unwrap_or_else(|| format!("{}-dataset", safe_file_stem(file)));
    let sample_id = sample_id.unwrap_or_else(|| safe_file_stem(file));

    let channel_specs: Vec<TrainingChannel> = channels
        .iter()
        .map(|name| {
            let units = infer_channel_units(name);
            TrainingChannel {
                name: name.clone(),
                units,
            }
        })
        .collect();

    let sample = GribSampleInput {
        file: file.display().to_string(),
        sample_id: Some(sample_id.clone()),
        messages,
        companion_files: Vec::new(),
    };

    let sample_manifest = feature_materialize::write_planned_training_sample(
        &sample,
        output_dir,
        dataset_name.clone(),
        sample_id.clone(),
        &channel_specs,
        colormap.into(),
        min,
        max,
    )?;

    println!("wrote {}", output_dir.display());
    println!("channels={}", sample_manifest.channel_count);
    println!("dataset={}", sample_manifest.dataset_name);
    println!("sample_id={}", sample_manifest.sample_id);
    for channel in &sample_manifest.channels {
        println!(
            "  {} [{}] {}",
            channel.name,
            channel.level,
            channel
                .stats
                .as_ref()
                .map(|s| format!("min={:.2} mean={:.2} max={:.2} std={:.2} count={} nan_count={}", s.min, s.mean, s.max, s.std, s.count, s.nan_count))
                .unwrap_or_else(|| "no stats".to_string())
        );
    }
    Ok(())
}

fn infer_channel_units(name: &str) -> String {
    match name.to_ascii_lowercase().as_str() {
        "wind_speed" | "wind_speed_10m" | "shear06" => "m/s".to_string(),
        "wind_direction" | "wind_direction_10m" => "degrees".to_string(),
        "relative_humidity" | "rh2m" => "%".to_string(),
        "t2m" | "d2m" | "t850" | "theta850" | "theta_e" | "wet_bulb"
        | "wet_bulb_potential_temperature" => "K".to_string(),
        "u10" | "v10" | "u850" | "v850" => "m/s".to_string(),
        "mslp" => "Pa".to_string(),
        "z500" => "gpm".to_string(),
        "vort500" | "div500" => "1/s".to_string(),
        "tadv850" => "K/s".to_string(),
        "sbcape" | "mlcape" | "mucape" | "dcape" => "J/kg".to_string(),
        "sbcin" | "mlcin" | "mucin" => "J/kg".to_string(),
        "srh01" | "srh03" => "m2/s2".to_string(),
        "stp" | "scp" => "1".to_string(),
        "pwat" => "mm".to_string(),
        "lcl_height" | "lfc_height" => "m".to_string(),
        "reflectivity" => "dBZ".to_string(),
        _ => "?".to_string(),
    }
}

fn write_grib_training_sample(
    file: &PathBuf,
    output_dir: &PathBuf,
    dataset_name: Option<String>,
    sample_id: Option<String>,
    messages: Vec<u64>,
    colormap: ColorMapArg,
    min: Option<f64>,
    max: Option<f64>,
) -> Result<wx_export::SampleBundleManifest, String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create output directory '{}': {err}",
            output_dir.display()
        )
    })?;

    let grib = GribEngine::new();
    let export = ExportEngine::new();
    let inventory = grib.scan_file(file)?;
    let message_ids = if messages.is_empty() {
        inventory
            .messages
            .iter()
            .map(|message| message.message_no)
            .collect::<Vec<_>>()
    } else {
        messages
    };

    if message_ids.is_empty() {
        return Err("no GRIB messages selected".to_string());
    }

    let dataset_name = dataset_name.unwrap_or_else(|| format!("{}-dataset", safe_file_stem(file)));
    let sample_id = sample_id.unwrap_or_else(|| safe_file_stem(file));

    let mut channel_specs = Vec::new();
    let mut artifacts = Vec::new();

    for message_no in message_ids {
        let field = grib.decode_file_message(file, message_no)?;
        let units = field
            .descriptor
            .units
            .clone()
            .unwrap_or_else(|| "?".to_string());
        channel_specs.push(TrainingChannel {
            name: field.descriptor.variable.clone(),
            units: units.clone(),
        });

        let stem = format!(
            "{:03}_{}_{}",
            field.descriptor.message_no,
            sanitize_filename_component(&field.descriptor.variable),
            sanitize_filename_component(&field.descriptor.level)
        );
        let data_name = format!("{stem}.npy");
        let preview_name = format!("{stem}.png");
        let data_path = output_dir.join(&data_name);
        let preview_path = output_dir.join(&preview_name);

        export.write_npy_f32_grid(&data_path, &field.grid)?;

        let stats = if let Some(full_stats) = compute_channel_stats(&field.grid) {
            let rgba = render_scalar_grid(
                &field.grid,
                min.unwrap_or(full_stats.min),
                max.unwrap_or(full_stats.max),
                colormap.into(),
            );
            write_png_rgba(
                &preview_path,
                field.grid.nx as u32,
                field.grid.ny as u32,
                &rgba,
            )?;
            Some(full_stats)
        } else {
            None
        };

        artifacts.push(SampleChannelArtifact {
            message_no: field.descriptor.message_no,
            name: field.descriptor.variable.clone(),
            level: field.descriptor.level.clone(),
            units,
            width: field.grid.nx,
            height: field.grid.ny,
            missing_count: field.missing_count,
            data_file: data_name,
            preview_file: if stats.is_some() {
                Some(preview_name)
            } else {
                None
            },
            stats,
        });
    }

    let plan = TrainingPlan::from_channels(dataset_name, channel_specs);
    let dataset_manifest = export.manifest_from_plan(&plan.export);
    export.write_manifest(output_dir.join("dataset_manifest.json"), &dataset_manifest)?;
    let sample_manifest = export.sample_bundle_manifest(
        &plan.export,
        sample_id,
        file.display().to_string(),
        artifacts,
    );
    export
        .write_sample_bundle_manifest(output_dir.join("sample_manifest.json"), &sample_manifest)?;
    write_channel_stats_json(output_dir, &sample_manifest.channels)?;
    Ok(sample_manifest)
}

#[derive(Clone)]
struct BuiltSample {
    index: usize,
    sample_ref: DatasetSampleRef,
    channels: Vec<TrainingChannel>,
}

fn build_grib_training_dataset(
    manifest: &PathBuf,
    output_dir: &PathBuf,
    colormap: ColorMapArg,
    min: Option<f64>,
    max: Option<f64>,
) -> Result<(), String> {
    let request_text = fs::read_to_string(manifest).map_err(|err| {
        format!(
            "failed to read dataset manifest '{}': {err}",
            manifest.display()
        )
    })?;
    let request: GribDatasetBuildRequest = serde_json::from_str(&request_text).map_err(|err| {
        format!(
            "failed to parse dataset manifest '{}': {err}",
            manifest.display()
        )
    })?;

    build_grib_training_dataset_request(&request, output_dir, colormap, min, max)
}

fn build_grib_training_dataset_request(
    request: &GribDatasetBuildRequest,
    output_dir: &PathBuf,
    colormap: ColorMapArg,
    min: Option<f64>,
    max: Option<f64>,
) -> Result<(), String> {
    build_grib_training_dataset_request_with_channels(request, output_dir, colormap, min, max, None)
}

fn build_grib_training_dataset_request_with_channels(
    request: &GribDatasetBuildRequest,
    output_dir: &PathBuf,
    colormap: ColorMapArg,
    min: Option<f64>,
    max: Option<f64>,
    requested_channels: Option<&[TrainingChannel]>,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create output directory '{}': {err}",
            output_dir.display()
        )
    })?;
    if request.samples.is_empty() {
        return Err("dataset manifest contains no samples".to_string());
    }
    if !(0.0..=1.0).contains(&request.train_fraction)
        || !(0.0..=1.0).contains(&request.validation_fraction)
        || request.train_fraction + request.validation_fraction > 1.0
    {
        return Err("invalid split fractions in dataset manifest".to_string());
    }

    let export = ExportEngine::new();
    let shard_count = request.shard_count.max(1);
    let parallelism = request.parallelism.max(1);
    let total_samples = request.samples.len();
    let train_cutoff = ((total_samples as f64) * request.train_fraction).round() as usize;
    let validation_cutoff =
        train_cutoff + ((total_samples as f64) * request.validation_fraction).round() as usize;
    let built_samples = build_dataset_samples_parallel(
        &request,
        output_dir,
        colormap,
        min,
        max,
        parallelism,
        shard_count,
        train_cutoff,
        validation_cutoff,
        requested_channels,
    )?;
    let mut dataset_channels: Vec<TrainingChannel> = Vec::new();
    let mut sample_refs = Vec::with_capacity(built_samples.len());
    let mut total_channel_count = 0usize;
    for built in built_samples {
        total_channel_count += built.sample_ref.channel_count;
        if dataset_channels.is_empty() {
            dataset_channels = built.channels;
        }
        sample_refs.push(built.sample_ref);
    }

    let mut plan = TrainingPlan::from_channels(request.dataset_name.clone(), dataset_channels);
    plan.export.format = request.format;
    plan.export.shard_count = shard_count;
    let dataset_manifest = export.manifest_from_plan(&plan.export);
    export.write_manifest(output_dir.join("dataset_manifest.json"), &dataset_manifest)?;
    let split_counts = DatasetSplitCounts {
        train: sample_refs
            .iter()
            .filter(|sample| sample.split == DatasetSplit::Train)
            .count(),
        validation: sample_refs
            .iter()
            .filter(|sample| sample.split == DatasetSplit::Validation)
            .count(),
        test: sample_refs
            .iter()
            .filter(|sample| sample.split == DatasetSplit::Test)
            .count(),
    };
    let build_manifest = DatasetBuildManifest {
        dataset_name: request.dataset_name.clone(),
        generated_at: Utc::now().to_rfc3339(),
        format: plan.export.format,
        shard_count,
        sample_count: sample_refs.len(),
        total_channel_count,
        split_counts,
        samples: sample_refs,
    };
    let build_manifest_json = serde_json::to_string_pretty(&build_manifest)
        .map_err(|err| format!("failed to serialize dataset build manifest: {err}"))?;
    fs::write(
        output_dir.join("dataset_build_manifest.json"),
        build_manifest_json,
    )
    .map_err(|err| {
        format!(
            "failed to write dataset build manifest '{}': {err}",
            output_dir.join("dataset_build_manifest.json").display()
        )
    })?;
    write_dataset_shard_manifests(output_dir, &build_manifest)?;
    if build_manifest.format == wx_export::ExportFormat::WebDataset {
        write_webdataset_shards(output_dir, &build_manifest)?;
    } else if build_manifest.format == wx_export::ExportFormat::Jsonl {
        write_jsonl_shards(output_dir, &build_manifest)?;
    } else if build_manifest.format == wx_export::ExportFormat::Parquet {
        write_parquet_shards(output_dir, &build_manifest)?;
    }

    println!("wrote {}", output_dir.display());
    println!("dataset={}", build_manifest.dataset_name);
    println!("samples={}", build_manifest.sample_count);
    println!("channels_total={}", build_manifest.total_channel_count);
    println!("shards={}", build_manifest.shard_count);
    println!("parallelism={parallelism}");
    Ok(())
}

fn write_dataset_shard_manifests(
    output_dir: &PathBuf,
    build_manifest: &DatasetBuildManifest,
) -> Result<(), String> {
    let shards_dir = output_dir.join("shards");
    fs::create_dir_all(&shards_dir).map_err(|err| {
        format!(
            "failed to create shards directory '{}': {err}",
            shards_dir.display()
        )
    })?;
    for shard_index in 0..build_manifest.shard_count {
        let shard_samples = build_manifest
            .samples
            .iter()
            .filter(|sample| sample.shard_index == shard_index)
            .cloned()
            .collect::<Vec<_>>();
        let shard_json = serde_json::json!({
            "dataset_name": build_manifest.dataset_name,
            "shard_index": shard_index,
            "sample_count": shard_samples.len(),
            "samples": shard_samples,
        });
        let shard_path = shards_dir.join(format!("shard-{shard_index:05}.json"));
        fs::write(
            &shard_path,
            serde_json::to_string_pretty(&shard_json)
                .map_err(|err| format!("failed to serialize shard manifest: {err}"))?,
        )
        .map_err(|err| {
            format!(
                "failed to write shard manifest '{}': {err}",
                shard_path.display()
            )
        })?;
    }
    Ok(())
}

fn build_dataset_samples_parallel(
    request: &GribDatasetBuildRequest,
    output_dir: &PathBuf,
    colormap: ColorMapArg,
    min: Option<f64>,
    max: Option<f64>,
    parallelism: usize,
    shard_count: usize,
    train_cutoff: usize,
    validation_cutoff: usize,
    requested_channels: Option<&[TrainingChannel]>,
) -> Result<Vec<BuiltSample>, String> {
    let next_index = Arc::new(Mutex::new(0usize));
    let results = Arc::new(Mutex::new(Vec::<BuiltSample>::with_capacity(
        request.samples.len(),
    )));
    let first_error = Arc::new(Mutex::new(None::<String>));
    let worker_count = parallelism.min(request.samples.len()).max(1);

    thread::scope(|scope| {
        for _ in 0..worker_count {
            let next_index = Arc::clone(&next_index);
            let results = Arc::clone(&results);
            let first_error = Arc::clone(&first_error);
            scope.spawn(move || loop {
                let index = {
                    let mut guard = next_index.lock().expect("index mutex poisoned");
                    let index = *guard;
                    if index >= request.samples.len() {
                        break;
                    }
                    *guard += 1;
                    index
                };

                if first_error
                    .lock()
                    .expect("error mutex poisoned")
                    .as_ref()
                    .is_some()
                {
                    break;
                }

                let sample = &request.samples[index];
                let file = PathBuf::from(&sample.file);
                let sample_id = sample
                    .sample_id
                    .clone()
                    .unwrap_or_else(|| format!("{:05}_{}", index + 1, safe_file_stem(&file)));
                let relative_dir = sanitize_filename_component(&sample_id);
                let sample_dir = output_dir.join(&relative_dir);
                let built = (|| -> Result<BuiltSample, String> {
                    let sample_manifest = if let Some(channels) = requested_channels {
                        feature_materialize::write_planned_training_sample(
                            sample,
                            &sample_dir,
                            request.dataset_name.clone(),
                            sample_id.clone(),
                            channels,
                            colormap.into(),
                            min,
                            max,
                        )?
                    } else {
                        write_grib_training_sample(
                            &file,
                            &sample_dir,
                            Some(request.dataset_name.clone()),
                            Some(sample_id.clone()),
                            sample.messages.clone(),
                            colormap,
                            min,
                            max,
                        )?
                    };
                    let split = if index < train_cutoff {
                        DatasetSplit::Train
                    } else if index < validation_cutoff.min(request.samples.len()) {
                        DatasetSplit::Validation
                    } else {
                        DatasetSplit::Test
                    };
                    let shard_index = index % shard_count;
                    let channels = sample_manifest
                        .channels
                        .iter()
                        .map(|channel| TrainingChannel {
                            name: channel.name.clone(),
                            units: channel.units.clone(),
                        })
                        .collect::<Vec<_>>();
                    Ok(BuiltSample {
                        index,
                        sample_ref: DatasetSampleRef {
                            sample_id,
                            source: sample.file.clone(),
                            relative_dir,
                            channel_count: sample_manifest.channel_count,
                            shard_index,
                            split,
                        },
                        channels,
                    })
                })();

                match built {
                    Ok(sample) => {
                        results.lock().expect("results mutex poisoned").push(sample);
                    }
                    Err(err) => {
                        let mut guard = first_error.lock().expect("error mutex poisoned");
                        if guard.is_none() {
                            *guard = Some(err);
                        }
                        break;
                    }
                }
            });
        }
    });

    if let Some(err) = first_error.lock().expect("error mutex poisoned").clone() {
        return Err(err);
    }
    let mut built = results.lock().expect("results mutex poisoned").clone();
    built.sort_by_key(|sample| sample.index);
    Ok(built)
}

fn write_webdataset_shards(
    output_dir: &PathBuf,
    build_manifest: &DatasetBuildManifest,
) -> Result<(), String> {
    let shards_dir = output_dir.join("shards");
    for shard_index in 0..build_manifest.shard_count {
        let shard_samples = build_manifest
            .samples
            .iter()
            .filter(|sample| sample.shard_index == shard_index)
            .collect::<Vec<_>>();
        let tar_path = shards_dir.join(format!("shard-{shard_index:05}.tar"));
        let mut tar_bytes = Vec::new();
        for sample in shard_samples {
            let sample_dir = output_dir.join(&sample.relative_dir);
            append_directory_to_tar(
                &mut tar_bytes,
                &sample_dir,
                &format!("{}/", sample.relative_dir),
            )?;
        }
        tar_bytes.extend_from_slice(&[0u8; 1024]);
        fs::write(&tar_path, tar_bytes).map_err(|err| {
            format!(
                "failed to write webdataset shard '{}': {err}",
                tar_path.display()
            )
        })?;
    }
    Ok(())
}

fn write_jsonl_shards(
    output_dir: &PathBuf,
    build_manifest: &DatasetBuildManifest,
) -> Result<(), String> {
    let shards_dir = output_dir.join("shards");
    for shard_index in 0..build_manifest.shard_count {
        let shard_samples = build_manifest
            .samples
            .iter()
            .filter(|sample| sample.shard_index == shard_index)
            .cloned()
            .collect::<Vec<_>>();
        let shard_path = shards_dir.join(format!("shard-{shard_index:05}.jsonl"));
        let mut lines = String::new();
        for sample in shard_samples {
            let sample_manifest_path = output_dir
                .join(&sample.relative_dir)
                .join("sample_manifest.json");
            let sample_manifest_text =
                fs::read_to_string(&sample_manifest_path).map_err(|err| {
                    format!(
                        "failed to read sample manifest '{}': {err}",
                        sample_manifest_path.display()
                    )
                })?;
            let sample_manifest_value: serde_json::Value =
                serde_json::from_str(&sample_manifest_text).map_err(|err| {
                    format!(
                        "failed to parse sample manifest '{}': {err}",
                        sample_manifest_path.display()
                    )
                })?;
            let row = serde_json::json!({
                "sample_id": sample.sample_id,
                "split": sample.split,
                "shard_index": sample.shard_index,
                "relative_dir": sample.relative_dir,
                "sample_manifest": sample_manifest_value,
            });
            lines.push_str(
                &serde_json::to_string(&row)
                    .map_err(|err| format!("failed to serialize jsonl row: {err}"))?,
            );
            lines.push('\n');
        }
        fs::write(&shard_path, lines).map_err(|err| {
            format!(
                "failed to write jsonl shard '{}': {err}",
                shard_path.display()
            )
        })?;
    }
    Ok(())
}

fn write_parquet_shards(
    output_dir: &PathBuf,
    build_manifest: &DatasetBuildManifest,
) -> Result<(), String> {
    let shards_dir = output_dir.join("shards");
    for shard_index in 0..build_manifest.shard_count {
        let shard_samples = build_manifest
            .samples
            .iter()
            .filter(|sample| sample.shard_index == shard_index)
            .cloned()
            .collect::<Vec<_>>();
        let rows = shard_samples
            .iter()
            .map(|sample| {
                let sample_manifest_path = output_dir
                    .join(&sample.relative_dir)
                    .join("sample_manifest.json");
                let sample_manifest_text =
                    fs::read_to_string(&sample_manifest_path).map_err(|err| {
                        format!(
                            "failed to read sample manifest '{}': {err}",
                            sample_manifest_path.display()
                        )
                    })?;
                Ok((
                    sample.sample_id.clone(),
                    format!("{:?}", sample.split),
                    sample.shard_index as i64,
                    sample.source.clone(),
                    sample.relative_dir.clone(),
                    sample.channel_count as i64,
                    sample_manifest_text,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let parquet_path = shards_dir.join(format!("shard-{shard_index:05}.parquet"));
        write_parquet_sample_rows(&parquet_path, &rows)?;
    }
    Ok(())
}

fn write_parquet_sample_rows(
    path: &PathBuf,
    rows: &[(String, String, i64, String, String, i64, String)],
) -> Result<(), String> {
    let schema = Arc::new(
        parse_message_type(
            "
        message wxtrain_dataset_shard {
          REQUIRED BYTE_ARRAY sample_id (STRING);
          REQUIRED BYTE_ARRAY split (STRING);
          REQUIRED INT64 shard_index;
          REQUIRED BYTE_ARRAY source (STRING);
          REQUIRED BYTE_ARRAY relative_dir (STRING);
          REQUIRED INT64 channel_count;
          REQUIRED BYTE_ARRAY sample_manifest_json (STRING);
        }
        ",
        )
        .map_err(|err| format!("failed to build parquet schema: {err}"))?,
    );
    let file = File::create(path)
        .map_err(|err| format!("failed to create parquet shard '{}': {err}", path.display()))?;
    let props = Arc::new(WriterProperties::builder().build());
    let mut writer = SerializedFileWriter::new(file, schema, props).map_err(|err| {
        format!(
            "failed to create parquet writer '{}': {err}",
            path.display()
        )
    })?;
    let mut row_group = writer
        .next_row_group()
        .map_err(|err| format!("failed to open row group '{}': {err}", path.display()))?;

    write_parquet_byte_array_column(
        &mut row_group,
        &rows
            .iter()
            .map(|row| ByteArray::from(row.0.as_str()))
            .collect::<Vec<_>>(),
    )?;
    write_parquet_byte_array_column(
        &mut row_group,
        &rows
            .iter()
            .map(|row| ByteArray::from(row.1.as_str()))
            .collect::<Vec<_>>(),
    )?;
    write_parquet_i64_column(
        &mut row_group,
        &rows.iter().map(|row| row.2).collect::<Vec<_>>(),
    )?;
    write_parquet_byte_array_column(
        &mut row_group,
        &rows
            .iter()
            .map(|row| ByteArray::from(row.3.as_str()))
            .collect::<Vec<_>>(),
    )?;
    write_parquet_byte_array_column(
        &mut row_group,
        &rows
            .iter()
            .map(|row| ByteArray::from(row.4.as_str()))
            .collect::<Vec<_>>(),
    )?;
    write_parquet_i64_column(
        &mut row_group,
        &rows.iter().map(|row| row.5).collect::<Vec<_>>(),
    )?;
    write_parquet_byte_array_column(
        &mut row_group,
        &rows
            .iter()
            .map(|row| ByteArray::from(row.6.as_str()))
            .collect::<Vec<_>>(),
    )?;

    row_group
        .close()
        .map_err(|err| format!("failed to close row group '{}': {err}", path.display()))?;
    writer
        .close()
        .map_err(|err| format!("failed to close parquet writer '{}': {err}", path.display()))?;
    Ok(())
}

fn write_parquet_byte_array_column(
    row_group: &mut SerializedRowGroupWriter<'_, File>,
    values: &[ByteArray],
) -> Result<(), String> {
    let Some(mut column) = row_group
        .next_column()
        .map_err(|err| format!("failed to open parquet column: {err}"))?
    else {
        return Err("parquet schema ended earlier than expected".to_string());
    };
    column
        .typed::<ByteArrayType>()
        .write_batch(values, None, None)
        .map_err(|err| format!("failed to write parquet byte-array column: {err}"))?;
    column
        .close()
        .map_err(|err| format!("failed to close parquet column: {err}"))?;
    Ok(())
}

fn write_parquet_i64_column(
    row_group: &mut SerializedRowGroupWriter<'_, File>,
    values: &[i64],
) -> Result<(), String> {
    let Some(mut column) = row_group
        .next_column()
        .map_err(|err| format!("failed to open parquet column: {err}"))?
    else {
        return Err("parquet schema ended earlier than expected".to_string());
    };
    column
        .typed::<Int64Type>()
        .write_batch(values, None, None)
        .map_err(|err| format!("failed to write parquet int64 column: {err}"))?;
    column
        .close()
        .map_err(|err| format!("failed to close parquet column: {err}"))?;
    Ok(())
}

fn append_directory_to_tar(
    tar_bytes: &mut Vec<u8>,
    dir: &PathBuf,
    tar_prefix: &str,
) -> Result<(), String> {
    for entry in fs::read_dir(dir)
        .map_err(|err| format!("failed to read directory '{}': {err}", dir.display()))?
    {
        let entry = entry.map_err(|err| format!("failed to read dir entry: {err}"))?;
        let path = entry.path();
        let name = format!("{tar_prefix}{}", entry.file_name().to_string_lossy());
        if path.is_dir() {
            append_directory_to_tar(tar_bytes, &path, &(name + "/"))?;
        } else {
            let file_bytes = fs::read(&path)
                .map_err(|err| format!("failed to read file '{}': {err}", path.display()))?;
            append_tar_file(tar_bytes, &name, &file_bytes)?;
        }
    }
    Ok(())
}

fn append_tar_file(tar_bytes: &mut Vec<u8>, name: &str, contents: &[u8]) -> Result<(), String> {
    if name.len() > 100 {
        return Err(format!("tar path too long for minimal writer: '{name}'"));
    }
    let mut header = [0u8; 512];
    write_tar_string(&mut header[0..100], name);
    write_tar_octal(&mut header[100..108], 0o644);
    write_tar_octal(&mut header[108..116], 0);
    write_tar_octal(&mut header[116..124], 0);
    write_tar_octal(&mut header[124..136], contents.len() as u64);
    write_tar_octal(&mut header[136..148], 0);
    for byte in &mut header[148..156] {
        *byte = b' ';
    }
    header[156] = b'0';
    header[257..263].copy_from_slice(b"ustar\0");
    header[263..265].copy_from_slice(b"00");
    let checksum: u32 = header.iter().map(|&byte| byte as u32).sum();
    write_tar_checksum(&mut header[148..156], checksum);
    tar_bytes.extend_from_slice(&header);
    tar_bytes.extend_from_slice(contents);
    let pad = (512 - (contents.len() % 512)) % 512;
    if pad > 0 {
        tar_bytes.extend_from_slice(&vec![0u8; pad]);
    }
    Ok(())
}

fn write_tar_string(dst: &mut [u8], value: &str) {
    let bytes = value.as_bytes();
    let len = bytes.len().min(dst.len());
    dst[..len].copy_from_slice(&bytes[..len]);
}

fn write_tar_octal(dst: &mut [u8], value: u64) {
    let width = dst.len();
    let octal = format!("{value:o}");
    let start = width.saturating_sub(octal.len() + 1);
    for byte in &mut *dst {
        *byte = b'0';
    }
    dst[width - 1] = 0;
    dst[start..start + octal.len()].copy_from_slice(octal.as_bytes());
}

fn write_tar_checksum(dst: &mut [u8], value: u32) {
    let octal = format!("{value:06o}\0 ");
    dst.copy_from_slice(octal.as_bytes());
}

fn safe_file_stem(path: &PathBuf) -> String {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .map(sanitize_filename_component)
        .unwrap_or_else(|| "sample".to_string())
}

fn sanitize_filename_component(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut prev_was_sep = false;
    for ch in raw.chars() {
        let safe = if ch.is_ascii_alphanumeric() {
            ch.to_ascii_lowercase()
        } else {
            '_'
        };
        if safe == '_' {
            if !prev_was_sep {
                out.push('_');
                prev_was_sep = true;
            }
        } else {
            out.push(safe);
            prev_was_sep = false;
        }
    }
    out.trim_matches('_').to_string()
}

fn fetch_subset_from_locations(
    grib_location: &str,
    idx_location: &str,
    search: &str,
    output: &PathBuf,
    limit: Option<usize>,
) -> Result<(), String> {
    if idx_location.trim().is_empty() {
        return Err("this source does not expose a directly queryable inventory URL".to_string());
    }
    let fetch = FetchEngine::new();
    let grib = GribEngine::new();
    let idx_text = fetch.read_text(idx_location)?;
    let inventory = grib.parse_idx_text(&idx_text)?;
    let mut matches = grib.search(&inventory, search);
    if let Some(limit) = limit {
        matches.truncate(limit);
    }
    if matches.is_empty() {
        return Err(format!(
            "no messages matched '{search}' in inventory '{idx_location}'"
        ));
    }

    let needs_total_length = matches.iter().any(|message| message.length_bytes == 0);
    let total_length = if needs_total_length {
        Some(fetch.content_length(grib_location)?)
    } else {
        None
    };

    let ranges = matches
        .iter()
        .map(|message| {
            resolve_offset_length(message.offset_bytes, message.length_bytes, total_length)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let bytes = fetch.fetch_ranges(grib_location, &ranges)?;
    fs::write(output, &bytes)
        .map_err(|err| format!("failed to write subset '{}': {err}", output.display()))?;

    println!("wrote {}", output.display());
    println!("matched_messages={}", matches.len());
    println!("bytes={}", bytes.len());
    for message in matches {
        println!(
            "#{:>3} ({}) off={} len={} {} {}",
            message.message_no,
            message.message_id,
            message.offset_bytes,
            message.length_bytes,
            message.variable,
            message.level
        );
    }
    Ok(())
}

/// Compute full channel statistics (min, max, mean, std, count, nan_count) from a Grid2D.
fn compute_channel_stats(grid: &Grid2D) -> Option<ChannelStats> {
    let total = grid.values.len();
    let mut count = 0usize;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;

    for value in &grid.values {
        if value.is_nan() {
            continue;
        }
        count += 1;
        sum += *value;
        sum_sq += *value * *value;
        min = min.min(*value);
        max = max.max(*value);
    }

    if count == 0 {
        return None;
    }

    let mean = sum / count as f64;
    let variance = (sum_sq / count as f64) - (mean * mean);
    let std = if variance > 0.0 { variance.sqrt() } else { 0.0 };
    let nan_count = total - count;

    Some(ChannelStats {
        min,
        mean,
        max,
        std,
        count,
        nan_count,
    })
}

/// Write `channel_stats.json` alongside the sample output for training normalization.
fn write_channel_stats_json(
    output_dir: &PathBuf,
    artifacts: &[SampleChannelArtifact],
) -> Result<(), String> {
    let entries: Vec<ChannelNormEntry> = artifacts
        .iter()
        .filter_map(|artifact| {
            artifact.stats.as_ref().map(|stats| ChannelNormEntry {
                name: artifact.name.clone(),
                units: artifact.units.clone(),
                min: stats.min,
                max: stats.max,
                mean: stats.mean,
                std: stats.std,
                count: stats.count,
                nan_count: stats.nan_count,
            })
        })
        .collect();

    let norm_stats = ChannelNormStats { channels: entries };
    let json = serde_json::to_string_pretty(&norm_stats)
        .map_err(|err| format!("failed to serialize channel_stats.json: {err}"))?;
    let path = output_dir.join("channel_stats.json");
    fs::write(&path, json)
        .map_err(|err| format!("failed to write '{}': {err}", path.display()))
}
