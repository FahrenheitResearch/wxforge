//! Source planning and fetch contracts.

use std::env;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::process::Command;
use std::thread;
use std::time::Duration;

use chrono::{DateTime, Timelike, Utc};
use reqwest::blocking::Client;
use reqwest::header::{CONTENT_LENGTH, RANGE};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceKind {
    Nomads,
    Aws,
    Unidata,
    Ecmwf,
    Cds,
    LocalFilesystem,
    Custom(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelKind {
    Hrrr,
    Gfs,
    Nam,
    Rap,
    EcmwfIfs,
    Era5,
}

impl ModelKind {
    pub fn cadence_hours(self) -> u32 {
        match self {
            Self::Hrrr | Self::Rap => 1,
            Self::Gfs | Self::Nam | Self::EcmwfIfs => 6,
            Self::Era5 => 1,
        }
    }

    pub fn default_source(self) -> SourceKind {
        match self {
            Self::Hrrr | Self::Gfs | Self::Nam | Self::Rap => SourceKind::Nomads,
            Self::EcmwfIfs => SourceKind::Ecmwf,
            Self::Era5 => SourceKind::Cds,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProductKind {
    Surface,
    Pressure,
    Native,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ByteRange {
    pub start: u64,
    pub end: u64,
}

impl ByteRange {
    pub fn len(&self) -> u64 {
        self.end.saturating_sub(self.start) + 1
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FetchRequest {
    pub model: ModelKind,
    pub run_time: DateTime<Utc>,
    pub product: ProductKind,
    pub forecast_hour: u16,
    pub source: Option<SourceKind>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FetchPlan {
    pub source: SourceKind,
    pub model: ModelKind,
    pub product: ProductKind,
    pub grib_url: String,
    pub idx_url: String,
    pub file_name: String,
    pub ranges: Vec<ByteRange>,
    pub requires_auth: bool,
    pub request_body_json: Option<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ModelDownloadOptions {
    pub variables: Vec<String>,
    pub pressure_levels: Vec<String>,
    pub area: Option<[f64; 4]>,
    pub grid: Option<[f64; 2]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CdsCredentials {
    pub url: String,
    pub key: String,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DownloadReceipt {
    pub bytes: u64,
    pub source: SourceKind,
    pub location: String,
    pub dataset: Option<String>,
    pub job_id: Option<String>,
}

impl CdsCredentials {
    fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }
}

#[derive(Clone)]
pub struct FetchEngine {
    client: Client,
}

impl std::fmt::Debug for FetchEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FetchEngine").finish_non_exhaustive()
    }
}

impl Default for FetchEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl FetchEngine {
    pub fn new() -> Self {
        let client = Client::builder()
            .user_agent("wxforge/0.1")
            .build()
            .expect("reqwest client should initialize");
        Self { client }
    }

    pub fn design_goal(&self) -> &'static str {
        "Parallel, byte-range aware model and archive fetch planning in pure Rust."
    }

    pub fn supports_parallel_ranges(&self) -> bool {
        true
    }

    pub fn latest_cycle_for(&self, model: ModelKind, now: DateTime<Utc>) -> DateTime<Utc> {
        let cadence = model.cadence_hours();
        let rounded_hour = now.hour() - (now.hour() % cadence);
        now.with_minute(0)
            .and_then(|dt| dt.with_second(0))
            .and_then(|dt| dt.with_nanosecond(0))
            .and_then(|dt| dt.with_hour(rounded_hour))
            .expect("valid UTC cycle time")
    }

    pub fn latest_available_cycle_for(
        &self,
        model: ModelKind,
        product: ProductKind,
        source: &SourceKind,
        now: DateTime<Utc>,
        forecast_hour: u16,
        max_lookback_cycles: usize,
    ) -> Result<DateTime<Utc>, String> {
        if matches!(source, SourceKind::Cds) {
            return Ok(self.latest_cycle_for(model, now));
        }
        validate_source_model(source, model)?;
        let cadence = model.cadence_hours() as i64;
        let mut cycle = self.latest_cycle_for(model, now);
        for _ in 0..max_lookback_cycles.max(1) {
            let grib_url = build_url(source, model, product, cycle, forecast_hour)?;
            let idx_url = build_inventory_url(source, model, &grib_url)?;
            if self.resource_exists(&idx_url)? || self.resource_exists(&grib_url)? {
                return Ok(cycle);
            }
            cycle -= chrono::Duration::hours(cadence);
        }

        Err(format!(
            "no available cycle found for {:?} {:?} within {} backsteps",
            model, source, max_lookback_cycles
        ))
    }

    pub fn plan(&self, request: &FetchRequest) -> Result<FetchPlan, String> {
        let source = request
            .source
            .clone()
            .unwrap_or_else(|| request.model.default_source());
        validate_source_model(&source, request.model)?;
        let file_name = build_file_name(
            request.model,
            request.product,
            request.run_time,
            request.forecast_hour,
        );
        let grib_url = build_url(
            &source,
            request.model,
            request.product,
            request.run_time,
            request.forecast_hour,
        )?;
        let idx_url = build_inventory_url(&source, request.model, &grib_url)?;
        let requires_auth = matches!(source, SourceKind::Cds);
        let request_body_json = build_request_body_json(
            &source,
            request.model,
            request.product,
            request.run_time,
            request.forecast_hour,
        );
        let notes = build_fetch_notes(&source, request.model);

        Ok(FetchPlan {
            source,
            model: request.model,
            product: request.product,
            grib_url,
            idx_url,
            file_name,
            ranges: Vec::new(),
            requires_auth,
            request_body_json,
            notes,
        })
    }

    pub fn execute_plan_to_file(
        &self,
        plan: &FetchPlan,
        output: impl AsRef<Path>,
        options: &ModelDownloadOptions,
    ) -> Result<DownloadReceipt, String> {
        let output = output.as_ref();
        match plan.source {
            SourceKind::Cds => self.retrieve_cds_plan_to_file(plan, output, options),
            _ => {
                let bytes = self.download_to_file(&plan.grib_url, output)?;
                Ok(DownloadReceipt {
                    bytes,
                    source: plan.source.clone(),
                    location: plan.grib_url.clone(),
                    dataset: None,
                    job_id: None,
                })
            }
        }
    }

    pub fn load_cds_credentials(&self) -> Result<CdsCredentials, String> {
        if let Some(credentials) =
            credentials_from_env("ECMWF_DATASTORES_URL", "ECMWF_DATASTORES_KEY")
        {
            return Ok(credentials.with_source("env:ECMWF_DATASTORES_URL/KEY"));
        }
        if let Some(credentials) = credentials_from_env("CDSAPI_URL", "CDSAPI_KEY") {
            return Ok(credentials.with_source("env:CDSAPI_URL/KEY"));
        }

        let mut candidates = home_candidate_paths(".ecmwfdatastoresrc");
        candidates.extend(home_candidate_paths(".cdsapirc"));
        for path in candidates {
            if let Ok(text) = fs::read_to_string(&path) {
                if let Some(credentials) = parse_cds_credentials_text(&text) {
                    return Ok(credentials.with_source(format!("file:{}", path.display())));
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            for path in ["~/.ecmwfdatastoresrc", "~/.cdsapirc"] {
                if let Some(text) = read_wsl_text(path) {
                    if let Some(credentials) = parse_cds_credentials_text(&text) {
                        return Ok(credentials.with_source(format!("wsl:{path}")));
                    }
                }
            }
        }

        Err(
            "could not find CDS credentials in env, local config files, or WSL config files"
                .to_string(),
        )
    }

    fn retrieve_cds_plan_to_file(
        &self,
        plan: &FetchPlan,
        output: &Path,
        options: &ModelDownloadOptions,
    ) -> Result<DownloadReceipt, String> {
        let credentials = self.load_cds_credentials()?;
        let dataset = dataset_from_cds_url(&plan.grib_url)?;
        let request = build_cds_request_from_plan(plan, options)?;
        let base_url = credentials.url.trim_end_matches('/');
        let submit_url = format!("{base_url}/retrieve/v1/processes/{dataset}/execution");
        let submit = self.cds_request_json(
            reqwest::Method::POST,
            &submit_url,
            &credentials,
            Some(json!({ "inputs": request })),
        )?;
        let job_id = submit
            .get("jobID")
            .and_then(Value::as_str)
            .ok_or_else(|| "CDS submit response is missing jobID".to_string())?
            .to_string();
        let monitor_url = link_href(&submit, "monitor")
            .unwrap_or_else(|| format!("{base_url}/retrieve/v1/jobs/{job_id}"));
        let final_status = self.poll_cds_job(&monitor_url, &credentials)?;
        if final_status != "successful" {
            return Err(format!(
                "CDS job {job_id} finished with status {final_status}"
            ));
        }

        let results_url = format!("{monitor_url}/results");
        let results =
            self.cds_request_json(reqwest::Method::GET, &results_url, &credentials, None)?;
        let result_url = results
            .pointer("/asset/value/href")
            .and_then(Value::as_str)
            .ok_or_else(|| "CDS results response is missing asset/value/href".to_string())?
            .to_string();
        let bytes = self.download_to_file(&result_url, output)?;
        Ok(DownloadReceipt {
            bytes,
            source: plan.source.clone(),
            location: result_url,
            dataset: Some(dataset),
            job_id: Some(job_id),
        })
    }

    fn poll_cds_job(
        &self,
        monitor_url: &str,
        credentials: &CdsCredentials,
    ) -> Result<String, String> {
        let mut sleep = Duration::from_secs(1);
        let max_sleep = Duration::from_secs(15);
        for _ in 0..120 {
            let response =
                self.cds_request_json(reqwest::Method::GET, monitor_url, credentials, None)?;
            let status = response
                .get("status")
                .and_then(Value::as_str)
                .ok_or_else(|| "CDS job response is missing status".to_string())?;
            match status {
                "successful" => return Ok(status.to_string()),
                "accepted" | "running" => {
                    thread::sleep(sleep);
                    sleep = (sleep + sleep / 2).min(max_sleep);
                }
                "failed" | "rejected" => {
                    return Err(format!(
                        "CDS job failed: {}",
                        response
                            .pointer("/error/reason")
                            .and_then(Value::as_str)
                            .unwrap_or(status)
                    ));
                }
                other => {
                    return Err(format!("CDS job entered unexpected status '{other}'"));
                }
            }
        }
        Err("CDS job polling timed out".to_string())
    }

    fn cds_request_json(
        &self,
        method: reqwest::Method,
        url: &str,
        credentials: &CdsCredentials,
        json_body: Option<Value>,
    ) -> Result<Value, String> {
        let mut request = self
            .client
            .request(method, url)
            .header("PRIVATE-TOKEN", credentials.key.as_str());
        if let Some(json_body) = json_body {
            request = request.json(&json_body);
        }
        let response = request
            .send()
            .map_err(|err| format!("failed to call CDS endpoint '{url}': {err}"))?;
        let status = response.status();
        let body = response
            .text()
            .map_err(|err| format!("failed to read CDS response body from '{url}': {err}"))?;
        if !status.is_success() {
            return Err(format!("CDS endpoint '{url}' returned {status}: {body}"));
        }
        serde_json::from_str(&body)
            .map_err(|err| format!("failed to parse CDS JSON response from '{url}': {err}"))
    }

    pub fn read_text(&self, location: &str) -> Result<String, String> {
        if is_remote(location) {
            let response = self
                .client
                .get(location)
                .send()
                .map_err(|err| format!("failed to GET '{location}': {err}"))?
                .error_for_status()
                .map_err(|err| format!("bad response from '{location}': {err}"))?;
            response
                .text()
                .map_err(|err| format!("failed to read text body from '{location}': {err}"))
        } else {
            fs::read_to_string(location)
                .map_err(|err| format!("failed to read text '{}': {err}", location))
        }
    }

    pub fn download_to_file(
        &self,
        location: &str,
        output: impl AsRef<Path>,
    ) -> Result<u64, String> {
        if is_remote(location) {
            let mut response = self
                .client
                .get(location)
                .send()
                .map_err(|err| format!("failed to GET '{location}': {err}"))?
                .error_for_status()
                .map_err(|err| format!("bad response from '{location}': {err}"))?;
            let mut file = fs::File::create(output.as_ref()).map_err(|err| {
                format!(
                    "failed to create download target '{}': {err}",
                    output.as_ref().display()
                )
            })?;
            response.copy_to(&mut file).map_err(|err| {
                format!(
                    "failed to stream download '{}' to '{}': {err}",
                    location,
                    output.as_ref().display()
                )
            })
        } else {
            fs::copy(location, output.as_ref()).map_err(|err| {
                format!(
                    "failed to copy local download '{}' to '{}': {err}",
                    location,
                    output.as_ref().display()
                )
            })
        }
    }

    pub fn read_bytes(&self, location: &str) -> Result<Vec<u8>, String> {
        if is_remote(location) {
            let response = self
                .client
                .get(location)
                .send()
                .map_err(|err| format!("failed to GET '{location}': {err}"))?
                .error_for_status()
                .map_err(|err| format!("bad response from '{location}': {err}"))?;
            response
                .bytes()
                .map(|body| body.to_vec())
                .map_err(|err| format!("failed to read bytes from '{location}': {err}"))
        } else {
            fs::read(location).map_err(|err| format!("failed to read bytes '{}': {err}", location))
        }
    }

    pub fn content_length(&self, location: &str) -> Result<u64, String> {
        if is_remote(location) {
            let response = self
                .client
                .head(location)
                .send()
                .map_err(|err| format!("failed to HEAD '{location}': {err}"))?;
            if let Some(length) = header_to_u64(response.headers().get(CONTENT_LENGTH))? {
                return Ok(length);
            }

            let response = self
                .client
                .get(location)
                .header(RANGE, "bytes=0-0")
                .send()
                .map_err(|err| format!("failed ranged GET '{location}': {err}"))?;

            if let Some(length) = header_to_u64(response.headers().get(CONTENT_LENGTH))? {
                return Ok(length);
            }

            let body = response
                .bytes()
                .map_err(|err| format!("failed to read fallback GET from '{location}': {err}"))?;
            Ok(body.len() as u64)
        } else {
            fs::metadata(location)
                .map_err(|err| format!("failed to stat '{}': {err}", location))?
                .len()
                .try_into()
                .map_err(|_| format!("file '{}' is too large to index", location))
        }
    }

    pub fn fetch_range(&self, location: &str, range: &ByteRange) -> Result<Vec<u8>, String> {
        if range.end < range.start {
            return Err(format!("invalid byte range {}-{}", range.start, range.end));
        }

        if is_remote(location) {
            let response = self
                .client
                .get(location)
                .header(RANGE, format!("bytes={}-{}", range.start, range.end))
                .send()
                .map_err(|err| format!("failed to fetch byte range from '{location}': {err}"))?
                .error_for_status()
                .map_err(|err| format!("bad range response from '{location}': {err}"))?;
            let bytes = response
                .bytes()
                .map_err(|err| format!("failed to read ranged body from '{location}': {err}"))?
                .to_vec();
            let expected = range.len() as usize;
            if bytes.len() != expected {
                return Err(format!(
                    "range {}-{} from '{location}' returned {} bytes, expected {}",
                    range.start,
                    range.end,
                    bytes.len(),
                    expected
                ));
            }
            Ok(bytes)
        } else {
            let mut file = fs::File::open(location)
                .map_err(|err| format!("failed to open '{}': {err}", location))?;
            file.seek(SeekFrom::Start(range.start))
                .map_err(|err| format!("failed to seek '{}': {err}", location))?;
            let mut bytes = vec![0u8; range.len() as usize];
            file.read_exact(&mut bytes)
                .map_err(|err| format!("failed to read range from '{}': {err}", location))?;
            Ok(bytes)
        }
    }

    pub fn fetch_ranges(&self, location: &str, ranges: &[ByteRange]) -> Result<Vec<u8>, String> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }

        let coalesced = coalesce_ranges(ranges);
        if is_remote(location) && coalesced.len() > 1 {
            let mut handles = Vec::with_capacity(coalesced.len());
            for (idx, range) in coalesced.iter().cloned().enumerate() {
                let location = location.to_string();
                let engine = self.clone();
                handles.push(thread::spawn(move || {
                    engine
                        .fetch_range(&location, &range)
                        .map(|bytes| (idx, bytes))
                }));
            }

            let mut parts = vec![Vec::new(); coalesced.len()];
            for handle in handles {
                let (idx, bytes) = handle
                    .join()
                    .map_err(|_| "parallel range fetch thread panicked".to_string())??;
                parts[idx] = bytes;
            }
            Ok(parts.into_iter().flatten().collect())
        } else {
            let mut out = Vec::new();
            for range in &coalesced {
                out.extend_from_slice(&self.fetch_range(location, range)?);
            }
            Ok(out)
        }
    }

    pub fn resource_exists(&self, location: &str) -> Result<bool, String> {
        if is_remote(location) {
            let response = self
                .client
                .head(location)
                .send()
                .map_err(|err| format!("failed to probe '{location}': {err}"))?;
            if response.status().is_success() {
                return Ok(true);
            }
            if response.status().as_u16() == 403 || response.status().as_u16() == 405 {
                let response = self
                    .client
                    .get(location)
                    .header(RANGE, "bytes=0-0")
                    .send()
                    .map_err(|err| format!("failed ranged probe for '{location}': {err}"))?;
                return Ok(response.status().is_success() || response.status().as_u16() == 206);
            }
            Ok(false)
        } else {
            Ok(Path::new(location).exists())
        }
    }
}

pub fn coalesce_ranges(ranges: &[ByteRange]) -> Vec<ByteRange> {
    if ranges.is_empty() {
        return Vec::new();
    }

    let mut sorted = ranges.to_vec();
    sorted.sort_by_key(|range| range.start);
    let mut merged = Vec::with_capacity(sorted.len());
    let mut current = sorted[0].clone();

    for range in sorted.into_iter().skip(1) {
        if range.start <= current.end.saturating_add(1) {
            current.end = current.end.max(range.end);
        } else {
            merged.push(current);
            current = range;
        }
    }
    merged.push(current);
    merged
}

pub fn resolve_offset_length(
    offset: u64,
    length_bytes: u64,
    total_length: Option<u64>,
) -> Result<ByteRange, String> {
    if length_bytes > 0 {
        return Ok(ByteRange {
            start: offset,
            end: offset + length_bytes - 1,
        });
    }

    let total_length = total_length
        .ok_or_else(|| format!("message at offset {offset} has unknown trailing length"))?;
    if total_length <= offset {
        return Err(format!(
            "content length {total_length} is not larger than offset {offset}"
        ));
    }
    Ok(ByteRange {
        start: offset,
        end: total_length - 1,
    })
}

pub fn build_file_name(
    model: ModelKind,
    product: ProductKind,
    run_time: DateTime<Utc>,
    forecast_hour: u16,
) -> String {
    let hour = run_time.format("%H");
    let stamp = run_time.format("%Y%m%d%H0000");
    match (model, product) {
        (ModelKind::Hrrr, ProductKind::Surface) => {
            format!("hrrr.t{hour}z.wrfsfcf{forecast_hour:02}.grib2")
        }
        (ModelKind::Hrrr, ProductKind::Pressure) => {
            format!("hrrr.t{hour}z.wrfprsf{forecast_hour:02}.grib2")
        }
        (ModelKind::Hrrr, ProductKind::Native) => {
            format!("hrrr.t{hour}z.wrfsfcf{forecast_hour:02}.grib2")
        }
        (ModelKind::Gfs, _) => format!("gfs.t{hour}z.pgrb2.0p25.f{forecast_hour:03}"),
        (ModelKind::Nam, ProductKind::Surface) => {
            format!("nam.t{hour}z.awphys{forecast_hour:02}.tm00.grib2")
        }
        (ModelKind::Nam, ProductKind::Pressure) => {
            format!("nam.t{hour}z.awip32{forecast_hour:02}.tm00.grib2")
        }
        (ModelKind::Nam, ProductKind::Native) => {
            format!("nam.t{hour}z.awphys{forecast_hour:02}.tm00.grib2")
        }
        (ModelKind::Rap, _) => format!("rap.t{hour}z.awp130bgrbf{forecast_hour:02}.grib2"),
        (ModelKind::EcmwfIfs, _) => {
            let stream = ecmwf_stream_for_cycle(run_time);
            format!("{stamp}-{forecast_hour}h-{stream}-fc.grib2")
        }
        (ModelKind::Era5, ProductKind::Surface) => {
            format!(
                "era5-single-levels-{}-f{forecast_hour:03}.grib",
                run_time.format("%Y%m%d%H")
            )
        }
        (ModelKind::Era5, ProductKind::Pressure) => {
            format!(
                "era5-pressure-levels-{}-f{forecast_hour:03}.grib",
                run_time.format("%Y%m%d%H")
            )
        }
        (ModelKind::Era5, ProductKind::Native) => {
            format!(
                "era5-{}-f{forecast_hour:03}.grib",
                run_time.format("%Y%m%d%H")
            )
        }
    }
}

pub fn supported_sources_for(model: ModelKind) -> &'static [SourceKind] {
    match model {
        ModelKind::Hrrr => &[SourceKind::Nomads, SourceKind::Aws],
        ModelKind::Gfs => &[SourceKind::Nomads, SourceKind::Aws, SourceKind::Unidata],
        ModelKind::Nam => &[SourceKind::Nomads, SourceKind::Aws],
        ModelKind::Rap => &[SourceKind::Nomads, SourceKind::Aws],
        ModelKind::EcmwfIfs => &[SourceKind::Ecmwf],
        ModelKind::Era5 => &[SourceKind::Cds],
    }
}

pub fn validate_source_model(source: &SourceKind, model: ModelKind) -> Result<(), String> {
    if matches!(source, SourceKind::Custom(_) | SourceKind::LocalFilesystem) {
        return Ok(());
    }
    if supported_sources_for(model)
        .iter()
        .any(|item| item == source)
    {
        Ok(())
    } else {
        Err(format!(
            "source {:?} is not supported for model {:?}; supported sources: {}",
            source,
            model,
            supported_sources_for(model)
                .iter()
                .map(|item| format!("{item:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }
}

pub fn build_url(
    source: &SourceKind,
    model: ModelKind,
    product: ProductKind,
    run_time: DateTime<Utc>,
    forecast_hour: u16,
) -> Result<String, String> {
    validate_source_model(source, model)?;
    let ymd = run_time.format("%Y%m%d").to_string();
    let hour = run_time.format("%H").to_string();
    let file = build_file_name(model, product, run_time, forecast_hour);
    let ecmwf_stream = ecmwf_stream_for_cycle(run_time);

    Ok(match (source, model) {
        (SourceKind::Nomads, ModelKind::Hrrr) => {
            format!(
                "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.{ymd}/conus/{file}"
            )
        }
        (SourceKind::Nomads, ModelKind::Gfs) => {
            format!(
                "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{ymd}/{hour}/atmos/{file}"
            )
        }
        (SourceKind::Nomads, ModelKind::Nam) => {
            format!("https://nomads.ncep.noaa.gov/pub/data/nccf/com/nam/prod/nam.{ymd}/{file}")
        }
        (SourceKind::Nomads, ModelKind::Rap) => {
            format!("https://nomads.ncep.noaa.gov/pub/data/nccf/com/rap/prod/rap.{ymd}/{file}")
        }
        (SourceKind::Aws, ModelKind::Hrrr) => {
            format!("https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{ymd}/conus/{file}")
        }
        (SourceKind::Aws, ModelKind::Gfs) => {
            format!("https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{ymd}/{hour}/atmos/{file}")
        }
        (SourceKind::Aws, ModelKind::Nam) => {
            format!("https://noaa-nam-pds.s3.amazonaws.com/nam.{ymd}/{file}")
        }
        (SourceKind::Aws, ModelKind::Rap) => {
            format!("https://noaa-rap-pds.s3.amazonaws.com/rap.{ymd}/{file}")
        }
        (SourceKind::Ecmwf, ModelKind::EcmwfIfs) => {
            format!("https://data.ecmwf.int/forecasts/{ymd}/{hour}z/ifs/0p25/{ecmwf_stream}/{file}")
        }
        (SourceKind::Cds, ModelKind::Era5) => match product {
            ProductKind::Surface => "cds://reanalysis-era5-single-levels".to_string(),
            ProductKind::Pressure => "cds://reanalysis-era5-pressure-levels".to_string(),
            ProductKind::Native => "cds://reanalysis-era5-complete".to_string(),
        },
        (SourceKind::Custom(base), _) => format!("{base}/{file}"),
        (SourceKind::Unidata, ModelKind::Gfs) => {
            format!("https://thredds.ucar.edu/fileServer/grib/NCEP/GFS/Global_0p25deg/{file}")
        }
        (SourceKind::LocalFilesystem, _) => file,
        _ => unreachable!("unsupported source/model combinations are validated above"),
    })
}

pub fn build_inventory_url(
    source: &SourceKind,
    model: ModelKind,
    grib_url: &str,
) -> Result<String, String> {
    validate_source_model(source, model)?;
    Ok(match (source, model) {
        (SourceKind::Ecmwf, ModelKind::EcmwfIfs) => grib_url
            .strip_suffix(".grib2")
            .map(|prefix| format!("{prefix}.index"))
            .unwrap_or_else(|| format!("{grib_url}.index")),
        (SourceKind::Cds, ModelKind::Era5) => String::new(),
        _ => format!("{grib_url}.idx"),
    })
}

fn ecmwf_stream_for_cycle(run_time: DateTime<Utc>) -> &'static str {
    match run_time.hour() {
        0 | 12 => "oper",
        6 | 18 => "scda",
        _ => "oper",
    }
}

fn build_request_body_json(
    source: &SourceKind,
    model: ModelKind,
    product: ProductKind,
    run_time: DateTime<Utc>,
    forecast_hour: u16,
) -> Option<String> {
    match (source, model, product) {
        (SourceKind::Cds, ModelKind::Era5, ProductKind::Surface) => Some(format!(
            "{{\"product_type\":[\"reanalysis\"],\"data_format\":\"grib\",\"date\":\"{}\",\"time\":\"{:02}:00\",\"variable\":[],\"download_format\":\"unarchived\"}}",
            run_time.format("%Y-%m-%d"),
            run_time.hour()
        )),
        (SourceKind::Cds, ModelKind::Era5, ProductKind::Pressure) => Some(format!(
            "{{\"product_type\":[\"reanalysis\"],\"data_format\":\"grib\",\"date\":\"{}\",\"time\":\"{:02}:00\",\"pressure_level\":[],\"variable\":[],\"download_format\":\"unarchived\"}}",
            run_time.format("%Y-%m-%d"),
            run_time.hour()
        )),
        (SourceKind::Cds, ModelKind::Era5, ProductKind::Native) => Some(format!(
            "{{\"dataset\":\"reanalysis-era5-complete\",\"date\":\"{}\",\"time\":\"{:02}:00\",\"step\":\"{}\",\"format\":\"grib\"}}",
            run_time.format("%Y-%m-%d"),
            run_time.hour(),
            forecast_hour
        )),
        _ => None,
    }
}

fn build_fetch_notes(source: &SourceKind, model: ModelKind) -> Vec<String> {
    match (source, model) {
        (SourceKind::Ecmwf, ModelKind::EcmwfIfs) => vec![
            "ECMWF open data uses .index JSON-lines inventories".to_string(),
            "Recent ECMWF open-data GRIB2 commonly uses CCSDS compression; decoder support may be required".to_string(),
        ],
        (SourceKind::Cds, ModelKind::Era5) => vec![
            "ERA5 retrieval normally requires CDS API credentials and asynchronous archive access".to_string(),
            "Use request_body_json as the seed for a Rust-native CDS client".to_string(),
        ],
        _ => Vec::new(),
    }
}

fn dataset_from_cds_url(url: &str) -> Result<String, String> {
    url.strip_prefix("cds://")
        .map(str::to_string)
        .ok_or_else(|| format!("expected a cds:// URL, got '{url}'"))
}

fn build_cds_request_from_plan(
    plan: &FetchPlan,
    options: &ModelDownloadOptions,
) -> Result<Value, String> {
    let raw = plan
        .request_body_json
        .as_deref()
        .ok_or_else(|| "CDS plan is missing request_body_json".to_string())?;
    let mut request: Value = serde_json::from_str(raw)
        .map_err(|err| format!("failed to parse request_body_json: {err}"))?;
    let object = request
        .as_object_mut()
        .ok_or_else(|| "CDS request body must be a JSON object".to_string())?;

    if !options.variables.is_empty() {
        object.insert(
            "variable".to_string(),
            Value::Array(
                options
                    .variables
                    .iter()
                    .map(|item| Value::String(item.clone()))
                    .collect(),
            ),
        );
    }
    if !options.pressure_levels.is_empty() {
        object.insert(
            "pressure_level".to_string(),
            Value::Array(
                options
                    .pressure_levels
                    .iter()
                    .map(|item| Value::String(item.clone()))
                    .collect(),
            ),
        );
    }
    if let Some(area) = options.area {
        object.insert(
            "area".to_string(),
            Value::Array(area.into_iter().map(Value::from).collect()),
        );
    }
    if let Some(grid) = options.grid {
        object.insert(
            "grid".to_string(),
            Value::Array(grid.into_iter().map(Value::from).collect()),
        );
    }

    let variable_count = object
        .get("variable")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    if !matches!(plan.product, ProductKind::Native) && variable_count == 0 {
        return Err("ERA5 retrieval requires at least one variable".to_string());
    }

    if matches!(plan.product, ProductKind::Pressure) {
        let level_count = object
            .get("pressure_level")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        if level_count == 0 {
            return Err(
                "ERA5 pressure-level retrieval requires at least one pressure level".to_string(),
            );
        }
    }

    Ok(request)
}

fn credentials_from_env(url_key: &str, secret_key: &str) -> Option<CdsCredentials> {
    let url = env::var(url_key).ok()?;
    let key = env::var(secret_key).ok()?;
    Some(CdsCredentials {
        url: url.trim().trim_end_matches('/').to_string(),
        key: key.trim().to_string(),
        source: String::new(),
    })
}

fn parse_cds_credentials_text(text: &str) -> Option<CdsCredentials> {
    let mut url = None;
    let mut key = None;
    for line in text.lines() {
        if let Some((name, value)) = line.split_once(':') {
            let name = name.trim();
            let value = value.trim();
            match name {
                "url" => url = Some(value.trim_end_matches('/').to_string()),
                "key" => key = Some(value.to_string()),
                _ => {}
            }
        }
    }
    Some(CdsCredentials {
        url: url?,
        key: key?,
        source: String::new(),
    })
}

fn home_candidate_paths(file_name: &str) -> Vec<std::path::PathBuf> {
    let mut paths = Vec::new();
    if let Some(home) = env::var_os("HOME") {
        paths.push(std::path::PathBuf::from(home).join(file_name));
    }
    if let Some(profile) = env::var_os("USERPROFILE") {
        paths.push(std::path::PathBuf::from(profile).join(file_name));
    }
    paths
}

#[cfg(target_os = "windows")]
fn read_wsl_text(path: &str) -> Option<String> {
    let command = format!("cat {path}");
    let output = Command::new("wsl.exe")
        .args(["sh", "-lc", &command])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
}

fn link_href(value: &Value, rel: &str) -> Option<String> {
    let links = value.get("links")?.as_array()?;
    links.iter().find_map(|item| {
        if item.get("rel").and_then(Value::as_str) == Some(rel) {
            item.get("href").and_then(Value::as_str).map(str::to_string)
        } else {
            None
        }
    })
}

fn is_remote(location: &str) -> bool {
    location.starts_with("http://") || location.starts_with("https://")
}

fn header_to_u64(value: Option<&reqwest::header::HeaderValue>) -> Result<Option<u64>, String> {
    match value {
        Some(value) => {
            let text = value
                .to_str()
                .map_err(|err| format!("invalid content-length header: {err}"))?;
            let parsed = text
                .parse::<u64>()
                .map_err(|err| format!("invalid content-length value '{text}': {err}"))?;
            Ok(Some(parsed))
        }
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use chrono::TimeZone;

    #[test]
    fn rounds_latest_cycle_for_hourly_models() {
        let engine = FetchEngine::new();
        let now = Utc.with_ymd_and_hms(2026, 3, 16, 18, 47, 9).unwrap();
        let cycle = engine.latest_cycle_for(ModelKind::Hrrr, now);
        assert_eq!(cycle, Utc.with_ymd_and_hms(2026, 3, 16, 18, 0, 0).unwrap());
    }

    #[test]
    fn rounds_latest_cycle_for_six_hour_models() {
        let engine = FetchEngine::new();
        let now = Utc.with_ymd_and_hms(2026, 3, 16, 18, 47, 9).unwrap();
        let cycle = engine.latest_cycle_for(ModelKind::Gfs, now);
        assert_eq!(cycle, Utc.with_ymd_and_hms(2026, 3, 16, 18, 0, 0).unwrap());
    }

    #[test]
    fn builds_hrrr_nomads_url() {
        let run = Utc.with_ymd_and_hms(2026, 3, 16, 18, 0, 0).unwrap();
        let url = build_url(
            &SourceKind::Nomads,
            ModelKind::Hrrr,
            ProductKind::Surface,
            run,
            3,
        )
        .expect("hrrr url should build");
        assert_eq!(
            url,
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.20260316/conus/hrrr.t18z.wrfsfcf03.grib2"
        );
    }

    #[test]
    fn builds_gfs_aws_url() {
        let run = Utc.with_ymd_and_hms(2026, 3, 16, 0, 0, 0).unwrap();
        let url = build_url(
            &SourceKind::Aws,
            ModelKind::Gfs,
            ProductKind::Pressure,
            run,
            12,
        )
        .expect("gfs url should build");
        assert_eq!(
            url,
            "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20260316/00/atmos/gfs.t00z.pgrb2.0p25.f012"
        );
    }

    #[test]
    fn builds_ecmwf_open_data_url_and_inventory() {
        let run = Utc.with_ymd_and_hms(2026, 3, 16, 0, 0, 0).unwrap();
        let url = build_url(
            &SourceKind::Ecmwf,
            ModelKind::EcmwfIfs,
            ProductKind::Surface,
            run,
            24,
        )
        .expect("ecmwf url should build");
        assert_eq!(
            url,
            "https://data.ecmwf.int/forecasts/20260316/00z/ifs/0p25/oper/20260316000000-24h-oper-fc.grib2"
        );
        assert_eq!(
            build_inventory_url(&SourceKind::Ecmwf, ModelKind::EcmwfIfs, &url)
                .expect("ecmwf inventory url should build"),
            "https://data.ecmwf.int/forecasts/20260316/00z/ifs/0p25/oper/20260316000000-24h-oper-fc.index"
        );
    }

    #[test]
    fn builds_era5_cds_plan_metadata() {
        let run = Utc.with_ymd_and_hms(2026, 3, 16, 6, 0, 0).unwrap();
        let url = build_url(
            &SourceKind::Cds,
            ModelKind::Era5,
            ProductKind::Surface,
            run,
            0,
        )
        .expect("era5 url should build");
        assert_eq!(url, "cds://reanalysis-era5-single-levels");
        let request = build_request_body_json(
            &SourceKind::Cds,
            ModelKind::Era5,
            ProductKind::Surface,
            run,
            0,
        )
        .expect("era5 cds request should exist");
        assert!(!request.contains("\"dataset\""));
        assert!(request.contains("\"time\":\"06:00\""));
    }

    #[test]
    fn rejects_unsupported_model_source_combo() {
        let run = Utc.with_ymd_and_hms(2026, 3, 16, 6, 0, 0).unwrap();
        let err = build_url(
            &SourceKind::Nomads,
            ModelKind::Era5,
            ProductKind::Surface,
            run,
            0,
        )
        .expect_err("nomads should not plan era5");
        assert!(err.contains("supported sources"));
    }

    #[test]
    fn plan_marks_era5_as_authenticated_request() {
        let engine = FetchEngine::new();
        let request = FetchRequest {
            model: ModelKind::Era5,
            run_time: Utc.with_ymd_and_hms(2026, 3, 16, 6, 0, 0).unwrap(),
            product: ProductKind::Surface,
            forecast_hour: 0,
            source: None,
        };
        let plan = engine.plan(&request).expect("era5 plan should build");
        assert!(plan.requires_auth);
        assert!(plan.request_body_json.is_some());
        assert!(plan.idx_url.is_empty());
    }

    #[test]
    fn fills_era5_request_overrides() {
        let engine = FetchEngine::new();
        let request = FetchRequest {
            model: ModelKind::Era5,
            run_time: Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            product: ProductKind::Pressure,
            forecast_hour: 0,
            source: Some(SourceKind::Cds),
        };
        let plan = engine.plan(&request).expect("era5 plan should build");
        let overrides = ModelDownloadOptions {
            variables: vec!["temperature".to_string()],
            pressure_levels: vec!["850".to_string()],
            area: Some([55.0, -130.0, 20.0, -60.0]),
            grid: Some([0.25, 0.25]),
        };
        let value = build_cds_request_from_plan(&plan, &overrides).expect("request should build");
        assert_eq!(
            value.pointer("/variable/0").and_then(Value::as_str),
            Some("temperature")
        );
        assert_eq!(
            value.pointer("/pressure_level/0").and_then(Value::as_str),
            Some("850")
        );
        assert_eq!(value.pointer("/area/0").and_then(Value::as_f64), Some(55.0));
        assert_eq!(value.pointer("/grid/1").and_then(Value::as_f64), Some(0.25));
    }

    #[test]
    fn parses_cdsapirc_text() {
        let credentials =
            parse_cds_credentials_text("url: https://cds.climate.copernicus.eu/api\nkey: abc123\n")
                .expect("credentials should parse");
        assert_eq!(credentials.url, "https://cds.climate.copernicus.eu/api");
        assert_eq!(credentials.key, "abc123");
    }

    #[test]
    fn coalesces_adjacent_ranges() {
        let merged = coalesce_ranges(&[
            ByteRange { start: 0, end: 99 },
            ByteRange {
                start: 100,
                end: 199,
            },
            ByteRange {
                start: 300,
                end: 399,
            },
        ]);
        assert_eq!(
            merged,
            vec![
                ByteRange { start: 0, end: 199 },
                ByteRange {
                    start: 300,
                    end: 399
                }
            ]
        );
    }

    #[test]
    fn resolves_trailing_range_from_content_length() {
        let range = resolve_offset_length(183, 0, Some(366)).expect("range should resolve");
        assert_eq!(
            range,
            ByteRange {
                start: 183,
                end: 365
            }
        );
    }

    #[test]
    fn fetches_local_ranges_and_stitches_bytes() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should move forward")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("wxforge_fetch_{suffix}.bin"));
        fs::write(&path, b"abcdefghij").expect("temp file should write");

        let engine = FetchEngine::new();
        let location = path.to_string_lossy().to_string();
        let bytes = engine
            .fetch_ranges(
                &location,
                &[
                    ByteRange { start: 2, end: 3 },
                    ByteRange { start: 6, end: 8 },
                ],
            )
            .expect("range fetch should succeed");
        let _ = fs::remove_file(path);

        assert_eq!(bytes, b"cdghi");
    }

    #[test]
    fn detects_local_resource_existence() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should move forward")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("wxforge_exists_{suffix}.txt"));
        fs::write(&path, b"ok").expect("temp file should write");

        let engine = FetchEngine::new();
        let location = path.to_string_lossy().to_string();
        assert!(engine
            .resource_exists(&location)
            .expect("resource probe should succeed"));
        let _ = fs::remove_file(path);
    }
}
