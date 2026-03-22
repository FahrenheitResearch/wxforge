//! Weather-to-training-data planning.

use serde::{Deserialize, Serialize};
use wx_export::{ExportFormat, ExportPlan};
use wx_types::{TrainingChannel, TrainingSpec};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LabelSpec {
    pub name: String,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingPlan {
    pub spec: TrainingSpec,
    pub labels: Vec<LabelSpec>,
    pub export: ExportPlan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelArchitecture {
    ClassicalMl,
    Diffusion,
    SwinTransformer,
    #[serde(alias = "fgn")]
    ForecastGraphNetwork,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearningTask {
    Regression,
    #[serde(alias = "classification")]
    BinaryClassification,
    MulticlassClassification,
    Segmentation,
    #[serde(alias = "diffusion_denoising")]
    Denoising,
    Forecasting,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureProfile {
    SurfaceCore,
    PressureCore,
    SevereDiagnostics,
    RadarCore,
    ThermodynamicProfiles,
    TabularStats,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CustomFeatureSpec {
    pub name: String,
    pub units: String,
    #[serde(default)]
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureRequest {
    #[serde(default)]
    pub profiles: Vec<FeatureProfile>,
    #[serde(default)]
    pub extra_channels: Vec<TrainingChannel>,
    #[serde(default)]
    pub custom_features: Vec<CustomFeatureSpec>,
}

impl Default for FeatureRequest {
    fn default() -> Self {
        Self {
            profiles: Vec::new(),
            extra_channels: Vec::new(),
            custom_features: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelWindowRequest {
    pub model: String,
    pub product: String,
    pub start: String,
    pub end: String,
    #[serde(default)]
    pub forecast_hours: Vec<u16>,
    #[serde(default)]
    pub variables: Vec<String>,
    #[serde(default)]
    pub pressure_levels: Vec<String>,
    #[serde(default)]
    pub area: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AgentDataSource {
    #[serde(alias = "existing_grib_samples")]
    ExistingGribFiles { samples: Vec<GribSampleInput> },
    #[serde(alias = "model_collection")]
    ModelWindow { request: ModelWindowRequest },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetOutputSpec {
    #[serde(default)]
    pub format: Option<ExportFormat>,
    #[serde(default = "default_shard_count")]
    pub shard_count: usize,
    #[serde(default = "default_parallelism")]
    pub parallelism: usize,
    #[serde(default = "default_train_fraction")]
    pub train_fraction: f64,
    #[serde(default = "default_validation_fraction")]
    pub validation_fraction: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRecipeSpec {
    pub architecture: ModelArchitecture,
    #[serde(alias = "objective")]
    pub task: LearningTask,
    #[serde(default)]
    pub lead_hours: Option<u16>,
    #[serde(default)]
    pub context_steps: usize,
    #[serde(default)]
    pub patch_size: Option<u16>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentJobSpec {
    pub job_name: String,
    pub dataset_name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(alias = "data")]
    pub data_source: AgentDataSource,
    #[serde(default)]
    pub features: FeatureRequest,
    #[serde(default)]
    pub labels: Vec<LabelSpec>,
    pub model: ModelRecipeSpec,
    #[serde(default)]
    pub output: Option<DatasetOutputSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FetchPlanOutline {
    pub model: String,
    pub product: String,
    pub start: String,
    pub end: String,
    pub forecast_hours: Vec<u16>,
    pub variables: Vec<String>,
    pub pressure_levels: Vec<String>,
    pub area: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRecipePlan {
    pub architecture: ModelArchitecture,
    pub task: LearningTask,
    pub input_layout: String,
    pub trainer_family: String,
    pub recommended_loss: String,
    pub recommended_format: ExportFormat,
    pub channels: Vec<TrainingChannel>,
    pub labels: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentJobPlan {
    pub job_name: String,
    pub description: Option<String>,
    pub training: TrainingPlan,
    pub dataset_build_request: Option<GribDatasetBuildRequest>,
    pub fetch_plan: Option<FetchPlanOutline>,
    pub model_recipe: ModelRecipePlan,
    pub assumptions: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GribSampleInput {
    pub file: String,
    #[serde(default)]
    pub sample_id: Option<String>,
    #[serde(default)]
    pub messages: Vec<u64>,
    #[serde(default)]
    pub companion_files: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GribDatasetBuildRequest {
    pub dataset_name: String,
    #[serde(default = "default_export_format")]
    pub format: ExportFormat,
    #[serde(default = "default_shard_count")]
    pub shard_count: usize,
    #[serde(default = "default_parallelism")]
    pub parallelism: usize,
    #[serde(default = "default_train_fraction")]
    pub train_fraction: f64,
    #[serde(default = "default_validation_fraction")]
    pub validation_fraction: f64,
    pub samples: Vec<GribSampleInput>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DatasetSplit {
    Train,
    Validation,
    Test,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetSampleRef {
    pub sample_id: String,
    pub source: String,
    pub relative_dir: String,
    pub channel_count: usize,
    pub shard_index: usize,
    pub split: DatasetSplit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetSplitCounts {
    pub train: usize,
    pub validation: usize,
    pub test: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetBuildManifest {
    pub dataset_name: String,
    pub generated_at: String,
    pub format: ExportFormat,
    pub shard_count: usize,
    pub sample_count: usize,
    pub total_channel_count: usize,
    pub split_counts: DatasetSplitCounts,
    pub samples: Vec<DatasetSampleRef>,
}

fn default_shard_count() -> usize {
    1
}

fn default_export_format() -> ExportFormat {
    ExportFormat::NpyDirectory
}

fn default_parallelism() -> usize {
    1
}

fn default_train_fraction() -> f64 {
    0.8
}

fn default_validation_fraction() -> f64 {
    0.1
}

impl TrainingPlan {
    pub fn from_channels(dataset_name: impl Into<String>, channels: Vec<TrainingChannel>) -> Self {
        let dataset_name = dataset_name.into();
        let spec = TrainingSpec {
            dataset_name: dataset_name.clone(),
            channels,
            labels: Vec::new(),
        };

        let export = ExportPlan {
            spec: spec.clone(),
            format: ExportFormat::NpyDirectory,
            shard_count: 1,
        };

        Self {
            spec,
            labels: Vec::new(),
            export,
        }
    }

    pub fn baseline() -> Self {
        let spec = TrainingSpec {
            dataset_name: "baseline-weather-dataset".to_string(),
            channels: vec![
                channel("reflectivity", "dBZ"),
                channel("velocity", "m/s"),
                channel("theta_e", "K"),
            ],
            labels: vec!["rotation".to_string(), "hail".to_string()],
        };

        let export = ExportPlan {
            spec: spec.clone(),
            format: ExportFormat::WebDataset,
            shard_count: 128,
        };

        Self {
            spec,
            labels: vec![
                LabelSpec {
                    name: "rotation".to_string(),
                    source: "radar_detection".to_string(),
                },
                LabelSpec {
                    name: "hail".to_string(),
                    source: "warning_or_report".to_string(),
                },
            ],
            export,
        }
    }

    pub fn radar_supervised() -> Self {
        let spec = TrainingSpec {
            dataset_name: "radar-supervised-weather-dataset".to_string(),
            channels: vec![
                channel("reflectivity", "dBZ"),
                channel("velocity", "m/s"),
                channel("spectrum_width", "m/s"),
                channel("theta_e", "K"),
                channel("cape", "J/kg"),
            ],
            labels: vec![
                "mesocyclone".to_string(),
                "hail".to_string(),
                "tornado_warning".to_string(),
            ],
        };

        let export = ExportPlan {
            spec: spec.clone(),
            format: ExportFormat::Parquet,
            shard_count: 64,
        };

        Self {
            spec,
            labels: vec![
                LabelSpec {
                    name: "mesocyclone".to_string(),
                    source: "radar_rotation_track".to_string(),
                },
                LabelSpec {
                    name: "hail".to_string(),
                    source: "hail_report_or_mesh".to_string(),
                },
                LabelSpec {
                    name: "tornado_warning".to_string(),
                    source: "warning_polygon".to_string(),
                },
            ],
            export,
        }
    }
}

impl AgentJobSpec {
    pub fn starter(
        job_name: impl Into<String>,
        dataset_name: impl Into<String>,
        architecture: ModelArchitecture,
        task: LearningTask,
    ) -> Self {
        let job_name = job_name.into();
        let dataset_name = dataset_name.into();
        let output = DatasetOutputSpec {
            format: Some(default_format_for_architecture(architecture)),
            shard_count: recommended_shard_count(architecture),
            parallelism: default_parallelism(),
            train_fraction: default_train_fraction(),
            validation_fraction: default_validation_fraction(),
        };
        Self {
            job_name,
            dataset_name,
            description: Some(format!(
                "Starter agent job for {:?} {:?} workflow",
                architecture, task
            )),
            data_source: AgentDataSource::ExistingGribFiles {
                samples: vec![GribSampleInput {
                    file: "examples/sample.grib2".to_string(),
                    sample_id: Some("sample_0001".to_string()),
                    messages: vec![1],
                    companion_files: Vec::new(),
                }],
            },
            features: FeatureRequest {
                profiles: default_profiles_for_architecture(architecture),
                extra_channels: Vec::new(),
                custom_features: Vec::new(),
            },
            labels: default_labels_for_architecture(architecture, task),
            model: ModelRecipeSpec {
                architecture,
                task,
                lead_hours: if matches!(task, LearningTask::Forecasting) {
                    Some(6)
                } else {
                    None
                },
                context_steps: default_context_steps(architecture),
                patch_size: default_patch_size(architecture),
                notes: architecture_notes(architecture),
            },
            output: Some(output),
        }
    }
}

pub fn plan_agent_job(spec: &AgentJobSpec) -> AgentJobPlan {
    let output = resolved_output_spec(spec);
    let channels = expand_training_channels(&spec.features, spec.model.architecture);
    let labels = if spec.labels.is_empty() {
        default_labels_for_architecture(spec.model.architecture, spec.model.task)
    } else {
        spec.labels.clone()
    };
    let label_names = labels
        .iter()
        .map(|label| label.name.clone())
        .collect::<Vec<_>>();

    let training_spec = TrainingSpec {
        dataset_name: spec.dataset_name.clone(),
        channels: channels.clone(),
        labels: label_names.clone(),
    };
    let training = TrainingPlan {
        spec: training_spec.clone(),
        labels,
        export: ExportPlan {
            spec: training_spec,
            format: output.format.expect("resolved output format should exist"),
            shard_count: output.shard_count,
        },
    };

    let dataset_build_request = match &spec.data_source {
        AgentDataSource::ExistingGribFiles { samples } => Some(GribDatasetBuildRequest {
            dataset_name: spec.dataset_name.clone(),
            format: output.format.expect("resolved output format should exist"),
            shard_count: output.shard_count,
            parallelism: output.parallelism,
            train_fraction: output.train_fraction,
            validation_fraction: output.validation_fraction,
            samples: samples.clone(),
        }),
        AgentDataSource::ModelWindow { .. } => None,
    };

    let fetch_plan = match &spec.data_source {
        AgentDataSource::ExistingGribFiles { .. } => None,
        AgentDataSource::ModelWindow { request } => Some(FetchPlanOutline {
            model: request.model.clone(),
            product: request.product.clone(),
            start: request.start.clone(),
            end: request.end.clone(),
            forecast_hours: if request.forecast_hours.is_empty() {
                vec![0]
            } else {
                request.forecast_hours.clone()
            },
            variables: request.variables.clone(),
            pressure_levels: request.pressure_levels.clone(),
            area: request.area.clone(),
        }),
    };

    let mut assumptions = vec![
        format!("architecture_defaults={:?}", spec.model.architecture),
        format!("task={:?}", spec.model.task),
        format!("channel_count={}", training.spec.channels.len()),
    ];
    if spec.features.custom_features.is_empty() {
        assumptions.push("no_custom_features_requested".to_string());
    } else {
        assumptions.push(format!(
            "custom_features={}",
            spec.features.custom_features.len()
        ));
    }
    if matches!(&spec.data_source, AgentDataSource::ModelWindow { .. }) {
        assumptions.push(
            "model_window source requires fetch/decode expansion before dataset execution"
                .to_string(),
        );
    }

    let mut warnings = Vec::new();
    if matches!(spec.model.architecture, ModelArchitecture::ClassicalMl)
        && output.format == Some(ExportFormat::WebDataset)
    {
        warnings.push(
            "classical_ml usually prefers Parquet or Jsonl over WebDataset shards".to_string(),
        );
    }
    if spec
        .features
        .custom_features
        .iter()
        .any(|feature| !custom_feature_materialization_supported(&feature.name))
    {
        warnings.push(
            "some custom features do not map to a registered dataset materializer yet and remain planner-only".to_string(),
        );
    }
    if matches!(
        spec.model.architecture,
        ModelArchitecture::ForecastGraphNetwork
    ) {
        warnings.push(
            "forecast_graph_network planning is architecture-aware, but graph node/edge materialization still needs a downstream builder".to_string(),
        );
    }

    AgentJobPlan {
        job_name: spec.job_name.clone(),
        description: spec.description.clone(),
        training: training.clone(),
        dataset_build_request,
        fetch_plan,
        model_recipe: build_model_recipe(&training, &spec.model),
        assumptions,
        warnings,
    }
}

fn resolved_output_spec(spec: &AgentJobSpec) -> DatasetOutputSpec {
    let mut output = spec.output.clone().unwrap_or(DatasetOutputSpec {
        format: None,
        shard_count: default_shard_count(),
        parallelism: default_parallelism(),
        train_fraction: default_train_fraction(),
        validation_fraction: default_validation_fraction(),
    });

    if output.format.is_none() {
        output.format = Some(default_format_for_architecture(spec.model.architecture));
    }
    if output.shard_count == 0 {
        output.shard_count = recommended_shard_count(spec.model.architecture);
    }
    if output.parallelism == 0 {
        output.parallelism = default_parallelism();
    }
    if !(0.0..=1.0).contains(&output.train_fraction) {
        output.train_fraction = default_train_fraction();
    }
    if !(0.0..=1.0).contains(&output.validation_fraction) {
        output.validation_fraction = default_validation_fraction();
    }
    if output.train_fraction + output.validation_fraction > 1.0 {
        output.train_fraction = default_train_fraction();
        output.validation_fraction = default_validation_fraction();
    }
    output
}

fn build_model_recipe(training: &TrainingPlan, recipe: &ModelRecipeSpec) -> ModelRecipePlan {
    ModelRecipePlan {
        architecture: recipe.architecture,
        task: recipe.task,
        input_layout: architecture_input_layout(recipe.architecture).to_string(),
        trainer_family: architecture_trainer_family(recipe.architecture).to_string(),
        recommended_loss: recommended_loss(recipe.task).to_string(),
        recommended_format: training.export.format,
        channels: training.spec.channels.clone(),
        labels: training.spec.labels.clone(),
        notes: recipe.notes.clone(),
    }
}

fn expand_training_channels(
    request: &FeatureRequest,
    architecture: ModelArchitecture,
) -> Vec<TrainingChannel> {
    let mut channels = Vec::new();
    let profiles = if request.profiles.is_empty() {
        default_profiles_for_architecture(architecture)
    } else {
        request.profiles.clone()
    };

    for profile in profiles {
        for channel in channels_for_profile(profile) {
            push_unique_channel(&mut channels, channel);
        }
    }
    for channel in &request.extra_channels {
        push_unique_channel(&mut channels, channel.clone());
    }
    for custom in &request.custom_features {
        push_unique_channel(
            &mut channels,
            TrainingChannel {
                name: custom.name.clone(),
                units: custom.units.clone(),
            },
        );
    }
    if channels.is_empty() {
        for channel in channels_for_profile(FeatureProfile::SurfaceCore) {
            push_unique_channel(&mut channels, channel);
        }
    }
    channels
}

fn push_unique_channel(channels: &mut Vec<TrainingChannel>, channel: TrainingChannel) {
    if channels
        .iter()
        .any(|existing| existing.name == channel.name)
    {
        return;
    }
    channels.push(channel);
}

fn channels_for_profile(profile: FeatureProfile) -> Vec<TrainingChannel> {
    match profile {
        FeatureProfile::SurfaceCore => vec![
            channel("t2m", "K"),
            channel("d2m", "K"),
            channel("u10", "m/s"),
            channel("v10", "m/s"),
            channel("mslp", "Pa"),
        ],
        FeatureProfile::PressureCore => vec![
            channel("z500", "gpm"),
            channel("t850", "K"),
            channel("u850", "m/s"),
            channel("v850", "m/s"),
            channel("vort500", "1/s"),
            channel("div500", "1/s"),
            channel("theta850", "K"),
            channel("tadv850", "K/s"),
        ],
        FeatureProfile::SevereDiagnostics => vec![
            channel("sbcape", "J/kg"),
            channel("sbcin", "J/kg"),
            channel("mlcape", "J/kg"),
            channel("mlcin", "J/kg"),
            channel("mucape", "J/kg"),
            channel("mucin", "J/kg"),
            channel("srh01", "m2/s2"),
            channel("srh03", "m2/s2"),
            channel("shear06", "m/s"),
            channel("stp", "1"),
            channel("scp", "1"),
            channel("pwat", "mm"),
        ],
        FeatureProfile::RadarCore => vec![
            channel("reflectivity", "dBZ"),
            channel("velocity", "m/s"),
            channel("spectrum_width", "m/s"),
        ],
        FeatureProfile::ThermodynamicProfiles => vec![
            channel("theta_e", "K"),
            channel("wet_bulb", "K"),
            channel("wet_bulb_potential_temperature", "K"),
            channel("lcl_height", "m"),
            channel("lfc_height", "m"),
            channel("dcape", "J/kg"),
        ],
        FeatureProfile::TabularStats => vec![
            channel("channel_min", "varies"),
            channel("channel_mean", "varies"),
            channel("channel_max", "varies"),
            channel("valid_hour_sin", "1"),
            channel("valid_hour_cos", "1"),
        ],
    }
}

fn channel(name: &str, units: &str) -> TrainingChannel {
    TrainingChannel {
        name: name.to_string(),
        units: units.to_string(),
    }
}

fn default_profiles_for_architecture(architecture: ModelArchitecture) -> Vec<FeatureProfile> {
    match architecture {
        ModelArchitecture::ClassicalMl => vec![
            FeatureProfile::SurfaceCore,
            FeatureProfile::SevereDiagnostics,
            FeatureProfile::TabularStats,
        ],
        ModelArchitecture::Diffusion => vec![
            FeatureProfile::SurfaceCore,
            FeatureProfile::PressureCore,
            FeatureProfile::SevereDiagnostics,
            FeatureProfile::ThermodynamicProfiles,
        ],
        ModelArchitecture::SwinTransformer => vec![
            FeatureProfile::SurfaceCore,
            FeatureProfile::PressureCore,
            FeatureProfile::SevereDiagnostics,
        ],
        ModelArchitecture::ForecastGraphNetwork => vec![
            FeatureProfile::SurfaceCore,
            FeatureProfile::PressureCore,
            FeatureProfile::ThermodynamicProfiles,
        ],
        ModelArchitecture::Custom => vec![FeatureProfile::SurfaceCore],
    }
}

fn custom_feature_materialization_supported(name: &str) -> bool {
    name.to_ascii_lowercase().starts_with("custom_srh_")
}

fn default_labels_for_architecture(
    architecture: ModelArchitecture,
    task: LearningTask,
) -> Vec<LabelSpec> {
    let names = match (architecture, task) {
        (_, LearningTask::Forecasting) => vec!["future_target".to_string()],
        (_, LearningTask::Regression) => vec!["target_value".to_string()],
        (_, LearningTask::BinaryClassification) => vec!["event".to_string()],
        (_, LearningTask::MulticlassClassification) => vec!["class_id".to_string()],
        (_, LearningTask::Segmentation) => vec!["mask".to_string()],
        (ModelArchitecture::Diffusion, LearningTask::Denoising) => {
            vec!["clean_target".to_string()]
        }
        (_, LearningTask::Denoising) => vec!["denoised_target".to_string()],
    };
    names
        .into_iter()
        .map(|name| LabelSpec {
            name,
            source: "agent_default".to_string(),
        })
        .collect()
}

fn default_format_for_architecture(architecture: ModelArchitecture) -> ExportFormat {
    match architecture {
        ModelArchitecture::ClassicalMl => ExportFormat::Parquet,
        ModelArchitecture::Diffusion => ExportFormat::WebDataset,
        ModelArchitecture::SwinTransformer => ExportFormat::WebDataset,
        ModelArchitecture::ForecastGraphNetwork => ExportFormat::Parquet,
        ModelArchitecture::Custom => ExportFormat::Parquet,
    }
}

fn recommended_shard_count(architecture: ModelArchitecture) -> usize {
    match architecture {
        ModelArchitecture::ClassicalMl => 16,
        ModelArchitecture::Diffusion => 128,
        ModelArchitecture::SwinTransformer => 96,
        ModelArchitecture::ForecastGraphNetwork => 48,
        ModelArchitecture::Custom => 32,
    }
}

fn default_context_steps(architecture: ModelArchitecture) -> usize {
    match architecture {
        ModelArchitecture::ClassicalMl => 1,
        ModelArchitecture::Diffusion => 4,
        ModelArchitecture::SwinTransformer => 4,
        ModelArchitecture::ForecastGraphNetwork => 6,
        ModelArchitecture::Custom => 1,
    }
}

fn default_patch_size(architecture: ModelArchitecture) -> Option<u16> {
    match architecture {
        ModelArchitecture::ClassicalMl => None,
        ModelArchitecture::Diffusion => Some(8),
        ModelArchitecture::SwinTransformer => Some(4),
        ModelArchitecture::ForecastGraphNetwork => None,
        ModelArchitecture::Custom => None,
    }
}

fn architecture_notes(architecture: ModelArchitecture) -> Vec<String> {
    match architecture {
        ModelArchitecture::ClassicalMl => vec![
            "Prefer engineered scalar features and compact parquet datasets".to_string(),
            "Add aggregation windows before training large gradient-boosted ensembles".to_string(),
        ],
        ModelArchitecture::Diffusion => vec![
            "Preserve raster geometry and deterministic normalization metadata".to_string(),
            "WebDataset shards fit distributed GPU training better than row-wise formats"
                .to_string(),
        ],
        ModelArchitecture::SwinTransformer => vec![
            "Patch-based raster encoder expects stable channel ordering".to_string(),
            "Keep context frames aligned on a fixed grid for spatiotemporal attention".to_string(),
        ],
        ModelArchitecture::ForecastGraphNetwork => vec![
            "Node/edge construction still needs a downstream graph builder".to_string(),
            "Parquet is the safe default until graph-native shard output exists".to_string(),
        ],
        ModelArchitecture::Custom => {
            vec!["Custom architecture requires explicit downstream trainer integration".to_string()]
        }
    }
}

fn architecture_input_layout(architecture: ModelArchitecture) -> &'static str {
    match architecture {
        ModelArchitecture::ClassicalMl => "tabular",
        ModelArchitecture::Diffusion => "raster_sequence",
        ModelArchitecture::SwinTransformer => "raster_sequence",
        ModelArchitecture::ForecastGraphNetwork => "graph_nodes",
        ModelArchitecture::Custom => "custom",
    }
}

fn architecture_trainer_family(architecture: ModelArchitecture) -> &'static str {
    match architecture {
        ModelArchitecture::ClassicalMl => "xgboost_or_tabular_mlp",
        ModelArchitecture::Diffusion => "unet_or_dit_diffusion",
        ModelArchitecture::SwinTransformer => "hierarchical_vision_transformer",
        ModelArchitecture::ForecastGraphNetwork => "message_passing_or_graph_operator",
        ModelArchitecture::Custom => "custom",
    }
}

fn recommended_loss(task: LearningTask) -> &'static str {
    match task {
        LearningTask::Regression => "mse",
        LearningTask::BinaryClassification => "binary_cross_entropy",
        LearningTask::MulticlassClassification => "cross_entropy",
        LearningTask::Segmentation => "dice_plus_bce",
        LearningTask::Denoising => "noise_prediction_mse",
        LearningTask::Forecasting => "smooth_l1",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_builds_existing_file_dataset_request() {
        let spec = AgentJobSpec::starter(
            "diffusion-job",
            "diffusion-dataset",
            ModelArchitecture::Diffusion,
            LearningTask::Forecasting,
        );
        let plan = plan_agent_job(&spec);

        assert_eq!(plan.training.spec.dataset_name, "diffusion-dataset");
        assert!(plan.dataset_build_request.is_some());
        assert_eq!(
            plan.model_recipe.recommended_format,
            ExportFormat::WebDataset
        );
        assert!(plan
            .training
            .spec
            .channels
            .iter()
            .any(|c| c.name == "theta850"));
        assert!(plan.fetch_plan.is_none());
    }

    #[test]
    fn planner_tracks_model_window_as_fetch_plan() {
        let spec = AgentJobSpec {
            job_name: "hrrr-custom-srh".to_string(),
            dataset_name: "hrrr-weekly-dataset".to_string(),
            description: Some("Build a weekly HRRR severe dataset".to_string()),
            data_source: AgentDataSource::ModelWindow {
                request: ModelWindowRequest {
                    model: "hrrr".to_string(),
                    product: "pressure".to_string(),
                    start: "2026-03-01T00:00:00Z".to_string(),
                    end: "2026-03-08T00:00:00Z".to_string(),
                    forecast_hours: vec![0, 1, 2, 3, 6],
                    variables: vec!["ugrd".to_string(), "vgrd".to_string(), "tmp".to_string()],
                    pressure_levels: vec!["1000 mb".to_string(), "850 mb".to_string()],
                    area: Some("25,-105,45,-85".to_string()),
                },
            },
            features: FeatureRequest {
                profiles: vec![
                    FeatureProfile::SurfaceCore,
                    FeatureProfile::SevereDiagnostics,
                ],
                extra_channels: Vec::new(),
                custom_features: vec![CustomFeatureSpec {
                    name: "custom_srh".to_string(),
                    units: "m2/s2".to_string(),
                    source: "project_specific_srh_v1".to_string(),
                }],
            },
            labels: vec![LabelSpec {
                name: "tornado_warning".to_string(),
                source: "warning_polygon".to_string(),
            }],
            model: ModelRecipeSpec {
                architecture: ModelArchitecture::ForecastGraphNetwork,
                task: LearningTask::Forecasting,
                lead_hours: Some(6),
                context_steps: 6,
                patch_size: None,
                notes: architecture_notes(ModelArchitecture::ForecastGraphNetwork),
            },
            output: Some(DatasetOutputSpec {
                format: Some(ExportFormat::Parquet),
                shard_count: 24,
                parallelism: 4,
                train_fraction: 0.75,
                validation_fraction: 0.15,
            }),
        };

        let plan = plan_agent_job(&spec);
        assert!(plan.dataset_build_request.is_none());
        assert!(plan.fetch_plan.is_some());
        assert!(plan
            .training
            .spec
            .channels
            .iter()
            .any(|channel| channel.name == "custom_srh"));
        assert!(plan
            .warnings
            .iter()
            .any(|warning| warning.contains("custom features")));
        assert_eq!(
            plan.model_recipe.trainer_family,
            "message_passing_or_graph_operator"
        );
    }
}
