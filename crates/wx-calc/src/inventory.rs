use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CalcCategory {
    Thermo,
    Alias,
    Wind,
    Kinematics,
    Severe,
    Grid,
    Atmo,
    Smooth,
    Utils,
    Support,
    Dataset,
    Interpolation,
}

impl CalcCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Thermo => "thermo",
            Self::Alias => "alias",
            Self::Wind => "wind",
            Self::Kinematics => "kinematics",
            Self::Severe => "severe",
            Self::Grid => "grid",
            Self::Atmo => "atmo",
            Self::Smooth => "smooth",
            Self::Utils => "utils",
            Self::Support => "support",
            Self::Dataset => "dataset",
            Self::Interpolation => "interpolation",
        }
    }

    fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "thermo" => Some(Self::Thermo),
            "alias" => Some(Self::Alias),
            "wind" => Some(Self::Wind),
            "kinematics" => Some(Self::Kinematics),
            "severe" => Some(Self::Severe),
            "grid" => Some(Self::Grid),
            "atmo" => Some(Self::Atmo),
            "smooth" => Some(Self::Smooth),
            "utils" => Some(Self::Utils),
            "support" => Some(Self::Support),
            "dataset" => Some(Self::Dataset),
            "interpolation" => Some(Self::Interpolation),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortStatus {
    Ported,
    Planned,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalcPortEntry {
    pub name: &'static str,
    pub category: CalcCategory,
    pub status: PortStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalcCategorySummary {
    pub category: CalcCategory,
    pub total: usize,
    pub ported: usize,
    pub missing: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalcPortSummary {
    pub total: usize,
    pub ported: usize,
    pub missing: usize,
    pub categories: Vec<CalcCategorySummary>,
}

pub fn calc_port_inventory() -> Vec<CalcPortEntry> {
    let ported: BTreeSet<&'static str> = include_str!("ported_names.txt")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();

    include_str!("calc_inventory.txt")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let (category, name) = line
                .split_once(',')
                .unwrap_or_else(|| panic!("invalid calc inventory line '{line}'"));
            let category = CalcCategory::from_tag(category)
                .unwrap_or_else(|| panic!("invalid calc category '{category}'"));
            let status = if ported.contains(name) {
                PortStatus::Ported
            } else {
                PortStatus::Planned
            };
            CalcPortEntry {
                name,
                category,
                status,
            }
        })
        .collect()
}

pub fn calc_port_summary() -> CalcPortSummary {
    let inventory = calc_port_inventory();
    let mut categories = BTreeMap::<CalcCategory, CalcCategorySummary>::new();
    let mut ported = 0usize;
    for entry in &inventory {
        let summary = categories
            .entry(entry.category)
            .or_insert(CalcCategorySummary {
                category: entry.category,
                total: 0,
                ported: 0,
                missing: 0,
            });
        summary.total += 1;
        match entry.status {
            PortStatus::Ported => {
                ported += 1;
                summary.ported += 1;
            }
            PortStatus::Planned => summary.missing += 1,
        }
    }
    CalcPortSummary {
        total: inventory.len(),
        ported,
        missing: inventory.len().saturating_sub(ported),
        categories: categories.into_values().collect(),
    }
}

pub fn missing_names(limit: usize) -> Vec<&'static str> {
    calc_port_inventory()
        .into_iter()
        .filter(|entry| entry.status == PortStatus::Planned)
        .take(limit)
        .map(|entry| entry.name)
        .collect()
}
