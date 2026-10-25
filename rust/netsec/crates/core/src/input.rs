use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

pub const MAX_ROWS: usize = 500;
pub const MAX_UPLOAD_BYTES: usize = 10 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSnapshot {
    pub id: String,
    pub origin: String,
    pub schema_hash: String,
    pub rows: Vec<Vec<Option<f64>>>,
}

pub fn snapshot(rows: Vec<Vec<Option<f64>>>, names: &[String], origin: &str) -> Result<InputSnapshot, String> {
    if rows.is_empty() || rows.len() > MAX_ROWS {
        return Err("Supply between 1 and 500 observations".into());
    }
    if names.is_empty() || names.len() > 4096 || names.iter().collect::<HashSet<_>>().len() != names.len() {
        return Err("Invalid ordered feature schema".into());
    }
    for (index, row) in rows.iter().enumerate() {
        if row.len() != names.len() {
            return Err(format!("Row {} requires exactly {} ordered features", index + 1, names.len()));
        }
        if row.iter().flatten().any(|v| !v.is_finite()) {
            return Err(format!("Row {} contains an infinite or nonnumeric value; use null for missing measurements", index + 1));
        }
        if row.iter().all(Option::is_none) {
            return Err(format!("Row {} contains no measured features", index + 1));
        }
    }
    let schema = serde_json::to_vec(names).map_err(|e| e.to_string())?;
    let data = serde_json::to_vec(&(1, origin, names, &rows)).map_err(|e| e.to_string())?;
    Ok(InputSnapshot {
        id: format!("{:x}", Sha256::digest(data)),
        origin: origin.into(),
        schema_hash: format!("{:x}", Sha256::digest(schema)),
        rows,
    })
}

pub fn parse_csv(text: &str, names: &[String]) -> Result<InputSnapshot, String> {
    if text.len() > MAX_UPLOAD_BYTES {
        return Err("CSV exceeds the 10 MiB upload limit".into());
    }
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut reader = csv::ReaderBuilder::new().trim(csv::Trim::All).flexible(false).from_reader(text.as_bytes());
    let headers = reader.headers().map_err(|e| e.to_string())?.clone();
    let unique: HashSet<_> = headers.iter().collect();
    if unique.len() != headers.len() || headers.iter().any(str::is_empty) {
        return Err("CSV column names must be nonempty and unique".into());
    }
    let positions: Vec<_> = names.iter().map(|name| {
        headers.iter().position(|header| header == name).ok_or_else(|| format!("CSV is missing required feature: {name}"))
    }).collect::<Result<_, _>>()?;
    let mut rows = Vec::new();
    for record in reader.records() {
        if rows.len() >= MAX_ROWS { return Err("CSV exceeds the 500-row limit".into()); }
        let record = record.map_err(|e| e.to_string())?;
        let mut row = Vec::with_capacity(names.len());
        for (column, &position) in positions.iter().enumerate() {
            let cell = &record[position];
            if cell.is_empty() || ["null", "none", "nan", "na", "n/a"].iter().any(|s| cell.eq_ignore_ascii_case(s)) {
                row.push(None);
            } else {
                let value: f64 = cell.parse().map_err(|_| format!("CSV row {}, {} must be numeric or empty", rows.len() + 1, names[column]))?;
                if !value.is_finite() { return Err(format!("CSV row {}, {} is infinite", rows.len() + 1, names[column])); }
                row.push(Some(value));
            }
        }
        rows.push(row);
    }
    snapshot(rows, names, "prepared_csv")
}

#[cfg(test)]
mod tests {
    use super::*;
    fn names() -> Vec<String> { vec!["count".into(), "value".into()] }
    #[test]
    fn csv_reorders_by_exact_names_and_preserves_zero_missing_and_signed_values() {
        let x = parse_csv("value,label4,count\n-3,attack,0\n,benign,4\n", &names()).unwrap();
        assert_eq!(x.rows, vec![vec![Some(0.0), Some(-3.0)], vec![Some(4.0), None]]);
    }
    #[test]
    fn duplicate_header_missing_feature_and_infinite_value_are_rejected() {
        for csv in ["count,count,value\n0,0,2", "count\n2", "count,value\n0,Infinity", "count,value\n0,1,2"] {
            assert!(parse_csv(csv, &names()).is_err(), "{csv}");
        }
    }
    #[test]
    fn snapshots_are_content_identified_with_source_and_schema() {
        let a = snapshot(vec![vec![Some(0.0), None]], &names(), "prepared_csv").unwrap();
        let b = snapshot(vec![vec![Some(0.0), None]], &names(), "prepared_csv").unwrap();
        let c = snapshot(vec![vec![None, Some(0.0)]], &names(), "prepared_csv").unwrap();
        assert_eq!(a.id, b.id);
        assert_ne!(a.id, c.id);
        assert_eq!(a.id.len(), 64);
        assert!(snapshot(vec![vec![None]], &names(), "row").is_err());
        assert!(snapshot(vec![], &names(), "row").is_err());
    }
    #[test]
    fn row_and_upload_limits_are_enforced_before_inference() {
        assert!(snapshot(vec![vec![None, None]; 501], &names(), "row").is_err());
        assert!(parse_csv(&"a".repeat(10 * 1024 * 1024 + 1), &names()).is_err());
    }
}
