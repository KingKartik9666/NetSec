use crate::{invalid, ModelError};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::{Component, Path, PathBuf}};

pub const MAX_BLOB: u64 = 512 * 1024 * 1024;

pub fn local_path(root: &Path, name: &str) -> Result<PathBuf, ModelError> {
    let path = Path::new(name);
    if name.is_empty() || name.contains(':') || name.contains('\\') || path.is_absolute()
        || path.components().any(|p| !matches!(p, Component::Normal(_))) {
        return Err(invalid("Artifact path must be portable and relative"));
    }
    let path = root.join(path).canonicalize()?;
    if !path.starts_with(root.canonicalize()?) { return Err(invalid("Artifact path escapes bundle")); }
    Ok(path)
}

pub fn checked_bytes(path: &Path, expected: &str, max: u64) -> Result<Vec<u8>, ModelError> {
    if fs::metadata(path)?.len() > max { return Err(invalid("Artifact exceeds configured byte bound")); }
    let bytes = fs::read(path)?;
    if format!("{:x}", Sha256::digest(&bytes)) != expected { return Err(invalid("Artifact SHA256 differs")); }
    Ok(bytes)
}

pub enum Data { F32(Vec<f32>), F64(Vec<f64>), I32(Vec<i32>), U32(Vec<u32>), U8(Vec<u8>) }
pub struct Tensor { pub shape: Vec<usize>, pub data: Data }
pub struct Tensors { pub values: BTreeMap<String, Tensor> }

impl Tensors {
    pub fn load(manifest: &Value, bytes: &[u8]) -> Result<Self, ModelError> {
        let description = manifest["tensors"].as_object().ok_or_else(|| invalid("Missing tensors"))?;
        if description.len() > 4096 { return Err(invalid("Too many tensors")); }
        let mut values = BTreeMap::new();
        let mut ranges = Vec::new();
        for (name, spec) in description {
            let shape: Vec<usize> = spec["shape"].as_array().ok_or_else(|| invalid("Invalid tensor shape"))?
                .iter().map(|v| v.as_u64().and_then(|n| usize::try_from(n).ok()).ok_or_else(|| invalid("Invalid tensor dimension")))
                .collect::<Result<_, _>>()?;
            if shape.len() > 4 { return Err(invalid("Tensor rank exceeds supported bound")); }
            let count = shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d)).ok_or_else(|| invalid("Tensor dimension overflow"))?;
            let offset = spec["offset_bytes"].as_u64().and_then(|n| usize::try_from(n).ok()).ok_or_else(|| invalid("Invalid tensor offset"))?;
            let size = spec["length_bytes"].as_u64().and_then(|n| usize::try_from(n).ok()).ok_or_else(|| invalid("Invalid tensor byte count"))?;
            let dtype = spec["dtype"].as_str().ok_or_else(|| invalid("Missing tensor type"))?;
            let width = match dtype { "f64" => 8, "f32"|"i32"|"u32" => 4, "u8" => 1, _ => return Err(invalid("Unsupported tensor dtype")) };
            if count.checked_mul(width) != Some(size) || offset.checked_add(size).map_or(true, |end| end > bytes.len()) {
                return Err(invalid("Tensor byte range invalid"));
            }
            let data = &bytes[offset..offset+size];
            let parsed = match dtype {
                "f32" => Data::F32(data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect()),
                "f64" => Data::F64(data.chunks_exact(8).map(|b| f64::from_le_bytes([b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7]])).collect()),
                "i32" => Data::I32(data.chunks_exact(4).map(|b| i32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect()),
                "u32" => Data::U32(data.chunks_exact(4).map(|b| u32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect()),
                "u8" => Data::U8(data.to_vec()),
                _ => return Err(invalid("Unsupported tensor dtype")),
            };
            if matches!(&parsed,Data::F32(v) if v.iter().any(|v|!v.is_finite()))
                || matches!(&parsed,Data::F64(v) if v.iter().any(|v|!v.is_finite())) {
                return Err(invalid("Nonfinite model weights"));
            }
            ranges.push((offset, offset+size));
            values.insert(name.clone(),Tensor {shape,data:parsed});
        }
        ranges.sort_unstable();
        if ranges.windows(2).any(|r|r[0].1>r[1].0) { return Err(invalid("Overlapping tensor ranges")); }
        Ok(Self{values})
    }
    pub fn f32(&self,name:&str) -> Result<&[f32],ModelError> {
        match &self.get(name)?.data { Data::F32(v)=>Ok(v), _=>Err(invalid(format!("Tensor {name} must be f32"))) }
    }
    pub fn f64(&self,name:&str) -> Result<&[f64],ModelError> {
        match &self.get(name)?.data { Data::F64(v)=>Ok(v), _=>Err(invalid(format!("Tensor {name} must be f64"))) }
    }
    pub fn i32(&self,name:&str) -> Result<&[i32],ModelError> {
        match &self.get(name)?.data { Data::I32(v)=>Ok(v), _=>Err(invalid(format!("Tensor {name} must be i32"))) }
    }
    pub fn u32(&self,name:&str) -> Result<&[u32],ModelError> {
        match &self.get(name)?.data { Data::U32(v)=>Ok(v), _=>Err(invalid(format!("Tensor {name} must be u32"))) }
    }
    pub fn get(&self,name:&str)->Result<&Tensor,ModelError>{self.values.get(name).ok_or_else(||invalid(format!("Missing tensor {name}")))}
    pub fn numbers(&self,name:&str)->Result<Vec<f64>,ModelError>{
        match &self.get(name)?.data {Data::F32(v)=>Ok(v.iter().map(|v|*v as f64).collect()),Data::F64(v)=>Ok(v.clone()),_=>Err(invalid("Expected floating tensor"))}
    }
}
