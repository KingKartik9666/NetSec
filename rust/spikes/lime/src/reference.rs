use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Cursor, Read};
use std::path::Path;
use zip::ZipArchive;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ReferenceMetadata {
    pub feature_names: Vec<String>,
    pub categorical_indices: Vec<usize>,
    pub rows: usize,
    pub sha256: String,
    #[serde(default)]
    pub kind: String,
}

#[derive(Clone, Debug)]
pub struct ReferenceData {
    pub metadata: ReferenceMetadata,
    pub training: Vec<Vec<Option<f32>>>,
    pub medians: Vec<f32>,
}

impl ReferenceData {
    pub fn load(path: &Path, expected_names: &[String]) -> Result<Self> {
        let metadata: ReferenceMetadata =
            serde_json::from_slice(&std::fs::read(path.with_extension("json"))?)?;
        if metadata.feature_names != expected_names {
            return Err(Error::Invalid(
                "The frozen training reference feature order does not match".into(),
            ));
        }
        let bytes = std::fs::read(path)?;
        if bytes.len() > 16 * 1024 * 1024 {
            return Err(Error::Invalid("Frozen reference exceeds 16 MiB".into()));
        }
        if format!("{:x}", Sha256::digest(&bytes)) != metadata.sha256 {
            return Err(Error::Invalid(
                "The frozen training reference checksum does not match".into(),
            ));
        }
        let mut archive = ZipArchive::new(Cursor::new(bytes))
            .map_err(|error| Error::Invalid(format!("Invalid reference ZIP: {error}")))?;
        let training = read_array(&mut archive, "training.npy")?;
        let medians = read_array(&mut archive, "medians.npy")?;
        if training.shape != [metadata.rows, expected_names.len()]
            || medians.shape != [expected_names.len()]
            || metadata.rows == 0
            || metadata.rows > 8192
            || expected_names.is_empty()
            || expected_names.len() > 2048
            || metadata
                .categorical_indices
                .iter()
                .any(|i| *i >= expected_names.len())
            || medians.values.iter().any(|value| !value.is_finite())
        {
            return Err(Error::Invalid(
                "Invalid training reference shapes or statistics".into(),
            ));
        }
        let columns = expected_names.len();
        let rows = (0..metadata.rows)
            .map(|i| {
                (0..columns)
                    .map(|j| {
                        let value = training.values[if training.fortran {
                            j * metadata.rows + i
                        } else {
                            i * columns + j
                        }];
                        value.is_finite().then_some(value)
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            metadata,
            training: rows,
            medians: medians.values,
        })
    }
}

struct Array {
    shape: Vec<usize>,
    fortran: bool,
    values: Vec<f32>,
}

fn read_array(archive: &mut ZipArchive<Cursor<Vec<u8>>>, name: &str) -> Result<Array> {
    let entry = archive
        .by_name(name)
        .map_err(|error| Error::Invalid(format!("Missing reference array {name}: {error}")))?;
    if entry.size() > 64 * 1024 * 1024 {
        return Err(Error::Invalid("Expanded reference exceeds 64 MiB".into()));
    }
    let mut bytes = Vec::new();
    entry.take(64 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
    parse_npy(&bytes)
}

// Bounded numeric NPY reader, independent of key order. No Python execution,
// pickle/object arrays or unbounded arbitrary literal parser is involved.
fn parse_npy(bytes: &[u8]) -> Result<Array> {
    let invalid = || Error::Invalid("Unsupported or malformed numeric NPY reference".into());
    if bytes.get(..6) != Some(b"\x93NUMPY") {
        return Err(invalid());
    }
    let (prefix, header_len) = match bytes.get(6..8) {
        Some([1, 0]) if bytes.len() >= 10 => {
            (10, usize::from(u16::from_le_bytes([bytes[8], bytes[9]])))
        }
        Some([2 | 3, 0]) if bytes.len() >= 12 => (
            12,
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
        ),
        _ => return Err(invalid()),
    };
    if header_len > 65536 {
        return Err(invalid());
    }
    let header = bytes.get(prefix..prefix + header_len).ok_or_else(invalid)?;
    let mut parser = Header {
        data: std::str::from_utf8(header).map_err(|_| invalid())?,
        offset: 0,
    };
    parser.symbol('{')?;
    let mut descr = None;
    let mut fortran = None;
    let mut shape = None;
    while !parser.consume('}') {
        let key = parser.string()?;
        parser.symbol(':')?;
        match key.as_str() {
            "descr" if descr.is_none() => descr = Some(parser.string()?),
            "fortran_order" if fortran.is_none() => {
                parser.whitespace();
                if parser.data[parser.offset..].starts_with("True") {
                    parser.offset += 4;
                    fortran = Some(true);
                } else if parser.data[parser.offset..].starts_with("False") {
                    parser.offset += 5;
                    fortran = Some(false);
                } else {
                    return Err(invalid());
                }
            }
            "shape" if shape.is_none() => {
                parser.symbol('(')?;
                let mut dimensions = Vec::new();
                while !parser.consume(')') {
                    parser.whitespace();
                    let begin = parser.offset;
                    while parser
                        .data
                        .as_bytes()
                        .get(parser.offset)
                        .is_some_and(u8::is_ascii_digit)
                    {
                        parser.offset += 1;
                    }
                    dimensions.push(
                        parser.data[begin..parser.offset]
                            .parse::<usize>()
                            .map_err(|_| invalid())?,
                    );
                    if dimensions.len() > 2 {
                        return Err(invalid());
                    }
                    if !parser.consume(',') {
                        parser.symbol(')')?;
                        break;
                    }
                }
                shape = Some(dimensions);
            }
            _ => return Err(invalid()),
        }
        if !parser.consume(',') {
            parser.symbol('}')?;
            break;
        }
    }
    parser.whitespace();
    if parser.offset != parser.data.len() {
        return Err(invalid());
    }
    let descr = descr.ok_or_else(invalid)?;
    let fortran = fortran.ok_or_else(invalid)?;
    let shape = shape.ok_or_else(invalid)?;
    if shape.is_empty() {
        return Err(invalid());
    }
    let (width, big) = match descr.as_str() {
        "<f4" | "=f4" => (4, false),
        ">f4" => (4, true),
        "<f8" | "=f8" => (8, false),
        ">f8" => (8, true),
        _ => return Err(invalid()),
    };
    let count = shape
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .ok_or_else(invalid)?;
    let data = &bytes[prefix + header_len..];
    if count.checked_mul(width) != Some(data.len()) || data.len() > 64 * 1024 * 1024 {
        return Err(invalid());
    }
    let values = data
        .chunks_exact(width)
        .map(|chunk| {
            if width == 4 {
                let raw = [chunk[0], chunk[1], chunk[2], chunk[3]];
                if big {
                    f32::from_be_bytes(raw)
                } else {
                    f32::from_le_bytes(raw)
                }
            } else {
                let raw = [
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ];
                (if big {
                    f64::from_be_bytes(raw)
                } else {
                    f64::from_le_bytes(raw)
                }) as f32
            }
        })
        .collect();
    Ok(Array {
        shape,
        fortran,
        values,
    })
}

struct Header<'a> {
    data: &'a str,
    offset: usize,
}
impl Header<'_> {
    fn whitespace(&mut self) {
        while self
            .data
            .as_bytes()
            .get(self.offset)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.offset += 1;
        }
    }
    fn consume(&mut self, c: char) -> bool {
        self.whitespace();
        if self.data[self.offset..].starts_with(c) {
            self.offset += 1;
            true
        } else {
            false
        }
    }
    fn symbol(&mut self, c: char) -> Result<()> {
        if self.consume(c) {
            Ok(())
        } else {
            Err(Error::Invalid("Malformed NPY header".into()))
        }
    }
    fn string(&mut self) -> Result<String> {
        self.whitespace();
        let quote = self
            .data
            .as_bytes()
            .get(self.offset)
            .copied()
            .ok_or_else(|| Error::Invalid("Malformed NPY string".into()))?;
        if quote != b'\'' && quote != b'"' {
            return Err(Error::Invalid("Malformed NPY string".into()));
        }
        self.offset += 1;
        let begin = self.offset;
        while self
            .data
            .as_bytes()
            .get(self.offset)
            .is_some_and(|c| *c != quote)
        {
            self.offset += 1;
        }
        if self.offset == self.data.len() {
            return Err(Error::Invalid("Malformed NPY string".into()));
        }
        let value = self.data[begin..self.offset].to_string();
        self.offset += 1;
        Ok(value)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // Replacing NaN with zero would falsely mark unmeasured reference inputs observed.
    #[test]
    fn sealed_npz_reads_original_float32_missing_values_and_medians() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../evaluation/fixtures/rust_migration/v1/models/xai_reference.npz");
        let metadata: ReferenceMetadata =
            serde_json::from_slice(&std::fs::read(path.with_extension("json")).unwrap()).unwrap();
        let reference = ReferenceData::load(&path, &metadata.feature_names).unwrap();
        assert_eq!(reference.training.len(), 120);
        assert_eq!(reference.medians.len(), 16);
        assert_eq!(
            reference
                .training
                .iter()
                .filter(|row| row[7].is_none())
                .count(),
            20
        );
        assert_eq!(
            reference
                .training
                .iter()
                .filter(|row| row[10].is_none())
                .count(),
            11
        );
        assert_eq!(reference.medians[7], 0.0);
    }

    #[test]
    fn feature_order_mismatch_is_rejected_before_explaining() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../evaluation/fixtures/rust_migration/v1/models/xai_reference.npz");
        let error = ReferenceData::load(&path, &["wrong_order".into()]).unwrap_err();
        assert!(error.to_string().contains("feature order"));
    }

    fn npy(header: &str, values: &[u8]) -> Vec<u8> {
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(values);
        bytes
    }

    // Field order and byte order must not change a frozen reference's values.
    #[test]
    fn numeric_npy_header_key_order_and_big_endian_are_supported() {
        let data: Vec<u8> = [1.25_f32, f32::NAN]
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .collect();
        let array = parse_npy(&npy(
            "{'shape': (2,), 'descr': '>f4', 'fortran_order': False}\n",
            &data,
        ))
        .unwrap();
        assert_eq!(array.shape, vec![2]);
        assert_eq!(array.values[0], 1.25);
        assert!(array.values[1].is_nan());
    }

    #[test]
    fn object_arrays_truncation_and_overflow_shapes_are_rejected() {
        for bytes in [
            npy(
                "{'descr': '|O', 'fortran_order': False, 'shape': (1,)}\n",
                &[],
            ),
            npy(
                "{'descr': '<f4', 'fortran_order': False, 'shape': (2,)}\n",
                &[0; 4],
            ),
            npy(
                "{'descr': '<f4', 'fortran_order': False, 'shape': (18446744073709551615, 2)}\n",
                &[],
            ),
        ] {
            assert!(parse_npy(&bytes).is_err());
        }
    }
}
