//! Bounded decoder for the UBJSON markers used by native XGBoost checkpoints.
//! Optimized typed/count arrays are decoded directly; model bytes are not converted.
use crate::Error;
use serde_json::{Map, Number, Value};

const MAX_ITEMS: usize = 16_000_000;

pub fn decode(bytes: &[u8]) -> Result<Value, Error> {
    let mut reader = Reader { bytes, position: 0 };
    let value = reader.value(None, 0)?;
    if reader.position != bytes.len() {
        return Err(Error::Invalid(
            "Trailing bytes after UBJSON document".into(),
        ));
    }
    Ok(value)
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl Reader<'_> {
    fn byte(&mut self) -> Result<u8, Error> {
        let result = *self
            .bytes
            .get(self.position)
            .ok_or_else(|| Error::Invalid("Truncated UBJSON".into()))?;
        self.position += 1;
        Ok(result)
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        let end = self
            .position
            .checked_add(N)
            .ok_or_else(|| Error::Invalid("UBJSON size overflow".into()))?;
        let data = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| Error::Invalid("Truncated UBJSON scalar".into()))?;
        self.position = end;
        data.try_into()
            .map_err(|_| Error::Invalid("Invalid scalar width".into()))
    }

    fn integer(&mut self, tag: u8) -> Result<i64, Error> {
        match tag {
            b'i' => Ok(i8::from_be_bytes(self.take()?) as i64),
            b'U' => Ok(self.byte()? as i64),
            b'I' => Ok(i16::from_be_bytes(self.take()?) as i64),
            b'l' => Ok(i32::from_be_bytes(self.take()?) as i64),
            b'L' => Ok(i64::from_be_bytes(self.take()?)),
            _ => Err(Error::Invalid(
                "UBJSON length requires an integer marker".into(),
            )),
        }
    }

    fn length(&mut self) -> Result<usize, Error> {
        let tag = self.byte()?;
        let length = usize::try_from(self.integer(tag)?)
            .map_err(|_| Error::Invalid("Negative UBJSON length".into()))?;
        if length > MAX_ITEMS {
            return Err(Error::Invalid("UBJSON item bound exceeded".into()));
        }
        Ok(length)
    }

    fn text(&mut self) -> Result<String, Error> {
        let length = self.length()?;
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| Error::Invalid("UBJSON text overflow".into()))?;
        let data = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| Error::Invalid("Truncated UBJSON text".into()))?;
        let text = std::str::from_utf8(data)
            .map_err(|_| Error::Invalid("Invalid UTF-8 in UBJSON".into()))?
            .to_owned();
        self.position = end;
        Ok(text)
    }

    fn float(number: f64) -> Result<Value, Error> {
        Number::from_f64(number)
            .map(Value::Number)
            .ok_or_else(|| Error::Invalid("Nonfinite checkpoint number".into()))
    }

    fn value(&mut self, kind: Option<u8>, depth: usize) -> Result<Value, Error> {
        if depth > 64 {
            return Err(Error::Invalid("UBJSON nesting bound exceeded".into()));
        }
        let tag = match kind {
            Some(tag) => tag,
            None => self.byte()?,
        };
        match tag {
            b'Z' => Ok(Value::Null),
            b'T' => Ok(Value::Bool(true)),
            b'F' => Ok(Value::Bool(false)),
            b'i' | b'U' | b'I' | b'l' | b'L' => Ok(Value::Number(self.integer(tag)?.into())),
            b'd' => Self::float(f32::from_be_bytes(self.take()?) as f64),
            b'D' => Self::float(f64::from_be_bytes(self.take()?)),
            b'S' => Ok(Value::String(self.text()?)),
            b'[' | b'{' => self.container(tag == b'{', depth + 1),
            _ => Err(Error::Invalid(format!("Unsupported UBJSON marker {tag}"))),
        }
    }

    fn container(&mut self, object: bool, depth: usize) -> Result<Value, Error> {
        let typed = if self.peek() == Some(b'$') {
            self.byte()?;
            Some(self.byte()?)
        } else {
            None
        };
        let count = if self.peek() == Some(b'#') {
            self.byte()?;
            Some(self.length()?)
        } else {
            None
        };
        if typed.is_some() && count.is_none() {
            return Err(Error::Invalid("Typed UBJSON container has no count".into()));
        }
        let terminal = if object { b'}' } else { b']' };
        let mut items = Vec::new();
        let mut fields = Map::new();
        let mut seen = 0;
        loop {
            if count == Some(seen) {
                break;
            }
            if count.is_none() && self.peek() == Some(terminal) {
                self.byte()?;
                break;
            }
            if seen >= MAX_ITEMS {
                return Err(Error::Invalid("UBJSON container bound exceeded".into()));
            }
            if object {
                let key = self.text()?;
                let value = self.value(typed, depth)?;
                if fields.insert(key, value).is_some() {
                    return Err(Error::Invalid("Duplicate UBJSON key".into()));
                }
            } else {
                items.push(self.value(typed, depth)?);
            }
            seen += 1;
        }
        Ok(if object {
            Value::Object(fields)
        } else {
            Value::Array(items)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn optimized_arrays_and_truncation() {
        assert_eq!(
            decode(b"[$U#i\x03\x01\x02\x03").unwrap(),
            serde_json::json!([1, 2, 3])
        );
        assert!(decode(b"[$l#i\x02\x00").is_err());
        assert!(decode(b"[#i\xff").is_err());
        assert!(decode(b"[]x").is_err());
    }
}
