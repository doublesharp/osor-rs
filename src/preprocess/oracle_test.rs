//! Opt-in comparison against a caller-generated, independent Pillow corpus.
use super::*;
use std::io::Read;

fn take<'a>(input: &mut &'a [u8], length: usize) -> Result<&'a [u8]> {
    let (head, tail) = input
        .split_at_checked(length)
        .ok_or_else(|| Error::InvalidInput("truncated preprocessing oracle corpus".into()))?;
    *input = tail;
    Ok(head)
}

fn integer(input: &mut &[u8]) -> Result<u32> {
    let bytes: [u8; 4] = take(input, 4)?
        .try_into()
        .map_err(|_| Error::InvalidInput("invalid oracle integer".into()))?;
    Ok(u32::from_le_bytes(bytes))
}

fn floats(bytes: &[u8]) -> Result<Vec<f32>> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(Error::InvalidInput("invalid oracle float bytes".into()));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(chunks.len())
        .map_err(|_| Error::InvalidInput("cannot allocate oracle float values".into()))?;
    for chunk in chunks {
        let bits: [u8; 4] = chunk
            .try_into()
            .map_err(|_| Error::InvalidInput("invalid oracle float".into()))?;
        values.push(f32::from_le_bytes(bits));
    }
    Ok(values)
}

#[test]
#[ignore = "requires an independently generated Pillow corpus on Scratch"]
fn pillow_oracle_corpus() -> Result<()> {
    let path = std::env::var_os("OSOR_PREPROCESS_ORACLE")
        .ok_or_else(|| Error::InvalidInput("set OSOR_PREPROCESS_ORACLE to the corpus".into()))?;
    let file = std::fs::File::open(path)?;
    const MAX_CORPUS: u64 = 64 * 1024 * 1024;
    let length = file.metadata()?.len();
    if length > MAX_CORPUS {
        return Err(Error::InvalidInput("oracle corpus exceeds 64 MiB".into()));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length as usize)
        .map_err(|_| Error::InvalidInput("cannot allocate oracle corpus".into()))?;
    file.take(MAX_CORPUS + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_CORPUS {
        return Err(Error::InvalidInput("oracle corpus exceeds 64 MiB".into()));
    }
    let mut input = bytes.as_slice();
    if take(&mut input, 8)? != b"OSORP001" {
        return Err(Error::InvalidInput("unknown oracle corpus format".into()));
    }
    let count = integer(&mut input)?;
    let mut cases = [0usize; 8];
    let mut compared = 0usize;
    let mut mismatches = 0usize;
    let mut max_difference = 0u8;
    for case in 0..count {
        let operation = *take(&mut input, 1)?
            .first()
            .ok_or_else(|| Error::InvalidInput("missing oracle operation".into()))?;
        let mut header = [0u32; 11];
        for value in &mut header {
            *value = integer(&mut input)?;
        }
        let [
            sw,
            sh,
            tw,
            th,
            aw,
            ah,
            resolution,
            a_len,
            b_len,
            c_len,
            expected_len,
        ] = header;
        let source = size(sw, sh);
        let target = size(tw, th);
        let auxiliary = size(aw, ah);
        let a = take(&mut input, a_len as usize)?;
        let b = take(&mut input, b_len as usize)?;
        let c = take(&mut input, c_len as usize)?;
        let expected = take(&mut input, expected_len as usize)?;
        let actual = match operation {
            0 => resize_rgb8(a, source, target)?,
            1 => resize_mask8(a, source, target)?,
            2 => resize_alpha8(a, source, target)?,
            3 => reference_output_rgb8(a, source, b, target, None)?,
            4 => reference_output_rgb8(a, source, b, target, Some((c, auxiliary)))?,
            5 => {
                let reference = resize_reference_inputs(a, source, b, auxiliary, resolution)?;
                if reference.size != target {
                    return Err(Error::InvalidInput(format!(
                        "oracle case {case} returned unexpected dimensions"
                    )));
                }
                let mut actual = reference.image_rgb8;
                actual
                    .try_reserve_exact(reference.mask8.len())
                    .map_err(|_| Error::InvalidInput("cannot allocate oracle comparison".into()))?;
                actual.extend_from_slice(&reference.mask8);
                actual
            }
            6 => quantize_rgb_chw_f32(&floats(a)?, source)?,
            7 => quantize_alpha_f32(&floats(a)?, source)?,
            _ => return Err(Error::InvalidInput("unknown oracle operation".into())),
        };
        if let Some(operation_count) = cases.get_mut(usize::from(operation)) {
            *operation_count += 1;
        }
        if actual.len() != expected.len() {
            return Err(Error::InvalidInput(format!(
                "oracle case {case} operation {operation}: expected {} bytes, got {}",
                expected.len(),
                actual.len()
            )));
        }
        for (offset, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            compared += 1;
            if actual != expected {
                mismatches += 1;
                max_difference = max_difference.max(actual.abs_diff(expected));
                if mismatches <= 12 {
                    eprintln!(
                        "case={case} operation={operation} source={source:?} target={target:?} \
                         auxiliary={auxiliary:?} offset={offset} expected={expected} actual={actual}"
                    );
                }
            }
        }
    }
    if !input.is_empty() {
        return Err(Error::InvalidInput("trailing oracle corpus bytes".into()));
    }
    println!(
        "Pillow oracle cases={count} by_operation={cases:?} bytes={compared} \
         mismatched_bytes={mismatches} max_byte_difference={max_difference}"
    );
    assert_eq!(mismatches, 0, "Pillow oracle byte mismatches");
    Ok(())
}
