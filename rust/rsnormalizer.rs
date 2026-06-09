use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use unidecode::unidecode_char;

/// Normalize text by transliterating to ASCII, converting to lowercase,
/// removing punctuation, and trimming whitespace.
///
/// # Arguments
///
/// * `text` - A string slice that holds the text to normalize.
///
/// # Returns
///
/// A String that holds the normalized text.
#[pyfunction]
pub fn rsnormalize(text: &str) -> String {
    // Single pass: unidecode emits ASCII, so lowercasing and punctuation
    // filtering can happen per char without intermediate allocations.
    let mut normalized = String::with_capacity(text.len());
    for c in text.chars() {
        for ascii in unidecode_char(c).chars() {
            if !ascii.is_ascii_punctuation() {
                normalized.push(ascii.to_ascii_lowercase());
            }
        }
    }

    let trimmed = normalized.trim();
    if trimmed.len() == normalized.len() {
        normalized
    } else {
        trimmed.to_string()
    }
}

/// Normalize multiple texts.
///
/// # Arguments
///
/// * `texts` - A vector of strings that holds the texts to normalize.
///
/// # Returns
///
/// A vector of strings that holds the normalized texts.
#[pyfunction]
pub fn rsnormalize_many(texts: Vec<String>) -> Vec<String> {
    texts.par_iter().map(|text| rsnormalize(text)).collect()
}

pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rsnormalize, m)?)?;
    m.add_function(wrap_pyfunction!(rsnormalize_many, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsnormalize() {
        assert_eq!(rsnormalize("Hello World! 😀"), "hello world");
        assert_eq!(rsnormalize("1,2,3,4"), "1234");
        assert_eq!(rsnormalize("Déjà vu"), "deja vu");
    }

    #[test]
    fn test_rsnormalize_many() {
        let input = vec!["Hello World! 😀".to_string(), "Goodbye, World!".to_string()];
        let expected = vec!["hello world".to_string(), "goodbye world".to_string()];
        assert_eq!(rsnormalize_many(input), expected);
    }
}
