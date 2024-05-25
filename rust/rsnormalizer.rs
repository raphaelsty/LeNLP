use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use unidecode::unidecode;

/// Normalize text by converting to lowercase, removing punctuation, and trimming whitespace.
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
    unidecode(text)
        .to_lowercase()
        .chars()
        .filter(|c| !c.is_ascii_punctuation())
        .collect::<String>()
        .trim()
        .to_string()
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

pub fn register_functions(m: &PyModule) -> PyResult<()> {
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
    }

    #[test]
    fn test_rsnormalize_many() {
        let input = vec!["Hello World! 😀".to_string(), "Goodbye, World!".to_string()];
        let expected = vec!["hello world".to_string(), "goodbye world".to_string()];
        assert_eq!(rsnormalize_many(input), expected);
    }
}
