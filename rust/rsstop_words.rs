use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use rustc_hash::FxHashSet;

fn filter(text: &str, stop_words: &FxHashSet<String>) -> String {
    let mut filtered = String::with_capacity(text.len());
    for word in text.split_whitespace() {
        if !stop_words.contains(word) {
            if !filtered.is_empty() {
                filtered.push(' ');
            }
            filtered.push_str(word);
        }
    }
    filtered
}

/// Function to filter stop words from a string
///
/// # Arguments
///
/// * `text` - The input text.
/// * `stop_words` - The stop words to filter.
///
/// # Returns
///
/// A string with the stop words removed.
#[pyfunction]
pub fn rsfilter_stop_words(text: &str, stop_words: Vec<String>) -> String {
    let stop_words: FxHashSet<String> = stop_words.into_iter().collect();
    filter(text, &stop_words)
}

/// Function to filter stop words from multiple strings
///
/// # Arguments
///
/// * `texts` - The input texts.
/// * `stop_words` - The stop words to filter.
///
/// # Returns
///
/// A vector of strings with the stop words removed.
#[pyfunction]
pub fn rsfilter_stop_words_many(texts: Vec<String>, stop_words: Vec<String>) -> Vec<String> {
    let stop_words: FxHashSet<String> = stop_words.into_iter().collect();
    texts
        .par_iter()
        .map(|text| filter(text, &stop_words))
        .collect()
}

pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rsfilter_stop_words, m)?)?;
    m.add_function(wrap_pyfunction!(rsfilter_stop_words_many, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsfilter_stop_words() {
        assert_eq!(
            rsfilter_stop_words("the quick brown fox", vec!["the".to_string()]),
            "quick brown fox"
        );
    }
}
