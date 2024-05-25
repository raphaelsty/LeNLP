use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use std::collections::HashSet;

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
    // Use HashSet for better performance in membership checks
    let stop_words_set: HashSet<_> = stop_words.into_iter().collect();
    text.split_whitespace()
        .filter(|word: &&str| !stop_words_set.contains(*word))
        .collect::<Vec<&str>>()
        .join(" ")
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
    // Use HashSet for better performance in membership checks
    let stop_words_set: HashSet<_> = stop_words.into_iter().collect();
    texts
        .into_par_iter()
        .map(|sentence: String| {
            sentence
                .split_whitespace()
                .filter(|word: &&str| !stop_words_set.contains(*word))
                .collect::<Vec<&str>>()
                .join(" ")
        })
        .collect()
}

pub fn register_functions(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rsfilter_stop_words, m)?)?;
    m.add_function(wrap_pyfunction!(rsfilter_stop_words_many, m)?)?;
    Ok(())
}
