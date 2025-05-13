use pyo3::prelude::*;
use pyo3::types::PyModule; // <- NEW: we now use the smart-pointer Bound<…, PyModule>
use pyo3::wrap_pyfunction;
use rayon::prelude::*;

/// Splits text into words.
///
/// # Arguments
///
/// * `text` - The input text.
/// * `n_sizes` - The size of the n-grams.
///
/// # Returns
///
/// A vector of words.
#[pyfunction]
pub fn rssplit_words(text: &str, n_sizes: Vec<usize>) -> Vec<String> {
    let mut ngrams: Vec<String> = Vec::new();

    for &n in &n_sizes {
        let words: Vec<&str> = text.split_whitespace().collect();
        for window in words.windows(n) {
            ngrams.push(window.join(" "));
        }
    }

    ngrams
}

/// Same as `rssplit_words` but for many texts at once.
#[pyfunction]
pub fn rssplit_words_many(texts: Vec<String>, n_sizes: Vec<usize>) -> Vec<Vec<String>> {
    texts
        .par_iter()
        .map(|text: &String| rssplit_words(text, n_sizes.clone()))
        .collect()
}

/// Computes character n-grams.
///
/// # Arguments
///
/// * `texts` - A vector of input texts.
/// * `n_sizes` - The size of the n-grams.
///
/// # Returns
///
/// A vector of character n-grams.
#[pyfunction]
pub fn rschar_ngrams(text: &str, n_sizes: Vec<usize>) -> Vec<String> {
    let mut ngrams: Vec<String> = Vec::new();

    for &n in &n_sizes {
        let chars: Vec<char> = text.chars().collect();
        for window in chars.windows(n) {
            ngrams.push(window.iter().collect::<String>());
        }
    }

    ngrams
}

/// Same as `rschar_ngrams` but for many texts at once.
#[pyfunction]
pub fn rschar_ngrams_many(texts: Vec<String>, n_sizes: Vec<usize>) -> Vec<Vec<String>> {
    texts
        .par_iter()
        .map(|text: &String| rschar_ngrams(text, n_sizes.clone()))
        .collect()
}

/// Character n-grams with word-boundary handling.
#[pyfunction]
pub fn rschar_wb_ngrams(text: &str, n_sizes: Vec<usize>) -> Vec<String> {
    let mut ngrams: Vec<String> = Vec::new();
    let chars: Vec<char> = text.chars().collect();

    for &n in &n_sizes {
        if n > chars.len() {
            continue;
        }
        for window in chars.windows(n) {
            ngrams.push(window.iter().collect::<String>());
        }
    }

    ngrams
}

/// Same as `rschar_wb_ngrams` but for many texts at once.
#[pyfunction]
pub fn rschar_wb_ngrams_many(texts: Vec<String>, n_sizes: Vec<usize>) -> Vec<Vec<String>> {
    texts
        .par_iter()
        .map(|text: &String| rschar_wb_ngrams(text, n_sizes.clone()))
        .collect()
}

/// Registers all the above functions in a Python sub-module.
///
/// Called from your `#[pymodule]` entry-point.
pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rssplit_words, m)?)?;
    m.add_function(wrap_pyfunction!(rssplit_words_many, m)?)?;
    m.add_function(wrap_pyfunction!(rschar_ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(rschar_ngrams_many, m)?)?;
    m.add_function(wrap_pyfunction!(rschar_wb_ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(rschar_wb_ngrams_many, m)?)?;
    Ok(())
}
