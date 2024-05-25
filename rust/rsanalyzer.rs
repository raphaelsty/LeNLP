use pyo3::prelude::*;
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

/// Splits text into words.
///
/// # Arguments
///
/// * `texts` - The input texts.
/// * `n_sizes` - The size of the n-grams.
///
/// # Returns
///
/// A vector of words.
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

    ngrams.into_iter().collect()
}

/// Computes character n-grams for multiple texts.
///
/// # Arguments
///
///
/// * `texts` - A vector of input texts.
/// * `n_sizes` - The size of the n-grams.
///
/// # Returns
///
/// A vector of vectors of character n-grams.
#[pyfunction]
pub fn rschar_ngrams_many(texts: Vec<String>, n_sizes: Vec<usize>) -> Vec<Vec<String>> {
    texts
        .par_iter()
        .map(|text: &String| rschar_ngrams(text, n_sizes.clone()))
        .collect()
}

/// Computes character n-grams with word boundaries.
///
/// # Arguments
///
/// * `text` - The input text.
/// * `n_sizes` - The size of the n-grams.
///
/// # Returns
///
/// A vector of character n-grams with word boundaries.
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
/// Computes character n-grams with word boundaries for multiple texts.
///
/// # Arguments
///
/// * `texts` - A vector of input texts.
/// * `n_sizes` - The size of the n-grams.
///
/// # Returns
///
/// A vector of vectors of character n-grams with word boundaries.
#[pyfunction]
pub fn rschar_wb_ngrams_many(texts: Vec<String>, n_sizes: Vec<usize>) -> Vec<Vec<String>> {
    texts
        .par_iter()
        .map(|text: &String| rschar_wb_ngrams(text, n_sizes.clone()))
        .collect()
}

pub fn register_functions(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rssplit_words, m)?)?;
    m.add_function(wrap_pyfunction!(rssplit_words_many, m)?)?;
    m.add_function(wrap_pyfunction!(rschar_ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(rschar_ngrams_many, m)?)?;
    m.add_function(wrap_pyfunction!(rschar_wb_ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(rschar_wb_ngrams_many, m)?)?;
    Ok(())
}
