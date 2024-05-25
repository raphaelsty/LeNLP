use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::rsanalyzer::rschar_ngrams;
use crate::rsanalyzer::rschar_wb_ngrams;
use crate::rsanalyzer::rssplit_words;
use crate::rscounter::rscount;
use crate::rsnormalizer::rsnormalize_many;
use crate::rsstop_words::rsfilter_stop_words_many;

pub fn process_texts(
    texts: Vec<String>,
    normalize: Option<bool>,
    stop_words: Option<Vec<String>>,
) -> Vec<String> {
    let texts: Vec<String> = match normalize {
        Some(true) => rsnormalize_many(texts),
        _ => texts,
    };

    match stop_words {
        Some(stop_words) => rsfilter_stop_words_many(texts, stop_words),
        None => texts,
    }
}

#[pyfunction]
pub fn rsvectorize_split_words_many(
    texts: Vec<String>,
    n_sizes: Vec<usize>,
    stop_words: Option<Vec<String>>,
    normalize: Option<bool>,
) -> Vec<HashMap<String, usize>> {
    let texts: Vec<String> = process_texts(texts, normalize, stop_words);
    texts
        .par_iter()
        .map(|text: &String| rscount(rssplit_words(text, n_sizes.clone())))
        .collect()
}

#[pyfunction]
pub fn rsvectorize_char_ngrams_many(
    texts: Vec<String>,
    n_sizes: Vec<usize>,
    stop_words: Option<Vec<String>>,
    normalize: Option<bool>,
) -> Vec<HashMap<String, usize>> {
    let texts: Vec<String> = process_texts(texts, normalize, stop_words);
    texts
        .par_iter()
        .map(|text: &String| rscount(rschar_ngrams(text, n_sizes.clone())))
        .collect()
}

#[pyfunction]
pub fn rsvectorize_char_wb_ngrams_many(
    texts: Vec<String>,
    n_sizes: Vec<usize>,
    stop_words: Option<Vec<String>>,
    normalize: Option<bool>,
) -> Vec<HashMap<String, usize>> {
    let texts: Vec<String> = process_texts(texts, normalize, stop_words);
    texts
        .par_iter()
        .map(|text: &String| rscount(rschar_wb_ngrams(text, n_sizes.clone())))
        .collect()
}

// Main vectorization function
#[pyfunction]
pub fn rsvectorize_many(
    texts: Vec<String>,
    n_sizes: Vec<usize>,
    analyzer: String,
    stop_words: Option<Vec<String>>,
    normalize: Option<bool>,
) -> Vec<HashMap<String, usize>> {
    match analyzer.as_str() {
        "word" => rsvectorize_split_words_many(texts, n_sizes, stop_words, normalize),
        "char" => rsvectorize_char_ngrams_many(texts, n_sizes, stop_words, normalize),
        "char_wb" => rsvectorize_char_wb_ngrams_many(texts, n_sizes, stop_words, normalize),
        _ => panic!("Invalid analyzer type"),
    }
}

pub fn register_functions(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rsvectorize_split_words_many, m)?)?;
    m.add_function(wrap_pyfunction!(rsvectorize_char_ngrams_many, m)?)?;
    m.add_function(wrap_pyfunction!(rsvectorize_char_wb_ngrams_many, m)?)?;
    m.add_function(wrap_pyfunction!(rsvectorize_many, m)?)?;

    Ok(())
}
