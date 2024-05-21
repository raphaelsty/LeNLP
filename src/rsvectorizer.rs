use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use std::collections::HashMap;

use crate::rsanalyzer::rschar_ngrams;
use crate::rsanalyzer::rschar_wb_ngrams;
use crate::rsnormalizer::rsnormalize;
use crate::rscounter::rscount;

#[pyfunction]
pub fn rsvectorize_char_wb_ngrams(text: &str, n_sizes: Vec<usize>)  -> HashMap<String, usize> {
    rscount(rschar_wb_ngrams(&rsnormalize(text), n_sizes))
}

#[pyfunction]
pub fn rsvectorize_char_wb_ngrams_many(texts: Vec<String>, n_sizes: Vec<usize>) -> Vec<HashMap<String, usize>> {
    texts.par_iter().map(|text| rsvectorize_char_wb_ngrams(text, n_sizes.clone())).collect()
}

#[pyfunction]
pub fn rsvectorize_char_ngrams(text: &str, n_sizes: Vec<usize>)  -> HashMap<String, usize> {
    rscount(rschar_ngrams(&rsnormalize(text), n_sizes))
}

#[pyfunction]
pub fn rsvectorize_char_ngrams_many(texts: Vec<String>, n_sizes: Vec<usize>) -> Vec<HashMap<String, usize>> {
    texts.par_iter().map(|text| rsvectorize_char_ngrams(text, n_sizes.clone())).collect()
}

pub fn register_functions(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rsvectorize_char_wb_ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(rsvectorize_char_wb_many, m)?)?;
    Ok(())
}
