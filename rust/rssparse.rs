use crate::rsvectorizer::rsvectorize_many;
use bincode::{deserialize, serialize};
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// In order to properly pickle, we need to map to the Python module.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[pyclass(module = "lenlp.sparse.count_vectorizer")]
pub struct SparseMatrixBuilder {
    analyzer: String,
    n_sizes: Vec<usize>,
    stop_words: Option<Vec<String>>,
    normalize: Option<bool>,
    vocab: HashMap<String, usize>,
    num_cols: usize,
}

#[pymethods]
impl SparseMatrixBuilder {
    #[new]
    pub fn new(
        n_sizes: Vec<usize>,
        analyzer: String,
        stop_words: Option<Vec<String>>,
        normalize: Option<bool>,
    ) -> Self {
        SparseMatrixBuilder {
            vocab: HashMap::new(),
            n_sizes,
            analyzer,
            stop_words,
            normalize,
            num_cols: 0,
        }
    }

    pub fn fit_transform(
        &mut self,
        texts: Vec<String>,
        py: Python,
    ) -> (
        Py<PyArray1<usize>>,
        Py<PyArray1<usize>>,
        Py<PyArray1<usize>>,
    ) {
        self.vocab = HashMap::new();
        let texts: Vec<HashMap<String, usize>> = rsvectorize_many(
            texts,
            self.n_sizes.clone(),
            self.analyzer.clone(),
            self.stop_words.clone(),
            self.normalize,
        );

        self._fit(texts.clone());

        // Scipy csr_matrix are faster to build from numpy arrays.
        let (vec1, vec2, vec3) = self._transform(texts);
        (
            PyArray1::from_vec_bound(py, vec1).into(),
            PyArray1::from_vec_bound(py, vec2).into(),
            PyArray1::from_vec_bound(py, vec3).into(),
        )
    }

    pub fn fit(&mut self, texts: Vec<String>) {
        self.vocab = HashMap::new();
        let texts: Vec<HashMap<String, usize>> = rsvectorize_many(
            texts,
            self.n_sizes.clone(),
            self.analyzer.clone(),
            self.stop_words.clone(),
            self.normalize,
        );

        self._fit(texts);
    }

    fn _fit(&mut self, texts: Vec<HashMap<String, usize>>) {
        let mut col_index: usize = 0;
        for doc in &texts {
            for token in doc.keys() {
                if !self.vocab.contains_key(token) {
                    self.vocab.insert(token.clone(), col_index);
                    col_index += 1;
                }
            }
        }
        self.num_cols = col_index;
    }

    pub fn transform(
        &self,
        texts: Vec<String>,
        py: Python,
    ) -> (
        Py<PyArray1<usize>>,
        Py<PyArray1<usize>>,
        Py<PyArray1<usize>>,
    ) {
        let texts: Vec<HashMap<String, usize>> = rsvectorize_many(
            texts,
            self.n_sizes.clone(),
            self.analyzer.clone(),
            self.stop_words.clone(),
            self.normalize,
        );

        // Scipy csr_matrix are faster to build from numpy arrays.
        let (vec1, vec2, vec3) = self._transform(texts);
        (
            PyArray1::from_vec_bound(py, vec1).into(),
            PyArray1::from_vec_bound(py, vec2).into(),
            PyArray1::from_vec_bound(py, vec3).into(),
        )
    }

    fn _transform(
        &self,
        texts: Vec<HashMap<String, usize>>,
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let mut values: Vec<usize> = Vec::new();
        let mut row_indices: Vec<usize> = Vec::new();
        let mut column_indices: Vec<usize> = Vec::new();

        for (row_idx, doc) in texts.iter().enumerate() {
            for (token, &count) in doc.iter() {
                if let Some(&col_idx) = self.vocab.get(token) {
                    values.push(count);
                    row_indices.push(row_idx);
                    column_indices.push(col_idx);
                }
            }
        }

        (values, row_indices, column_indices)
    }

    pub fn get_vocab(&self) -> HashMap<String, usize> {
        self.vocab.clone()
    }

    pub fn get_num_cols(&self) -> usize {
        self.num_cols
    }

    pub fn __setstate__(&mut self, state: &PyBytes) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
        Ok(PyBytes::new(py, &serialize(&self).unwrap()))
    }

    pub fn __getnewargs__(
        &self,
    ) -> PyResult<(Vec<usize>, String, Option<Vec<String>>, Option<bool>)> {
        Ok((
            self.n_sizes.clone(),
            self.analyzer.clone(),
            self.stop_words.clone(),
            self.normalize,
        ))
    }
}

pub fn register_functions(m: &PyModule) -> PyResult<()> {
    m.add_class::<SparseMatrixBuilder>()?;
    Ok(())
}
