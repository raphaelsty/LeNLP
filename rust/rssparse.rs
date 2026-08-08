use crate::rsvectorizer::rsvectorize_many;
use bincode::{deserialize, serialize};
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Sparse-matrix builder
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[pyclass(module = "lenlp.sparse.count_vectorizer", skip_from_py_object)]
pub struct SparseMatrixBuilder {
    analyzer: String,
    n_sizes: Vec<usize>,
    stop_words: Option<Vec<String>>,
    normalize: Option<bool>,
    vocab: FxHashMap<String, usize>,
    num_cols: usize,
}

/// CSR arrays as returned to Python: (data, indices, indptr).
type CsrArrays = (Py<PyArray1<f32>>, Py<PyArray1<i32>>, Py<PyArray1<i32>>);

#[pymethods]
impl SparseMatrixBuilder {
    #[new]
    pub fn new(
        n_sizes: Vec<usize>,
        analyzer: String,
        stop_words: Option<Vec<String>>,
        normalize: Option<bool>,
    ) -> Self {
        Self {
            vocab: FxHashMap::default(),
            n_sizes,
            analyzer,
            stop_words,
            normalize,
            num_cols: 0,
        }
    }

    /// Build the vocabulary and return the CSR arrays.
    pub fn fit_transform(&mut self, texts: Vec<String>, py: Python<'_>) -> CsrArrays {
        let docs = self.vectorize(texts);
        self._fit(&docs);
        self.to_numpy(self._transform(&docs), py)
    }

    pub fn fit(&mut self, texts: Vec<String>) {
        let docs = self.vectorize(texts);
        self._fit(&docs);
    }

    pub fn transform(&self, texts: Vec<String>, py: Python<'_>) -> CsrArrays {
        let docs = self.vectorize(texts);
        self.to_numpy(self._transform(&docs), py)
    }

    // ---------------------------------------------------------------------
    // Accessors
    // ---------------------------------------------------------------------

    pub fn get_vocab(&self) -> FxHashMap<String, usize> {
        self.vocab.clone()
    }

    pub fn get_num_cols(&self) -> usize {
        self.num_cols
    }

    // ---------------------------------------------------------------------
    // Pickle support
    // ---------------------------------------------------------------------

    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
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

impl SparseMatrixBuilder {
    fn vectorize(&self, texts: Vec<String>) -> Vec<FxHashMap<String, usize>> {
        rsvectorize_many(
            texts,
            self.n_sizes.clone(),
            self.analyzer.clone(),
            self.stop_words.clone(),
            self.normalize,
        )
    }

    fn _fit(&mut self, docs: &[FxHashMap<String, usize>]) {
        self.vocab = FxHashMap::default();
        for doc in docs {
            for token in doc.keys() {
                if !self.vocab.contains_key(token) {
                    self.vocab.insert(token.clone(), self.vocab.len());
                }
            }
        }
        self.num_cols = self.vocab.len();
    }

    /// Build CSR arrays from the vectorized documents.
    ///
    /// Tokens missing from the vocabulary are dropped; column indices are
    /// sorted within each row so scipy can use the arrays as-is.
    fn _transform(&self, docs: &[FxHashMap<String, usize>]) -> (Vec<f32>, Vec<i32>, Vec<i32>) {
        let rows: Vec<Vec<(i32, f32)>> = docs
            .par_iter()
            .map(|doc| {
                let mut row: Vec<(i32, f32)> = doc
                    .iter()
                    .filter_map(|(token, &count)| {
                        self.vocab
                            .get(token)
                            .map(|&col| (col as i32, count as f32))
                    })
                    .collect();
                row.sort_unstable_by_key(|&(col, _)| col);
                row
            })
            .collect();

        let nnz: usize = rows.iter().map(Vec::len).sum();
        let mut data: Vec<f32> = Vec::with_capacity(nnz);
        let mut indices: Vec<i32> = Vec::with_capacity(nnz);
        let mut indptr: Vec<i32> = Vec::with_capacity(rows.len() + 1);

        indptr.push(0);
        for row in &rows {
            for &(col, value) in row {
                indices.push(col);
                data.push(value);
            }
            indptr.push(indices.len() as i32);
        }

        (data, indices, indptr)
    }

    fn to_numpy(&self, csr: (Vec<f32>, Vec<i32>, Vec<i32>), py: Python<'_>) -> CsrArrays {
        let (data, indices, indptr) = csr;
        (
            PyArray1::from_vec(py, data).into(),
            PyArray1::from_vec(py, indices).into(),
            PyArray1::from_vec(py, indptr).into(),
        )
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SparseMatrixBuilder>()?;
    Ok(())
}
