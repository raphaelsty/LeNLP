use pyo3::prelude::*;
use pyo3::types::PyModule; // NEW

mod rsanalyzer;
mod rscounter;
mod rsflashtext;
mod rsnormalizer;
mod rssparse;
mod rsstop_words;
mod rsvectorizer;

#[pymodule]
fn _rslenlp(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    rsanalyzer::register_functions(m)?;
    rscounter::register_functions(m)?;
    rsflashtext::register_functions(m)?;
    rsnormalizer::register_functions(m)?;
    rssparse::register_functions(m)?;
    rsstop_words::register_functions(m)?;
    rsvectorizer::register_functions(m)?;
    Ok(())
}
