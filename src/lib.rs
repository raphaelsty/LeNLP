use pyo3::prelude::*;

mod rsnormalizer;
mod rsvectorizer;
mod rsanalyzer;
mod rscounter;

#[pymodule]
fn rsantelope(_py: Python, m: &PyModule) -> PyResult<()> {
    rsnormalizer::register_functions(m)?;
    rsvectorizer::register_functions(m)?;
    rsanalyzer::register_functions(m)?;
    rscounter::register_functions(m)?;
    Ok(())
}
