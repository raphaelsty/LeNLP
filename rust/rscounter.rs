use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use std::collections::HashMap;

/// Counts the number of times each word appears in the input text.
///
/// # Arguments
///
/// * `text` - The input text as a vector of words.
///
/// # Returns
///
/// A hashmap with the words as keys and the number of times they appear as values.
#[pyfunction]
pub fn rscount(text: Vec<String>) -> HashMap<String, usize> {
    let mut word_counter = HashMap::new();
    for word in text {
        *word_counter.entry(word).or_insert(0) += 1;
    }
    word_counter
}

/// Counts the number of times each word appears for each input text.
///
/// # Arguments
///
/// * `texts` - The input texts as a vector of vectors of words.
///
/// # Returns
///
/// A vector of hashmaps with the words as keys and the number of times they appear as values.
#[pyfunction]
pub fn rscount_many(texts: Vec<Vec<String>>) -> Vec<HashMap<String, usize>> {
    texts.par_iter().map(|text| rscount(text.clone())).collect()
}

pub fn register_functions(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rscount, m)?)?;
    m.add_function(wrap_pyfunction!(rscount_many, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rscount() {
        let text = vec![
            "hello".to_string(),
            "world".to_string(),
            "hello".to_string(),
            "hello".to_string(),
        ];
        let result = rscount(text);
        let mut expected = HashMap::new();
        expected.insert("hello".to_string(), 3);
        expected.insert("world".to_string(), 1);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_rscount_many() {
        let texts = vec![
            vec!["hello".to_string(), "world".to_string()],
            vec![
                "hello".to_string(),
                "world".to_string(),
                "hello".to_string(),
            ],
        ];
        let result = rscount_many(texts);
        let mut expected = Vec::new();
        let mut map1 = HashMap::new();
        map1.insert("hello".to_string(), 1);
        map1.insert("world".to_string(), 1);
        let mut map2 = HashMap::new();
        map2.insert("hello".to_string(), 2);
        map2.insert("world".to_string(), 1);
        expected.push(map1);
        expected.push(map2);
        assert_eq!(result, expected);
    }
}
