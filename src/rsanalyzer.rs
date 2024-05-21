use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use std::collections::HashSet;

/// Splits text into words.
///
/// # Arguments
///
/// * `text` - The input text.
///
/// # Returns
///
/// A vector of words.
#[pyfunction]
pub fn rssplit_words(text: &str) -> Vec<String> {
    text.split_whitespace().map(|s| s.to_string()).collect()
}

#[pyfunction]
pub fn rssplit_words_many(texts: Vec<String>) -> Vec<Vec<String>> {
    texts.par_iter().map(|text| rssplit_words(text)).collect()
}

/// Computes character n-grams.
///
/// # Arguments
///
/// * `text` - The input text.
/// * `n_sizes` - The size of the n-grams.
///
/// # Returns
///
/// A vector of character n-grams.
#[pyfunction]
fn rschar_ngrams(text: &str, n_sizes: Vec<usize>) -> Vec<String> {
    let mut ngrams = Vec::new();

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
    texts.par_iter().map(|text| rschar_ngrams(text, n_sizes.clone())).collect()
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
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut ngrams = Vec::new();

    for word in words {
        let chars: Vec<char> = word.chars().collect();
        for &n in &n_sizes {
            for window in chars.windows(n) {
                ngrams.push(window.iter().collect::<String>());
            }
        }
    }

    ngrams.into_iter().collect()
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
    texts.par_iter().map(|text| rschar_wb_ngrams(text, n_sizes.clone())).collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rschar_ngrams() {
        let text = "hello world";
        let n_sizes = vec![3];
        let mut result = rschar_ngrams(text, n_sizes);
        result.sort();
        let expected = vec![
            " wo".to_string(),
            "ell".to_string(),
            "hel".to_string(),
            "llo".to_string(),
            "lo ".to_string(),
            "o w".to_string(),
            "orl".to_string(),
            "rld".to_string(),
            "wor".to_string(),
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_rschar_wb_ngrams() {
        let text = "hello world";
        let n_sizes = vec![3];
        let mut result = rschar_wb_ngrams(text, n_sizes);
        result.sort();
        let expected: HashSet<String> = vec![
            "rld".to_string(),
            "wor".to_string(),
            "ell".to_string(),
            "llo".to_string(),
            "orl".to_string(),
            "hel".to_string(),
        ]
        .into_iter()
        .collect();
        assert_eq!(result.into_iter().collect::<HashSet<_>>(), expected);
    }

    #[test]
    fn test_rschar_wb_ngrams_many() {
        let input = vec!["hello world".to_string(), "goodbye world".to_string()];
        let n_sizes = vec![3];
        let mut result = rschar_wb_ngrams_many(input, n_sizes);
        let expected: HashSet<Vec<String>> = vec![
            vec!["ell".to_string(), "llo".to_string(), "rld".to_string(), "wor".to_string()],
            vec!["bye".to_string(), "goo".to_string(), "rld".to_string(), "wor".to_string()],
        ]
        .into_iter()
        .collect();
        assert_eq!(result.into_iter().collect::<HashSet<_>>(), expected);
    }

    #[test]
    fn test_rssplit_words() {
        let text = "hello world";
        let mut result = rssplit_words(text);
        result.sort();
        let expected: HashSet<String> = vec![
            "hello".to_string(),
            "world".to_string(),
        ]
        .into_iter()
        .collect();
        assert_eq!(result.into_iter().collect::<HashSet<_>>(), expected);
    }

    #[test]
    fn test_rssplit_words_many() {
        let input = vec!["hello world".to_string(), "goodbye world".to_string()];
        let mut result = rssplit_words_many(input);
        result.sort();
        let expected: HashSet<Vec<String>> = vec![
            vec!["hello".to_string(), "world".to_string()],
            vec!["goodbye".to_string(), "world".to_string()],
        ]
        .into_iter()
        .collect();
        assert_eq!(result.into_iter().collect::<HashSet<_>>(), expected);
    }
}
