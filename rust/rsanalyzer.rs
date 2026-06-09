use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;

/// Byte offsets of every char boundary in `text`, with `text.len()` appended.
///
/// Slicing `text[boundaries[i]..boundaries[i + n]]` yields the n-gram of `n`
/// chars starting at char `i` without re-walking the string.
pub fn char_boundaries(text: &str) -> Vec<usize> {
    let mut boundaries: Vec<usize> = Vec::with_capacity(text.len() + 1);
    boundaries.extend(text.char_indices().map(|(offset, _)| offset));
    boundaries.push(text.len());
    boundaries
}

fn split_words(text: &str, n_sizes: &[usize]) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let capacity: usize = n_sizes
        .iter()
        .map(|&n| (words.len() + 1).saturating_sub(n))
        .sum();

    let mut ngrams: Vec<String> = Vec::with_capacity(capacity);
    for &n in n_sizes {
        if n == 0 {
            continue;
        }
        for window in words.windows(n) {
            ngrams.push(window.join(" "));
        }
    }
    ngrams
}

fn char_ngrams(text: &str, n_sizes: &[usize]) -> Vec<String> {
    let boundaries = char_boundaries(text);
    let num_chars = boundaries.len() - 1;
    let capacity: usize = n_sizes
        .iter()
        .map(|&n| (num_chars + 1).saturating_sub(n))
        .sum();

    let mut ngrams: Vec<String> = Vec::with_capacity(capacity);
    for &n in n_sizes {
        if n == 0 || n > num_chars {
            continue;
        }
        for start in 0..=(num_chars - n) {
            ngrams.push(text[boundaries[start]..boundaries[start + n]].to_string());
        }
    }
    ngrams
}

/// Splits text into word n-grams.
///
/// # Arguments
///
/// * `text` - The input text.
/// * `n_sizes` - The sizes of the n-grams.
///
/// # Returns
///
/// A vector of word n-grams, grouped by n-gram size.
#[pyfunction]
pub fn rssplit_words(text: &str, n_sizes: Vec<usize>) -> Vec<String> {
    split_words(text, &n_sizes)
}

/// Same as `rssplit_words` but for many texts at once.
#[pyfunction]
pub fn rssplit_words_many(texts: Vec<String>, n_sizes: Vec<usize>) -> Vec<Vec<String>> {
    texts
        .par_iter()
        .map(|text| split_words(text, &n_sizes))
        .collect()
}

/// Computes character n-grams.
///
/// # Arguments
///
/// * `text` - The input text.
/// * `n_sizes` - The sizes of the n-grams.
///
/// # Returns
///
/// A vector of character n-grams, grouped by n-gram size.
#[pyfunction]
pub fn rschar_ngrams(text: &str, n_sizes: Vec<usize>) -> Vec<String> {
    char_ngrams(text, &n_sizes)
}

/// Same as `rschar_ngrams` but for many texts at once.
#[pyfunction]
pub fn rschar_ngrams_many(texts: Vec<String>, n_sizes: Vec<usize>) -> Vec<Vec<String>> {
    texts
        .par_iter()
        .map(|text| char_ngrams(text, &n_sizes))
        .collect()
}

/// Character n-grams with word-boundary handling.
#[pyfunction]
pub fn rschar_wb_ngrams(text: &str, n_sizes: Vec<usize>) -> Vec<String> {
    char_ngrams(text, &n_sizes)
}

/// Same as `rschar_wb_ngrams` but for many texts at once.
#[pyfunction]
pub fn rschar_wb_ngrams_many(texts: Vec<String>, n_sizes: Vec<usize>) -> Vec<Vec<String>> {
    rschar_ngrams_many(texts, n_sizes)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rssplit_words() {
        assert_eq!(
            rssplit_words("hello world again", vec![1, 2]),
            vec!["hello", "world", "again", "hello world", "world again"]
        );
    }

    #[test]
    fn test_rschar_ngrams() {
        assert_eq!(rschar_ngrams("abcd", vec![3]), vec!["abc", "bcd"]);
        assert_eq!(rschar_ngrams("ab", vec![3]), Vec::<String>::new());
    }

    #[test]
    fn test_rschar_ngrams_unicode() {
        assert_eq!(
            rschar_ngrams("héllo", vec![2]),
            vec!["hé", "él", "ll", "lo"]
        );
    }
}
