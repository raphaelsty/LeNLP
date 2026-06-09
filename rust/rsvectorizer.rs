use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::borrow::Cow;

use crate::rsanalyzer::char_boundaries;
use crate::rsnormalizer::rsnormalize;

/// Counts word n-grams without materializing the n-gram list.
///
/// Unigrams are counted as borrowed slices; larger n-grams reuse a single
/// scratch buffer so a `String` is only allocated per unique n-gram.
fn count_word_ngrams(text: &str, n_sizes: &[usize]) -> FxHashMap<String, usize> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut counts: FxHashMap<String, usize> = FxHashMap::default();
    let mut buffer = String::new();

    for &n in n_sizes {
        if n == 0 || n > words.len() {
            continue;
        }
        for window in words.windows(n) {
            buffer.clear();
            for (position, word) in window.iter().enumerate() {
                if position > 0 {
                    buffer.push(' ');
                }
                buffer.push_str(word);
            }
            if let Some(count) = counts.get_mut(buffer.as_str()) {
                *count += 1;
            } else {
                counts.insert(buffer.clone(), 1);
            }
        }
    }
    counts
}

/// Counts character n-grams by slicing the text at char boundaries.
///
/// Counting happens on borrowed slices; owned keys are only allocated once
/// per unique n-gram when the map is finalized.
fn count_char_ngrams(text: &str, n_sizes: &[usize]) -> FxHashMap<String, usize> {
    let boundaries = char_boundaries(text);
    let num_chars = boundaries.len() - 1;
    let mut counts: FxHashMap<&str, usize> = FxHashMap::default();

    for &n in n_sizes {
        if n == 0 || n > num_chars {
            continue;
        }
        for start in 0..=(num_chars - n) {
            *counts
                .entry(&text[boundaries[start]..boundaries[start + n]])
                .or_insert(0) += 1;
        }
    }

    counts
        .into_iter()
        .map(|(ngram, count)| (ngram.to_string(), count))
        .collect()
}

fn filter_stop_words(text: &str, stop_words: &FxHashSet<String>) -> String {
    let mut filtered = String::with_capacity(text.len());
    for word in text.split_whitespace() {
        if !stop_words.contains(word) {
            if !filtered.is_empty() {
                filtered.push(' ');
            }
            filtered.push_str(word);
        }
    }
    filtered
}

/// Vectorizes a single text: normalize, drop stop words, then count n-grams.
fn vectorize(
    text: &str,
    n_sizes: &[usize],
    analyzer: &str,
    stop_words: Option<&FxHashSet<String>>,
    normalize: Option<bool>,
) -> FxHashMap<String, usize> {
    let text: Cow<str> = match normalize {
        Some(true) => Cow::Owned(rsnormalize(text)),
        _ => Cow::Borrowed(text),
    };
    let text: Cow<str> = match stop_words {
        Some(stop_words) => Cow::Owned(filter_stop_words(&text, stop_words)),
        None => text,
    };

    match analyzer {
        "word" => count_word_ngrams(&text, n_sizes),
        "char" | "char_wb" => count_char_ngrams(&text, n_sizes),
        _ => panic!("Invalid analyzer type"),
    }
}

/// Main vectorization function.
#[pyfunction]
pub fn rsvectorize_many(
    texts: Vec<String>,
    n_sizes: Vec<usize>,
    analyzer: String,
    stop_words: Option<Vec<String>>,
    normalize: Option<bool>,
) -> Vec<FxHashMap<String, usize>> {
    let stop_words: Option<FxHashSet<String>> =
        stop_words.map(|words| words.into_iter().collect());

    texts
        .par_iter()
        .map(|text| vectorize(text, &n_sizes, &analyzer, stop_words.as_ref(), normalize))
        .collect()
}

#[pyfunction]
pub fn rsvectorize_split_words_many(
    texts: Vec<String>,
    n_sizes: Vec<usize>,
    stop_words: Option<Vec<String>>,
    normalize: Option<bool>,
) -> Vec<FxHashMap<String, usize>> {
    rsvectorize_many(texts, n_sizes, "word".to_string(), stop_words, normalize)
}

#[pyfunction]
pub fn rsvectorize_char_ngrams_many(
    texts: Vec<String>,
    n_sizes: Vec<usize>,
    stop_words: Option<Vec<String>>,
    normalize: Option<bool>,
) -> Vec<FxHashMap<String, usize>> {
    rsvectorize_many(texts, n_sizes, "char".to_string(), stop_words, normalize)
}

#[pyfunction]
pub fn rsvectorize_char_wb_ngrams_many(
    texts: Vec<String>,
    n_sizes: Vec<usize>,
    stop_words: Option<Vec<String>>,
    normalize: Option<bool>,
) -> Vec<FxHashMap<String, usize>> {
    rsvectorize_many(texts, n_sizes, "char_wb".to_string(), stop_words, normalize)
}

pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rsvectorize_split_words_many, m)?)?;
    m.add_function(wrap_pyfunction!(rsvectorize_char_ngrams_many, m)?)?;
    m.add_function(wrap_pyfunction!(rsvectorize_char_wb_ngrams_many, m)?)?;
    m.add_function(wrap_pyfunction!(rsvectorize_many, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_word_ngrams() {
        let counts = count_word_ngrams("hello world hello", &[1, 2]);
        assert_eq!(counts.get("hello"), Some(&2));
        assert_eq!(counts.get("world"), Some(&1));
        assert_eq!(counts.get("hello world"), Some(&1));
        assert_eq!(counts.get("world hello"), Some(&1));
    }

    #[test]
    fn test_count_char_ngrams() {
        let counts = count_char_ngrams("abab", &[2]);
        assert_eq!(counts.get("ab"), Some(&2));
        assert_eq!(counts.get("ba"), Some(&1));
    }

    #[test]
    fn test_vectorize_normalize_and_stop_words() {
        let stop_words: FxHashSet<String> = ["world".to_string()].into_iter().collect();
        let counts = vectorize("Hello, World!", &[1], "word", Some(&stop_words), Some(true));
        assert_eq!(counts.get("hello"), Some(&1));
        assert_eq!(counts.get("world"), None);
    }
}
