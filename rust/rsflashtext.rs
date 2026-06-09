use pyo3::prelude::*;

use std::collections::HashMap;

use rayon::prelude::*;
use unidecode::unidecode_char;

use crate::rsnormalizer::rsnormalize;

/// Word characters: keyword matches must be delimited by anything else.
fn is_non_word_boundary(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

#[pyclass()]
pub struct RSKeywordProcessor {
    keyword_trie_dict: HashMap<char, RSTrieNode>,
    lowercase: bool,
    normalize: bool,
    #[pyo3(get)]
    terms_in_trie: usize,
}

/// Emit the pending match, if it ends on a complete keyword.
fn flush_match(
    current_match: &mut Option<(&RSTrieNode, usize, usize)>,
    spans: &[(usize, usize)],
    extracted: &mut Vec<(String, usize, usize)>,
) {
    if let Some((node, start, end)) = current_match.take() {
        if let Some(clean_name) = &node.clean_name {
            extracted.push((clean_name.clone(), spans[start].0, spans[end - 1].1));
        }
    }
}

#[pyclass()]
pub struct RSTrieNode {
    children: HashMap<char, RSTrieNode>,
    clean_name: Option<String>,
}

impl RSTrieNode {
    pub fn new() -> Self {
        RSTrieNode {
            children: HashMap::new(),
            clean_name: None,
        }
    }
}

#[pymethods]
impl RSKeywordProcessor {
    #[new]
    pub fn new(lowercase: bool, normalize: bool) -> Self {
        RSKeywordProcessor {
            keyword_trie_dict: HashMap::new(),
            lowercase,
            normalize,
            terms_in_trie: 0,
        }
    }

    pub fn add_keywords_many(
        &mut self,
        keywords: Vec<String>,
        clean_name: Option<&str>,
    ) -> Vec<bool> {
        keywords
            .iter()
            .map(|keyword| self.add_keyword(keyword, clean_name))
            .collect()
    }

    pub fn add_keyword(&mut self, keyword: &str, clean_name: Option<&str>) -> bool {
        let clean_name: &str = clean_name.unwrap_or(keyword);
        let keyword: String = if self.normalize {
            rsnormalize(keyword)
        } else if self.lowercase {
            keyword.to_lowercase()
        } else {
            keyword.to_string()
        };

        let mut node: &mut RSTrieNode = self
            .keyword_trie_dict
            .entry(keyword.chars().next().unwrap_or(' '))
            .or_insert_with(RSTrieNode::new);
        for c in keyword.chars().skip(1) {
            node = node.children.entry(c).or_insert_with(RSTrieNode::new);
        }

        if node.clean_name.is_none() {
            node.clean_name = Some(clean_name.to_string());
            self.terms_in_trie += 1;
            true
        } else {
            false
        }
    }

    pub fn extract_keywords_many(
        &self,
        sentences: Vec<String>,
    ) -> Vec<Vec<(String, usize, usize)>> {
        sentences
            .par_iter()
            .map(|sentence| self.extract_keywords(sentence))
            .collect()
    }

    pub fn extract_keywords(&self, sentence: &str) -> Vec<(String, usize, usize)> {
        // Chars of the processed sentence, each paired with its byte span in
        // the original sentence so matches report original positions.
        let mut chars: Vec<char> = Vec::with_capacity(sentence.len());
        let mut spans: Vec<(usize, usize)> = Vec::with_capacity(sentence.len());

        for (offset, c) in sentence.char_indices() {
            let span = (offset, offset + c.len_utf8());
            if self.normalize {
                if c.is_ascii_punctuation() {
                    continue;
                }
                for ascii in unidecode_char(c).chars() {
                    chars.push(ascii.to_ascii_lowercase());
                    spans.push(span);
                }
            } else if self.lowercase {
                for lower in c.to_lowercase() {
                    chars.push(lower);
                    spans.push(span);
                }
            } else {
                chars.push(c);
                spans.push(span);
            }
        }

        let mut extracted: Vec<(String, usize, usize)> = Vec::new();
        let mut current_match: Option<(&RSTrieNode, usize, usize)> = None;
        // A keyword may only begin right after a word boundary, so once a
        // match breaks mid-word we wait for the next boundary before retrying.
        let mut at_word_start = true;

        for (idx, &c) in chars.iter().enumerate() {
            if !is_non_word_boundary(c) {
                // Word boundary: emit the pending match, restart after it.
                flush_match(&mut current_match, &spans, &mut extracted);
                at_word_start = true;
            } else {
                current_match = match current_match {
                    Some((node, start, _)) => {
                        node.children.get(&c).map(|child| (child, start, idx + 1))
                    }
                    None if at_word_start => self
                        .keyword_trie_dict
                        .get(&c)
                        .map(|child| (child, idx, idx + 1)),
                    None => None,
                };
                at_word_start = false;
            }
        }
        flush_match(&mut current_match, &spans, &mut extracted);

        extracted
    }
}

/// Registers all the above functions in a Python sub-module.
///
/// Called from your `#[pymodule]` entry-point.
pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RSKeywordProcessor>()?;
    m.add_class::<RSTrieNode>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_keywords() {
        let mut processor = RSKeywordProcessor::new(true, true);
        assert!(processor.add_keyword("hello", None));
        assert!(processor.add_keyword("world", None));
        assert!(!processor.add_keyword("hello", None));

        assert_eq!(
            processor.extract_keywords("Hello, world!"),
            vec![
                ("hello".to_string(), 0, 5),
                ("world".to_string(), 7, 12),
            ]
        );
    }

    #[test]
    fn test_extract_keywords_no_normalize() {
        let mut processor = RSKeywordProcessor::new(true, false);
        processor.add_keyword("hello", None);
        assert_eq!(
            processor.extract_keywords("Hello world"),
            vec![("hello".to_string(), 0, 5)]
        );
    }

    #[test]
    fn test_no_partial_word_match() {
        let mut processor = RSKeywordProcessor::new(true, true);
        processor.add_keyword("hell", None);
        assert!(processor.extract_keywords("hello").is_empty());
    }

    #[test]
    fn test_only_matches_at_word_boundaries() {
        let mut processor = RSKeywordProcessor::new(true, true);
        processor.add_keyword("s", None);
        processor.add_keyword("gives", None);
        // "s" only matches as a standalone word, not the trailing s of PBMCs.
        assert_eq!(
            processor.extract_keywords("human PBMCs together"),
            Vec::<(String, usize, usize)>::new()
        );
        assert_eq!(
            processor.extract_keywords("which gives rise"),
            vec![("gives".to_string(), 6, 11)]
        );
    }

    #[test]
    fn test_match_span_covers_multibyte_chars() {
        let mut processor = RSKeywordProcessor::new(true, true);
        processor.add_keyword("tgfb", None);
        // "TGF-β": the final β is two bytes, so the span must end at byte 6.
        assert_eq!(
            processor.extract_keywords("TGF-β"),
            vec![("tgfb".to_string(), 0, 6)]
        );
    }
}
