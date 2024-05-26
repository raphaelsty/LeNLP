use pyo3::prelude::*;

use std::collections::HashMap;
use std::collections::HashSet;

use rayon::prelude::*;
use unidecode::unidecode;

#[pyclass()]
pub struct RSKeywordProcessor {
    keyword: String,
    non_word_boundaries: HashSet<char>,
    keyword_trie_dict: HashMap<char, RSTrieNode>,
    lowercase: bool,
    normalize: bool,
    terms_in_trie: usize,
}

#[pyclass()]
pub struct RSTrieNode {
    children: HashMap<char, RSTrieNode>,
    is_end: bool,
    clean_name: Option<String>,
}

impl RSTrieNode {
    pub fn new() -> Self {
        RSTrieNode {
            children: HashMap::new(),
            is_end: false,
            clean_name: None,
        }
    }
}

#[pymethods]
impl RSKeywordProcessor {
    #[new]
    pub fn new(lowercase: bool, normalize: bool) -> Self {
        let keyword: String = "_keyword_".to_string();
        let non_word_boundaries: HashSet<char> = {
            let mut set: HashSet<char> = HashSet::new();
            set.extend('0'..='9');
            set.extend('a'..='z');
            set.extend('A'..='Z');
            set.insert('_');
            set
        };

        RSKeywordProcessor {
            keyword,
            non_word_boundaries,
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
            .map(|keyword: &String| self.add_keyword(&keyword, clean_name))
            .collect()
    }

    pub fn add_keyword(&mut self, keyword: &str, clean_name: Option<&str>) -> bool {
        let clean_name: &str = clean_name.unwrap_or(keyword);
        let keyword: String = if self.normalize {
            unidecode(keyword)
                .to_lowercase()
                .chars()
                .filter(|c| !c.is_ascii_punctuation())
                .collect::<String>()
                .trim()
                .to_string()
        } else if self.lowercase {
            keyword.to_lowercase()
        } else {
            keyword.to_string()
        };

        let mut current_node: &mut HashMap<char, RSTrieNode> = &mut self.keyword_trie_dict;
        for char in keyword.chars() {
            current_node = &mut current_node
                .entry(char)
                .or_insert_with(RSTrieNode::new)
                .children;
        }

        if !current_node.contains_key(&self.keyword.chars().next().unwrap()) {
            self.terms_in_trie += 1;
            current_node.insert(
                self.keyword.chars().next().unwrap(),
                RSTrieNode {
                    children: HashMap::new(),
                    is_end: true,
                    clean_name: Some(clean_name.to_string()),
                },
            );
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
            .map(|sentence: &String| self.extract_keywords(&sentence))
            .collect()
    }

    pub fn extract_keywords(&self, sentence: &str) -> Vec<(String, usize, usize)> {
        // Map from the index in the normalized sentence to the index in the original sentence
        let mut index_map: Vec<usize> = Vec::with_capacity(sentence.len());
        let mut original_idx = 0;

        let normalized_sentence: String = if self.normalize {
            let mut normalized = String::new();
            for c in sentence.chars() {
                if c.is_ascii_punctuation() {
                    original_idx += c.len_utf8();
                    continue;
                }
                let normalized_char = unidecode::unidecode_char(c).to_lowercase();
                for nc in normalized_char.chars() {
                    normalized.push(nc);
                    index_map.push(original_idx);
                }
                original_idx += c.len_utf8();
            }
            normalized.to_string()
        } else if self.lowercase {
            sentence.to_lowercase()
        } else {
            sentence.to_string()
        };

        let mut extracted_keywords: Vec<(String, usize, usize)> = Vec::new();
        let mut current_node: &HashMap<char, RSTrieNode> = &self.keyword_trie_dict;
        let mut start_pos: usize = 0;
        let mut end_pos: usize = 0;

        let mut idx: usize = 0;
        let sentence_len: usize = normalized_sentence.len();
        while idx < sentence_len {
            let char: char = normalized_sentence.chars().nth(idx).unwrap();
            if !self.non_word_boundaries.contains(&char) {
                if let Some(node) = current_node.get(&self.keyword.chars().next().unwrap()) {
                    if node.is_end {
                        let clean_name: &String = node.clean_name.as_ref().unwrap();
                        let original_start_pos = index_map[start_pos];
                        let original_end_pos = index_map[end_pos - 1] + 1;
                        extracted_keywords.push((
                            clean_name.clone(),
                            original_start_pos,
                            original_end_pos,
                        ));
                    }
                }
                current_node = &self.keyword_trie_dict;
                start_pos = idx + 1;
            } else if let Some(node) = current_node.get(&char) {
                current_node = &node.children;
                end_pos = idx + 1;
            } else {
                current_node = &self.keyword_trie_dict;
                start_pos = idx + 1;
            }
            idx += 1;
        }

        // Check if the last segment is a keyword
        if let Some(node) = current_node.get(&self.keyword.chars().next().unwrap()) {
            if node.is_end {
                let clean_name: &String = node.clean_name.as_ref().unwrap();
                let original_start_pos = index_map[start_pos];
                let original_end_pos = index_map[end_pos - 1] + 1;
                extracted_keywords.push((clean_name.clone(), original_start_pos, original_end_pos));
            }
        }

        extracted_keywords
    }
}

pub fn register_functions(m: &PyModule) -> PyResult<()> {
    m.add_class::<RSKeywordProcessor>()?;
    m.add_class::<RSTrieNode>()?;
    Ok(())
}
