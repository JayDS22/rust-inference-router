//! Prefix radix tree over token sequences. Keys are &[u32] (vocab IDs).
//!
//! Design notes:
//!   - Nodes are keyed by first token for cheap descent; edges carry the
//!     remaining shared prefix. Classic radix (compressed trie), not a plain
//!     trie, so storage is O(unique prefixes) not O(total tokens).
//!   - Each node optionally owns a WorkerId. On longest_prefix_match we
//!     return the deepest WorkerId we passed through, which is the right
//!     answer for prefix-cache routing: the worker that already prefixed
//!     the farthest in the request wins.
//!   - LRU eviction tracks node-id -> tick; we evict the k oldest leaves.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use crate::router::WorkerId;

type NodeId = u32;
const ROOT: NodeId = 0;

#[derive(Debug)]
struct Node {
    edge_tokens: Vec<u32>,
    children: HashMap<u32, NodeId>,
    worker: Option<WorkerId>,
    last_access: u64,
}

impl Node {
    fn new(edge_tokens: Vec<u32>) -> Self {
        Self { edge_tokens, children: HashMap::new(), worker: None, last_access: 0 }
    }
}

pub struct RadixTree {
    inner: RwLock<Inner>,
    tick: AtomicU64,
    capacity: usize,
}

struct Inner {
    nodes: Vec<Node>,
}

impl RadixTree {
    pub fn new(capacity: usize) -> Self {
        let mut nodes = Vec::with_capacity(1024);
        nodes.push(Node::new(Vec::new()));
        Self {
            inner: RwLock::new(Inner { nodes }),
            tick: AtomicU64::new(0),
            capacity,
        }
    }

    pub fn len(&self) -> usize {
        self.inner.read().nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() <= 1
    }

    pub fn insert(&self, tokens: &[u32], worker: WorkerId) {
        if tokens.is_empty() {
            return;
        }
        let t = self.tick.fetch_add(1, Ordering::Relaxed);
        let mut inner = self.inner.write();
        let mut cur = ROOT;
        let mut i = 0;
        while i < tokens.len() {
            let first = tokens[i];
            let child_id = inner.nodes[cur as usize].children.get(&first).copied();
            match child_id {
                None => {
                    let new_id = inner.nodes.len() as NodeId;
                    inner.nodes.push(Node::new(tokens[i..].to_vec()));
                    inner.nodes[new_id as usize].worker = Some(worker.clone());
                    inner.nodes[new_id as usize].last_access = t;
                    inner.nodes[cur as usize].children.insert(first, new_id);
                    if inner.nodes.len() > self.capacity {
                        drop(inner);
                        self.evict_lru(self.capacity / 10);
                    }
                    return;
                }
                Some(child) => {
                    let shared = {
                        let edge = &inner.nodes[child as usize].edge_tokens;
                        common_prefix_len(edge, &tokens[i..])
                    };
                    let edge_len = inner.nodes[child as usize].edge_tokens.len();
                    if shared == edge_len {
                        inner.nodes[child as usize].last_access = t;
                        cur = child;
                        i += shared;
                        if i == tokens.len() {
                            inner.nodes[child as usize].worker = Some(worker);
                            return;
                        }
                    } else {
                        let split_id = inner.nodes.len() as NodeId;
                        let child_old_edge = inner.nodes[child as usize].edge_tokens.clone();
                        let (common_part, suffix_old) = child_old_edge.split_at(shared);
                        let mut split_node = Node::new(common_part.to_vec());
                        split_node.last_access = t;
                        inner.nodes.push(split_node);

                        inner.nodes[child as usize].edge_tokens = suffix_old.to_vec();
                        let child_new_first = suffix_old[0];
                        inner.nodes[split_id as usize].children.insert(child_new_first, child);

                        inner.nodes[cur as usize].children.insert(first, split_id);

                        let remaining = &tokens[i + shared..];
                        if remaining.is_empty() {
                            inner.nodes[split_id as usize].worker = Some(worker);
                        } else {
                            let leaf_id = inner.nodes.len() as NodeId;
                            let mut leaf = Node::new(remaining.to_vec());
                            leaf.worker = Some(worker);
                            leaf.last_access = t;
                            inner.nodes.push(leaf);
                            inner.nodes[split_id as usize].children.insert(remaining[0], leaf_id);
                        }
                        return;
                    }
                }
            }
        }
    }

    pub fn longest_prefix_match(&self, tokens: &[u32]) -> Option<WorkerId> {
        if tokens.is_empty() {
            return None;
        }
        let inner = self.inner.read();
        let mut cur = ROOT;
        let mut i = 0;
        let mut best: Option<WorkerId> = None;
        while i < tokens.len() {
            let first = tokens[i];
            let Some(&child) = inner.nodes[cur as usize].children.get(&first) else {
                break;
            };
            let edge = &inner.nodes[child as usize].edge_tokens;
            let shared = common_prefix_len(edge, &tokens[i..]);
            if shared < edge.len() {
                if let Some(w) = &inner.nodes[child as usize].worker {
                    best = Some(w.clone());
                }
                break;
            }
            if let Some(w) = &inner.nodes[child as usize].worker {
                best = Some(w.clone());
            }
            cur = child;
            i += shared;
        }
        best
    }

    pub fn evict_lru(&self, k: usize) -> usize {
        if k == 0 {
            return 0;
        }
        let mut inner = self.inner.write();
        let mut leaves: Vec<(u64, NodeId)> = inner
            .nodes
            .iter()
            .enumerate()
            .filter(|(id, n)| *id != ROOT as usize && n.children.is_empty())
            .map(|(id, n)| (n.last_access, id as NodeId))
            .collect();
        leaves.sort_by_key(|(t, _)| *t);
        let victims: Vec<NodeId> = leaves.into_iter().take(k).map(|(_, id)| id).collect();
        if victims.is_empty() {
            return 0;
        }
        let victim_set: std::collections::HashSet<NodeId> = victims.iter().copied().collect();
        for n in inner.nodes.iter_mut() {
            n.children.retain(|_, child| !victim_set.contains(child));
        }
        for id in &victims {
            inner.nodes[*id as usize].worker = None;
            inner.nodes[*id as usize].edge_tokens.clear();
        }
        victims.len()
    }
}

fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_then_match() {
        let t = RadixTree::new(1000);
        t.insert(&[1, 2, 3, 4], "w1".into());
        assert_eq!(t.longest_prefix_match(&[1, 2, 3, 4, 9]), Some("w1".into()));
    }

    #[test]
    fn split_edge() {
        let t = RadixTree::new(1000);
        t.insert(&[1, 2, 3, 4], "w1".into());
        t.insert(&[1, 2, 5, 6], "w2".into());
        assert_eq!(t.longest_prefix_match(&[1, 2, 3, 4]), Some("w1".into()));
        assert_eq!(t.longest_prefix_match(&[1, 2, 5, 6]), Some("w2".into()));
    }

    #[test]
    fn deeper_wins() {
        let t = RadixTree::new(1000);
        t.insert(&[1, 2], "w1".into());
        t.insert(&[1, 2, 3, 4], "w2".into());
        assert_eq!(t.longest_prefix_match(&[1, 2, 3, 4, 5]), Some("w2".into()));
    }

    #[test]
    fn evict_lru_runs() {
        let t = RadixTree::new(1000);
        for i in 0..50u32 {
            t.insert(&[i, i + 1, i + 2], format!("w{i}"));
        }
        let n = t.evict_lru(10);
        assert!(n > 0);
    }
}
