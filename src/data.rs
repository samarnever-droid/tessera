//! Data: byte-level corpus for language modeling, plus the Needle-in-a-Haystack task.
//!
//! NIAH format (char level, single planted key→value pair per the spec):
//!   [filler] " K7Q2 M3X9 . " [filler] " K7Q2 ? M3X9"
//!                                              ^^^^^ loss only on these 4 bytes
//! The key and its value are ADJACENT so the k=4 causal depthwise conv (Pillar 2) can
//! bind them into the same slot patterns — the mechanism the memory substrate is designed
//! around. The final query repeats the key; the model must emit the paired value.

use crate::rng::Rng;

pub struct NiahCfg {
    pub key_len: usize,
    pub val_len: usize,
    pub n_distract: usize,
    pub seed: u64,
    /// loss weight on the answer positions (retrieval gradient SNR vs the
    /// unpredictable needle-span noise targets added by --needle-loss)
    pub ans_w: f64,
}

impl NiahCfg {
    /// Smallest sequence length this task fits in WITHOUT truncating the
    /// answer. The builder spends noise = seq - tail bytes as head + needle +
    /// tail-fill; the query+answer (tail bytes) must survive after
    /// floor(noise/2) head bytes and the full needle. Solving
    /// floor((S-tail)/2) + needle_len + tail <= S gives S >= 2*needle + tail - 1.
    /// (The old formula undercounted the needle's " . " by one byte and the
    /// head split entirely, so tiny seqs silently cut the answer off.)
    pub fn min_seq(&self) -> usize {
        let tail = 1 + self.key_len + 3 + self.val_len; // " KEY ? VAL"
        let needle = 1 + self.key_len + 1 + self.val_len + 3; // " KEY VAL . "
        2 * needle + tail - 1
    }
}

impl Default for NiahCfg {
    fn default() -> Self {
        Self { key_len: 4, val_len: 4, n_distract: 0, seed: 1234, ans_w: 1.0 }
    }
}

const KEY_ALPHABET: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
const POOL: &[&[u8]] = &[
    b"the sky is clear today",
    b"rivers run low in summer",
    b"old towers watch the road",
    b"small birds cross the field",
    b"quiet streets hold warm light",
    b"three ships sail north",
    b"green hills fade to blue",
    b"cold winds follow rain",
];

pub struct Niah {
    pub cfg: NiahCfg,
    pub rng: Rng,
}

impl Niah {
    pub fn new(cfg: NiahCfg) -> Self {
        Self { rng: Rng::new(cfg.seed), cfg }
    }

    fn rand_code(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| KEY_ALPHABET[self.rng.below(KEY_ALPHABET.len())]).collect()
    }

    fn filler(&mut self, budget: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(budget);
        while out.len() < budget {
            let s = POOL[self.rng.below(POOL.len())];
            if !out.is_empty() {
                out.push(b' ');
            }
            out.extend_from_slice(s);
        }
        out.truncate(budget);
        out
    }

    fn needle(&mut self, with_query: bool) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let key = self.rand_code(self.cfg.key_len);
        let val = self.rand_code(self.cfg.val_len);
        let mut nd = Vec::new();
        nd.push(b' ');
        nd.extend_from_slice(&key);
        nd.push(b' ');
        nd.extend_from_slice(&val);
        if with_query {
            nd.extend_from_slice(b" . ");
        } else {
            nd.extend_from_slice(b" . ");
        }
        (nd, key, val)
    }

    /// Build one sequence of exactly `seq_len` tokens with loss masked to the answer.
    pub fn sample(&mut self, seq_len: usize) -> NiahSample {
        let (val_len, key_len) = (self.cfg.val_len, self.cfg.key_len);
        // final: [filler] needle(true key) [filler] [distractors...] " KEY ? VALUE"
        let tail_len = 1 + key_len + 3 + val_len; // " KEY ? VAL" (space+key+" ? "+val)
        let mut noise_budget = seq_len.saturating_sub(tail_len); // degenerate ok, never panics
        // (pre-fix tail_len undercounted " ? " by 1, silently truncating the last answer char)
        let mut seq: Vec<u8> = Vec::with_capacity(seq_len);
        let mut keys: Vec<Vec<u8>> = Vec::new();
        let mut vals: Vec<Vec<u8>> = Vec::new();

        let mut spans: Vec<(usize, usize)> = Vec::new(); // [start,end) of each " KEY VAL ."
        let (nd, key, val) = self.needle(false);
        let head = self.filler(noise_budget / 2);
        seq.extend_from_slice(&head);
        noise_budget -= head.len();
        spans.push((seq.len(), seq.len() + nd.len()));
        seq.extend_from_slice(&nd);
        noise_budget = noise_budget.saturating_sub(nd.len()); // degenerate short-seq safe
        keys.push(key);
        vals.push(val);
        for _ in 0..self.cfg.n_distract {
            let (nd, dk, dv) = self.needle(false);
            if nd.len() + 1 > noise_budget {
                break;
            }
            spans.push((seq.len(), seq.len() + nd.len()));
            seq.extend_from_slice(&nd);
            noise_budget -= nd.len();
            keys.push(dk);
            vals.push(dv);
        }
        let tail_fill = self.filler(noise_budget);
        seq.extend_from_slice(&tail_fill);
        // query + answer
        seq.push(b' ');
        seq.extend_from_slice(&keys[0]);
        seq.extend_from_slice(b" ? ");
        seq.extend_from_slice(&vals[0]);
        while seq.len() < seq_len {
            seq.push(b' '); // degenerate short-seq path: pad to requested length
        }
        seq.truncate(seq_len);
        debug_assert_eq!(seq.len(), seq_len);

        // targets (shared with the transplant CLI via targets_from_tokens)
        let n = seq.len();
        let (tgt1, tgt2, mask, i) = targets_from_tokens(&seq, val_len, self.cfg.ans_w);
        let answer: Vec<u8> = seq[i..(i + val_len).min(n)].to_vec();
        NiahSample { tokens: seq, tgt1, tgt2, mask, answer, answer_start: i, needle_spans: spans }
    }

    /// Build an evaluation prompt (context truncated just before the answer).
    pub fn eval_prompt(&mut self, seq_len: usize) -> (Vec<u8>, Vec<u8>) {
        let s = self.sample(seq_len);
        let mut prompt = s.tokens.clone();
        prompt.truncate(s.answer_start);
        (prompt, s.answer)
    }
}

fn find_sub(h: &[u8], nd: &[u8]) -> Option<usize> {
    if nd.len() > h.len() {
        return None;
    }
    (0..=h.len() - nd.len()).rev().find(|&i| &h[i..i + nd.len()] == nd)
}

/// The retrieval alphabet, public for the transplant CLI and tests.
pub fn key_alphabet() -> &'static [u8] {
    KEY_ALPHABET
}

/// Rebuild (tgt1, tgt2, mask, answer_start) from prompt bytes — the exact
/// target block of `Niah::sample`, factored out so the transplant trainer
/// consumes stored pools with byte-identical supervision.
pub fn targets_from_tokens(
    seq: &[u8],
    val_len: usize,
    ans_w: f64,
) -> (Vec<i32>, Vec<i32>, Vec<f64>, usize) {
    let n = seq.len();
    let mut tgt1 = vec![-1i32; n];
    let mut tgt2 = vec![-1i32; n];
    let mut mask = vec![0.0f64; n];
    for t in 0..n {
        if t + 1 < n {
            tgt1[t] = seq[t + 1] as i32;
        }
        if t + 2 < n {
            tgt2[t] = seq[t + 2] as i32;
        }
    }
    // mask only the value chars after " ? "
    let mut i = n;
    if let Some(q) = find_sub(seq, &[b' ', b'?', b' ']) {
        i = q + 3;
    }
    for j in 0..val_len {
        let t = i as isize + j as isize - 1; // position whose TARGET is seq[i+j]
        if t >= 0 && (t as usize) < n {
            mask[t as usize] = ans_w;
        }
    }
    (tgt1, tgt2, mask, i)
}

pub struct NiahSample {
    /// [start, end) byte ranges of each inserted " KEY VAL ." needle
    pub needle_spans: Vec<(usize, usize)>,
    pub tokens: Vec<u8>,
    pub tgt1: Vec<i32>,
    pub tgt2: Vec<i32>,
    pub mask: Vec<f64>,
    pub answer: Vec<u8>,
    pub answer_start: usize,
}

/// Byte-level text corpus with sequential sampling.
pub struct Corpus {
    pub data: Vec<u8>,
    pub pos: usize,
}

impl Corpus {
    pub fn from_file(path: &str, max_bytes: usize) -> std::io::Result<Self> {
        let all = std::fs::read(path)?;
        if all.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("corpus file '{path}' is empty"),
            ));
        }
        let data = if all.len() > max_bytes { all[..max_bytes].to_vec() } else { all };
        Ok(Self { data, pos: 0 })
    }
    /// Next `n` bytes, wrapping around.
    pub fn next_chunk(&mut self, n: usize) -> Vec<usize> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            if self.pos >= self.data.len() {
                self.pos = 0;
            }
            out.push(self.data[self.pos] as usize);
            self.pos += 1;
        }
        out
    }
}

/// LM targets: main = t+1, aux = t+2, all positions masked in.
pub fn lm_targets(toks: &[usize]) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let n = toks.len();
    let mut tgt1 = vec![-1i32; n];
    let mut tgt2 = vec![-1i32; n];
    let mask = vec![1.0f64; n];
    for t in 0..n {
        if t + 1 < n {
            tgt1[t] = toks[t + 1] as i32;
        }
        if t + 2 < n {
            tgt2[t] = toks[t + 2] as i32;
        }
    }
    (tgt1, tgt2, mask)
}

/// Greedy exact-match NIAH evaluation: feed the prompt, then require every answer
/// byte to be the top-logit prediction. Returns the number of fully-correct samples.
pub fn niah_eval_model(model: &mut crate::model::Model<f32>, gen: &mut Niah, seq: usize, n: usize) -> usize {
    let mut correct = 0;
    for _ in 0..n {
        let (prompt, answer) = gen.eval_prompt(seq);
        if answer.is_empty() {
            continue; // degenerate sample: cannot recall anything
        }
        let toks: Vec<usize> = prompt.iter().map(|&b| b as usize).collect();
        model.reset_state();
        // feed the whole prompt; logits after the last prompt byte predict answer[0]
        let mut z: Vec<f32> = Vec::new();
        for &t in &toks {
            let (zz, _) = model.fwd_step(t, false);
            z = zz;
        }
        let mut ok = true;
        let mut prev;
        for (i, &expect) in answer.iter().enumerate() {
            let arg = z
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(j, _)| j)
                .unwrap_or(0);
            if arg != expect as usize {
                ok = false;
                break;
            }
            prev = arg;
            if i + 1 < answer.len() {
                let (zz, _) = model.fwd_step(prev, false);
                z = zz;
            }
        }
        if ok {
            correct += 1;
        }
    }
    correct
}



#[cfg(test)]
mod tests {
    // U01..U10 — data-generation robustness (see tests/cli.rs for C01..C40)
    use super::*;

    fn gen(kl: usize, vl: usize, seed: u64) -> Niah {
        Niah::new(NiahCfg { key_len: kl, val_len: vl, seed, ..NiahCfg::default() })
    }

    #[test]
    fn u01_sample_exact_length_across_configs() {
        for (kl, vl, seq) in [(1usize, 1usize, 64usize), (2, 3, 128), (4, 4, 256), (1, 4, 100)] {
            let s = gen(kl, vl, 1).sample(seq);
            assert_eq!(s.tokens.len(), seq, "kl={kl} vl={vl} seq={seq}");
        }
    }

    #[test]
    fn u02_needle_spans_exact_match() {
        let (kl, vl) = (2usize, 3usize);
        let s = gen(kl, vl, 5).sample(128);
        assert!(!s.needle_spans.is_empty());
        let (a, b) = s.needle_spans[0];
        let span = &s.tokens[a..b];
        // " KEY VAL . " (needle() appends the trailing space too)
        assert_eq!(span.len(), 1 + kl + 1 + vl + 3);
        assert_eq!(span[0], b' ');
        assert_eq!(span[1 + kl], b' ');
        assert_eq!(span[1 + kl + 1 + vl], b' ');
        assert_eq!(span[span.len() - 2], b'.');
        assert_eq!(span[span.len() - 1], b' ');
        let key = &span[1..1 + kl];
        let val = &span[1 + kl + 1..1 + kl + 1 + vl];
        assert!(key
            .iter()
            .chain(val.iter())
            .all(|&c| b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789".contains(&c)));
    }

    #[test]
    fn u03_mask_positions_default_weight() {
        let (kl, vl, seq) = (2usize, 3usize, 128usize);
        let s = gen(kl, vl, 7).sample(seq);
        let hot: Vec<usize> = (0..seq).filter(|&t| s.mask[t] != 0.0).collect();
        assert_eq!(hot.len(), vl, "exactly val_len supervised positions");
        for (j, &t) in hot.iter().enumerate() {
            assert_eq!(s.mask[t], 1.0);
            // position t has TARGET seq[t+1]; the supervised targets are the answer chars
            assert_eq!(s.tgt1[t], s.answer[j] as i32);
        }
    }

    #[test]
    fn u04_ans_w_weight_propagates_to_mask() {
        let mut g = Niah::new(NiahCfg { key_len: 1, val_len: 2, ans_w: 3.5, seed: 3, ..NiahCfg::default() });
        let s = g.sample(96);
        let hot: Vec<f64> = s.mask.iter().copied().filter(|&m| m != 0.0).collect();
        assert_eq!(hot.len(), 2);
        assert!(hot.iter().all(|&m| (m - 3.5).abs() < 1e-12), "hot={hot:?}");
    }

    #[test]
    fn u05_eval_prompt_truncation() {
        let s = gen(2, 2, 11).sample(128);
        let (prompt, answer) = gen(2, 2, 11).eval_prompt(128);
        assert_eq!(prompt.len(), s.answer_start);
        assert_eq!(answer, s.answer);
        assert_eq!(&s.tokens[s.answer_start..s.answer_start + answer.len()], &answer[..]);
    }

    #[test]
    fn u06_query_tail_format() {
        let (kl, vl) = (3usize, 2usize);
        let s = gen(kl, vl, 13).sample(128);
        let a = s.answer_start;
        // " KEY ? " occupies the kl+4 bytes before the answer
        let head = &s.tokens[a - (kl + 4)..a];
        assert_eq!(head[0], b' ');
        assert_eq!(head[kl + 1], b' ');
        assert_eq!(head[kl + 2], b'?');
        // the queried key equals the first needle's key
        let (na, _) = s.needle_spans[0];
        assert_eq!(&head[1..1 + kl], &s.tokens[na + 1..na + 1 + kl]);
    }

    #[test]
    fn u07_query_key_matches_first_needle_with_distractors() {
        let mut g = Niah::new(NiahCfg { key_len: 2, val_len: 2, n_distract: 1, seed: 17, ..NiahCfg::default() });
        let s = g.sample(200);
        let (na, nb) = s.needle_spans[0];
        let key0 = s.tokens[na + 1..na + 1 + 2].to_vec();
        let a = s.answer_start;
        // layout before the answer: ' ', k0, k1, ' ', '?', ' ' -> key at a-5..a-3
        let qkey = s.tokens[a - 5..a - 3].to_vec();
        assert_eq!(qkey, key0, "query must reference the FIRST needle");
        assert!(nb > na);
    }

    #[test]
    fn u08_min_seq_boundary_is_safe() {
        for (kl, vl) in [(1usize, 1usize), (2, 4), (4, 4), (8, 8)] {
            let cfg = NiahCfg { key_len: kl, val_len: vl, ..Default::default() };
            let min = cfg.min_seq();
            let s = gen(kl, vl, 23).sample(min);
            assert_eq!(s.tokens.len(), min, "min_seq must be constructible");
            // one below min must not panic either (saturating degenerate path)
            let s2 = gen(kl, vl, 23).sample(min - 1);
            assert_eq!(s2.tokens.len(), min - 1);
        }
    }

    #[test]
    fn u09_same_seed_same_stream() {
        let a = gen(2, 2, 99).sample(128);
        let b = gen(2, 2, 99).sample(128);
        assert_eq!(a.tokens, b.tokens);
        assert_eq!(a.answer, b.answer);
    }

    #[test]
    fn u10_diff_seed_diff_stream() {
        let a = gen(2, 2, 1).sample(128);
        let b = gen(2, 2, 2).sample(128);
        assert_ne!(a.tokens, b.tokens);
    }
}
