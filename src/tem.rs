//! `.tem` — TESSERA Embedded Model container (v1).
//!
//! Design goals, in order: smallest, fastest-loading, lovable.
//!
//! - **8-bit grouped symmetric quantization**: weights are stored as `i8`
//!   with one f32 scale per 128-element group (≈ 4x smaller than the f32
//!   `TSS1` dump). No entropy coder — decoding is a widening multiply that
//!   auto-vectorizes, so loading is a single linear pass at memory speed.
//! - **One allocation, one read**: the whole model is ≤ a few MB, so the
//!   loader does a single `read_to_end`, verifies one FNV-1a64 checksum over
//!   the payload, and dequantizes straight into the parameter vectors.
//! - **Self-describing**: the header carries the full `ModelCfg` (including
//!   the float fields `TSS1` silently dropped) and a named tensor directory,
//!   so `tem::load` builds a ready-to-run model with no external config.
//!
//! Layout (all integers little-endian):
//!
//! ```text
//! magic        8B   "TEMST8V1"
//! hdr_len      u32  byte length of the header block that follows
//! header:
//!   version    u32  = 1
//!   cfg ints   u32 x10   d, layers, heads, m, ff, lora_r, conv_k, fine, coarse, vocab
//!   cfg floats f64 x4    mtp_alpha, zloss_w, cap, rope_base
//!   n_tensors  u32
//!   per tensor u8 name_len | name | u64 elems | u32 group (quant group size)
//!   payload_len u64
//!   fnv64      u64  FNV-1a64 over the payload region
//! payload:
//!   per tensor elems bytes of i8 quanta, then n_groups f32 scales
//! ```

use crate::model::{Model, ModelCfg};
use std::io::Write;

pub const MAGIC: &[u8; 8] = b"TEMST8V1";
pub const VERSION: u32 = 1;
/// quantization group size (equals the model width d=128; tails get their
/// own smaller group and scale)
pub const GROUP: usize = 128;

fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn push_u32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}
fn push_u64(v: &mut Vec<u8>, x: u64) {
    v.extend_from_slice(&x.to_le_bytes());
}
fn push_f64(v: &mut Vec<u8>, x: f64) {
    v.extend_from_slice(&x.to_le_bytes());
}

struct TensorDir {
    name: &'static str,
    elems: usize,
}

/// Serialize the model into the `.tem` byte format.
pub fn encode<T: crate::num::Num>(model: &mut Model<T>) -> Vec<u8> {
    let params = model.params();
    let c = &model.cfg;
    let dir: Vec<TensorDir> = params.iter().map(|p| TensorDir { name: p.name, elems: unsafe { (*p.w).len() } }).collect();

    let mut hdr = Vec::with_capacity(64 + dir.len() * 24);
    push_u32(&mut hdr, VERSION);
    for x in [c.d, c.layers, c.heads, c.m, c.ff, c.lora_r, c.conv_k, c.fine, c.coarse, c.vocab] {
        push_u32(&mut hdr, x as u32);
    }
    for x in [c.mtp_alpha, c.zloss_w, c.cap, c.rope_base] {
        push_f64(&mut hdr, x);
    }
    push_u32(&mut hdr, dir.len() as u32);
    for t in &dir {
        hdr.push(t.name.len() as u8);
        hdr.extend_from_slice(t.name.as_bytes());
        push_u64(&mut hdr, t.elems as u64);
        push_u32(&mut hdr, GROUP as u32);
    }

    // quantize + assemble payload (bytes; i8 quanta stored as raw u8)
    let mut payload: Vec<u8> = Vec::with_capacity(params.iter().map(|p| unsafe { (*p.w).len() }).sum::<usize>() + 1024);
    unsafe {
        for p in &params {
            let w = &*p.w;
            let n = w.len();
            let mut q = vec![0i8; n];
            let groups = (n + GROUP - 1) / GROUP;
            let mut scales = vec![0f32; groups];
            for g in 0..groups {
                let (s, e) = (g * GROUP, ((g + 1) * GROUP).min(n));
                let mut max = 0.0f64;
                for x in &w[s..e] {
                    max = max.max(x.to().abs());
                }
                let scale = if max > 0.0 { max / 127.0 } else { 1.0 };
                scales[g] = scale as f32;
                for (k, x) in w[s..e].iter().enumerate() {
                    let v = (x.to() / scale).round();
                    q[s + k] = v.clamp(-127.0, 127.0) as i8;
                }
            }
            payload.extend(q.iter().map(|&b| b as u8));
            for s in &scales {
                payload.extend_from_slice(&s.to_le_bytes());
            }
        }
    }
    let payload_len = payload.len();
    push_u64(&mut hdr, payload_len as u64);
    push_u64(&mut hdr, fnv1a64(&payload));

    let mut out = Vec::with_capacity(12 + hdr.len() + payload.len());
    out.extend_from_slice(MAGIC);
    push_u32(&mut out, hdr.len() as u32);
    out.extend_from_slice(&hdr);
    out.extend_from_slice(&payload);
    out
}

/// Parse the header; returns (cfg, tensor dir, payload slice bounds).
struct Parsed {
    cfg: ModelCfg,
    names: Vec<(&'static str, usize)>, // (name, elems) — name matched against statics
    payload: Vec<u8>,
    payload_len: usize,
    fnv: u64,
}

fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn rd_u64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}
fn rd_f64(b: &[u8], o: usize) -> f64 {
    f64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

fn parse(raw: &[u8]) -> Result<Parsed, String> {
    if raw.len() < 12 || &raw[0..8] != MAGIC {
        return Err("not a .tem file (magic mismatch — expected 'TEMST8V1'); is this a .bin checkpoint?".to_string());
    }
    let hdr_len = rd_u32(raw, 8) as usize;
    let h_end = 12usize.checked_add(hdr_len).ok_or("corrupt header length")?;
    if raw.len() < h_end {
        return Err(format!(".tem header truncated: have {} B, header needs {h_end} B", raw.len()));
    }
    let h = &raw[12..h_end];
    if h.len() < 4 {
        return Err(".tem header too small".to_string());
    }
    let ver = rd_u32(h, 0);
    if ver != VERSION {
        return Err(format!(".tem version {ver} not supported (this build reads v{VERSION})"));
    }
    let mut o = 4usize;
    let mut ci = [0u32; 10];
    for i in 0..10 {
        ci[i] = rd_u32(h, o);
        o += 4;
    }
    let mut cf = [0f64; 4];
    for i in 0..4 {
        cf[i] = rd_f64(h, o);
        o += 8;
    }
    let ntens = rd_u32(h, o) as usize;
    o += 4;
    // resolve names against the model's static name table for zero-alloc maps
    let known: &[&str] = &[
        "embed", "g1", "conv.w", "conv.gate", "conv.bg", "attn.wq", "attn.gq", "attn.lam_a",
        "attn.lam_b", "attn.scale", "rope.eta", "vres_g", "wo", "g2", "ffn.wg", "ffn.wu",
        "ffn.wd", "lora.a", "lora.b", "mem.wkw", "mem.wvw", "mem.gu", "mem.gb", "gf",
        "head1", "head2",
    ];
    let mut names = Vec::with_capacity(ntens);
    for _ in 0..ntens {
        if o >= h.len() {
            return Err(".tem tensor directory truncated".to_string());
        }
        let nl = h[o] as usize;
        o += 1;
        if o + nl > h.len() {
            return Err(".tem tensor name truncated".to_string());
        }
        let nm = &h[o..o + nl];
        o += nl;
        let nm = std::str::from_utf8(nm).map_err(|_| ".tem tensor name is not UTF-8".to_string())?;
        let stat = known.iter().find(|k| **k == nm).copied().ok_or_else(|| format!(".tem: unknown tensor '{nm}'"))?;
        let elems = rd_u64(h, o) as usize;
        o += 8;
        let group = rd_u32(h, o) as usize;
        o += 4;
        if group != GROUP {
            return Err(format!(".tem: tensor '{nm}' uses group {group}, this build reads {GROUP}"));
        }
        names.push((stat, elems));
    }
    let payload_len = rd_u64(h, o) as usize;
    o += 8;
    let fnv = rd_u64(h, o);
    let payload = raw[h_end..].to_vec();
    if payload.len() < payload_len {
        return Err(format!(".tem payload truncated: have {} B, need {payload_len} B", payload.len()));
    }
    Ok(Parsed { cfg: ModelCfg {
        d: ci[0] as usize, layers: ci[1] as usize, heads: ci[2] as usize, m: ci[3] as usize,
        ff: ci[4] as usize, lora_r: ci[5] as usize, conv_k: ci[6] as usize,
        fine: ci[7] as usize, coarse: ci[8] as usize, vocab: ci[9] as usize,
        mtp_alpha: cf[0], zloss_w: cf[1], cap: cf[2], rope_base: cf[3],
    }, names, payload, payload_len, fnv })
}

/// Load a `.tem` file into a fresh model. The checksum is verified before any
/// weights are touched; a corrupt file fails loudly with a friendly error.
pub fn load(path: &str) -> Result<Model<f32>, String> {
    let raw = std::fs::read(path).map_err(|e| format!("cannot read '{path}': {e}"))?;
    let p = parse(&raw)?;
    if fnv1a64(&p.payload[..p.payload_len]) != p.fnv {
        return Err(format!(
            "'.tem' checksum mismatch in '{path}' — file is corrupt (stored {:016x})",
            p.fnv
        ));
    }
    let mut model: Model<f32> = Model::new(p.cfg, 0);
    let params = model.params();
    if params.len() != p.names.len() {
        return Err(format!(
            "'.tem' tensor count {} does not match this build's model ({}) — different tessera version?",
            p.names.len(),
            params.len()
        ));
    }
    let mut off = 0usize; // byte offset into payload
    unsafe {
        for (k, pr) in params.iter().enumerate() {
            let (nm, elems) = p.names[k];
            if pr.name != nm {
                return Err(format!(".tem tensor {} is '{}' but this build expects '{}' — model layout mismatch", k, nm, pr.name));
            }
            let w = &mut *pr.w;
            if w.len() != elems {
                return Err(format!(".tem tensor '{nm}' has {elems} elems, model wants {}", w.len()));
            }
            let groups = (elems + GROUP - 1) / GROUP;
            let q = &p.payload[off..off + elems];
            let so = off + elems;
            for g in 0..groups {
                let s = g * GROUP;
                let e = ((g + 1) * GROUP).min(elems);
                let scale = f32::from_le_bytes(p.payload[so + g * 4..so + g * 4 + 4].try_into().unwrap());
                for (k2, item) in w[s..e].iter_mut().enumerate() {
                    *item = q[s + k2] as i8 as f32 * scale;
                }
            }
            off = so + groups * 4;
        }
    }
    Ok(model)
}

/// Write the model to `path` in `.tem` format.
pub fn save<T: crate::num::Num>(model: &mut Model<T>, path: &str) -> std::io::Result<()> {
    let blob = encode(model);
    if let Some(dir) = std::path::Path::new(path).parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).ok();
        }
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&blob)?;
    Ok(())
}

/// Byte length the equivalent f32 (TSS1-style) dump would have.
pub fn raw_f32_bytes(model: &mut Model<f32>) -> usize {
    let params = model.params();
    4 * 4 + params.iter().map(|p| 4 + 4 * unsafe { (*p.w).len() }).sum::<usize>()
}
