//! TESSERA-Q — an O(1)-working-memory autoregressive sequence model built from
//! bare-metal Rust with AVX-512/AVX2 SIMD kernels.
//!
//! Modules map 1:1 to the 9 invariant pillars of the specification.

pub mod attention;
pub mod baseline;
pub mod conv;
pub mod data;
pub mod ffn;
pub mod gradcheck;
pub mod layer;
pub mod mrm;
pub mod model;
pub mod norms;
pub mod num;
pub mod optim;
pub mod rng;
pub mod pjrt;
pub mod rope;
pub mod tem;
