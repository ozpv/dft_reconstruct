#![allow(clippy::inline_always)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]

use num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

pub struct ChunkedRealFft {
    chunk_size: usize,
    planner: RealFftPlanner<f64>,
    forward: Arc<dyn RealToComplex<f64>>,
    inverse: Arc<dyn ComplexToReal<f64>>,
}

impl ChunkedRealFft {
    pub fn new(chunk_size: usize) -> Self {
        let mut planner = RealFftPlanner::new();
        let forward = planner.plan_fft_forward(chunk_size);
        let inverse = planner.plan_fft_inverse(chunk_size);

        Self {
            chunk_size,
            planner,
            forward,
            inverse,
        }
    }

    pub fn chunk_size(mut self, value: usize) -> Self {
        self.chunk_size = value;

        let forward = self.planner.plan_fft_forward(value);
        let inverse = self.planner.plan_fft_inverse(value);

        self.forward = forward;
        self.inverse = inverse;

        self
    }

    #[inline(always)]
    fn forward_real_fft(&mut self, real_signal: impl Into<Vec<f64>>) -> Vec<Complex<f64>> {
        let mut real_signal = real_signal.into();

        let mut output = self.forward.make_output_vec();

        self.forward.process(&mut real_signal, &mut output).unwrap();

        output
    }

    pub fn forward(&mut self, data: impl Into<Vec<f64>>) -> Vec<Vec<Complex<f64>>> {
        let mut data = data.into();

        let Some(new_length) = data.len().checked_next_multiple_of(self.chunk_size) else {
            return vec![];
        };

        data.resize(new_length, 0.0);

        data.chunks_exact(self.chunk_size)
            .map(|chunk| self.forward_real_fft(chunk))
            .collect::<Vec<Vec<Complex<f64>>>>()
    }

    #[inline(always)]
    fn inverse_real_fft(&mut self, mut fft_real_signal: Vec<Complex<f64>>) -> Vec<f64> {
        let mut real_output = self.inverse.make_output_vec();

        self.inverse
            .process(&mut fft_real_signal, &mut real_output)
            .unwrap();

        let chunk_size_f64 = self.chunk_size as f64;

        real_output
            .into_iter()
            .map(|sample| sample / chunk_size_f64)
            .collect::<Vec<f64>>()
    }

    pub fn inverse(&mut self, data: Vec<Vec<Complex<f64>>>) -> Vec<f64> {
        data.into_iter()
            .flat_map(|chunk| self.inverse_real_fft(chunk))
            .collect::<Vec<f64>>()
    }

    pub fn i32_to_f64(item: Vec<i32>) -> Vec<f64> {
        item.into_iter().map(f64::from).collect::<Vec<f64>>()
    }

    pub fn f64_to_i32(item: Vec<f64>) -> Vec<i32> {
        item.into_iter()
            .map(|value| value as i32)
            .collect::<Vec<i32>>()
    }
}
