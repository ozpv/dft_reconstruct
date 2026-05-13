#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]

use crate::window::{HannWindow, HammingWindow, BlackmanHarris, RectangularWindow, WindowFunction};
use num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

pub struct WindowedRealFft {
    fft_size: usize,
    planner: RealFftPlanner<f64>,
    forward: Arc<dyn RealToComplex<f64>>,
    inverse: Arc<dyn ComplexToReal<f64>>,
    window_function: Box<dyn WindowFunction>,
    original_length: usize,
}

impl WindowedRealFft {
    pub fn new(fft_size: usize) -> Self {
        let mut planner = RealFftPlanner::new();
        let forward = planner.plan_fft_forward(fft_size);
        let inverse = planner.plan_fft_inverse(fft_size);

        let window_function = Box::new(HannWindow::new(fft_size));

        Self {
            fft_size,
            planner,
            forward,
            inverse,
            window_function,
            original_length: 0,
        }
    }

    pub fn fft_size(mut self, value: usize) -> Self {
        self.fft_size = value;

        let forward = self.planner.plan_fft_forward(value);
        let inverse = self.planner.plan_fft_inverse(value);

        self.forward = forward;
        self.inverse = inverse;

        self
    }

    pub fn original_length(&mut self, value: usize) {
        self.original_length = value;
    }

    pub fn forward(&mut self, data: impl Into<Vec<f64>>) -> Vec<Vec<Complex<f64>>> {
        let mut data = data.into();

        let Some(new_length) = data.len().checked_next_multiple_of(self.fft_size) else {
            return vec![];
        };

        self.original_length = data.len();

        data.resize(new_length, 0.0);

        self.window_function
            .apply(data)
			.into_iter()
            .map(|mut chunk| {
                let mut output = self.forward.make_output_vec();

                self.forward
                    .process(&mut chunk, &mut output)
                    .unwrap();

                output
            })
            .collect::<Vec<Vec<Complex<f64>>>>()
    }

    fn inverse_real_fft(&mut self, mut fft_real_signal: Vec<Complex<f64>>) -> Vec<f64> {
        let mut real_output = self.inverse.make_output_vec();

        self.inverse
            .process(&mut fft_real_signal, &mut real_output)
            .unwrap();

        let chunk_size_f64 = self.fft_size as f64;

        real_output
            .into_iter()
            .map(|sample| sample / chunk_size_f64)
            .collect::<Vec<f64>>()
    }

    pub fn inverse(&mut self, data: Vec<Vec<Complex<f64>>>) -> Vec<f64> {
        let data = data
            .into_iter()
            .map(|chunk| self.inverse_real_fft(chunk))
            .collect::<Vec<Vec<f64>>>();

        self.window_function
            .reverse(data)
			.into_iter()
            .take(self.original_length)
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
