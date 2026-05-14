#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::unused)]

use std::{f64::consts::PI, iter::zip};

enum Overlap {
    None,
    OneQuarter,
    Half,
    ThreeQuarter,
}

impl Overlap {
    fn step_by(self, window_size: usize) -> usize {
        match self {
            Self::None => window_size,
            Self::OneQuarter => window_size * 3 / 4,
            Self::Half => window_size / 2,
            Self::ThreeQuarter => window_size / 4,
        }
    }
}

pub trait WindowFunction {
    fn apply(&self, signal: Vec<f64>) -> Vec<Vec<f64>>;
    fn reverse(&self, signal: Vec<Vec<f64>>) -> Vec<f64>;
}

pub struct HannWindow {
    window_size: usize,
    function: Vec<f64>,
}

impl HannWindow {
    pub fn new(window_size: usize) -> Self {
        let window_size_f64 = window_size as f64;

        let function = (0..window_size)
            .map(|n| {
                let n = n as f64;

                0.5 * (1.0 - f64::cos((2.0 * PI * n) / (window_size_f64 - 1.0)))
            })
            .collect::<Vec<f64>>();

        Self {
            window_size,
            function,
        }
    }
}

impl WindowFunction for HannWindow {
    fn apply(&self, signal: Vec<f64>) -> Vec<Vec<f64>> {
        let half_window_size = self.window_size / 2;

        let mut padded = vec![0.0; half_window_size];
        padded.extend_from_slice(&signal);

        padded
            .windows(self.window_size)
            .step_by(half_window_size)
            .map(|slice| {
                zip(slice, self.function.iter())
                    .map(|(sample, multiplier)| sample * multiplier)
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>()
    }

    fn reverse(&self, signal: Vec<Vec<f64>>) -> Vec<f64> {
        let half_window_size = self.window_size / 2;
        let len = (signal.len() - 1) * half_window_size + self.window_size;

        let mut output = vec![0.0; len];
        let mut window_sum = vec![0.0; len];

        for (i, slice) in signal.iter().enumerate() {
            let start = i * half_window_size;
            for j in 0..self.window_size {
                output[start + j] += slice[j] * self.function[j];
                window_sum[start + j] += self.function[j] * self.function[j];
            }
        }

        output
            .into_iter()
            .skip(half_window_size)
            .zip(window_sum.into_iter().skip(half_window_size))
            .map(|(s, w)| if w > 1e-6 { s / w } else { 0.0 })
            .collect::<Vec<f64>>()
    }
}

pub struct HammingWindow {
    window_size: usize,
    function: Vec<f64>,
}

impl HammingWindow {
    pub fn new(window_size: usize) -> Self {
        let window_size_f64 = window_size as f64;

        let function = (0..window_size)
            .map(|n| {
                let n = n as f64;

                0.53836 - (0.46164 * f64::cos((2.0 * PI * n) / (window_size_f64 - 1.0)))
            })
            .collect::<Vec<f64>>();

        Self {
            window_size,
            function,
        }
    }
}

impl WindowFunction for HammingWindow {
    fn apply(&self, signal: Vec<f64>) -> Vec<Vec<f64>> {
        let half_window_size = self.window_size / 2;

        let mut padded = vec![0.0; half_window_size];
        padded.extend_from_slice(&signal);

        padded
            .windows(self.window_size)
            .step_by(half_window_size)
            .map(|slice| {
                zip(slice, self.function.iter())
                    .map(|(sample, multiplier)| sample * multiplier)
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>()
    }

    fn reverse(&self, signal: Vec<Vec<f64>>) -> Vec<f64> {
        let half_window_size = self.window_size / 2;
        let len = (signal.len() - 1) * half_window_size + self.window_size;

        let mut output = vec![0.0; len];
        let mut window_sum = vec![0.0; len];

        for (i, slice) in signal.iter().enumerate() {
            let start = i * half_window_size;
            for j in 0..self.window_size {
                output[start + j] += slice[j] * self.function[j];
                window_sum[start + j] += self.function[j] * self.function[j];
            }
        }

        output
            .into_iter()
            .skip(half_window_size)
            .zip(window_sum.into_iter().skip(half_window_size))
            .map(|(s, w)| if w > 1e-6 { s / w } else { 0.0 })
            .collect::<Vec<f64>>()
    }
}

pub struct BlackmanHarrisWindow {
    window_size: usize,
    function: Vec<f64>,
}

// TODO: add more term counts
impl BlackmanHarrisWindow {
    pub fn new(window_size: usize) -> Self {
        let window_size_f64 = window_size as f64;
        let a_0 = 0.35875;
        let a_1 = 0.48829;
        let a_2 = 0.14128;
        let a_3 = 0.01168;

        let function = (0..window_size)
            .map(|n| {
                let n = n as f64;

                let two = f64::cos((2.0 * PI * n) / (window_size_f64 - 1.0));
                let four = f64::cos((4.0 * PI * n) / (window_size_f64 - 1.0));
                let six = f64::cos((6.0 * PI * n) / (window_size_f64 - 1.0));

                a_0 - (a_1 * two) + (a_2 * four) - (a_3 * six)
            })
            .collect::<Vec<f64>>();

        Self {
            window_size,
            function,
        }
    }
}

impl WindowFunction for BlackmanHarrisWindow {
    fn apply(&self, signal: Vec<f64>) -> Vec<Vec<f64>> {
        let quarter_window_size = self.window_size / 4;

        let mut padded = vec![0.0; quarter_window_size];
        padded.extend_from_slice(&signal);

        padded
            .windows(self.window_size)
            .step_by(quarter_window_size)
            .map(|slice| {
                zip(slice, self.function.iter())
                    .map(|(sample, multiplier)| sample * multiplier)
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>()
    }

    fn reverse(&self, signal: Vec<Vec<f64>>) -> Vec<f64> {
        let quarter_window_size = self.window_size / 4;
        let len = (signal.len() - 1) * quarter_window_size + self.window_size;

        let mut output = vec![0.0; len];
        let mut window_sum = vec![0.0; len];

        for (i, slice) in signal.iter().enumerate() {
            let start = i * quarter_window_size;
            for j in 0..self.window_size {
                output[start + j] += slice[j] * self.function[j];
                window_sum[start + j] += self.function[j] * self.function[j];
            }
        }

        output
            .into_iter()
            .skip(quarter_window_size)
            .zip(window_sum.into_iter().skip(quarter_window_size))
            .map(|(s, w)| if w > 1e-6 { s / w } else { 0.0 })
            .collect::<Vec<f64>>()
    }
}

pub struct RectangularWindow {
    window_size: usize,
}

impl RectangularWindow {
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }
}

impl WindowFunction for RectangularWindow {
    fn apply(&self, signal: Vec<f64>) -> Vec<Vec<f64>> {
        signal
            .chunks_exact(self.window_size)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<Vec<f64>>>()
    }

    fn reverse(&self, signal: Vec<Vec<f64>>) -> Vec<f64> {
        signal.into_iter().flatten().collect::<Vec<f64>>()
    }
}
