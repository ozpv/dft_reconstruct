#![feature(iter_array_chunks)]
#![allow(clippy::inline_always)]

use clap::Parser;
use dft_reconstruct::windowed_fft::ChunkedRealFft;
use hashbrown::HashSet;
use itertools::Itertools;
use num_complex::Complex;
use std::{fs, path::Path, time::Instant};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    order: usize,
}

#[allow(clippy::struct_field_names)]
struct Data {
    header: Vec<u8>,
    data_offset: usize,
    left: Vec<i32>,
    right: Vec<i32>,
}

fn read_32_bit_stereo_pcm_wav(file: impl AsRef<Path>) -> std::io::Result<Data> {
    let bytes = fs::read(file)?;
    let data_offset = bytes.windows(4).position(|s| s == b"data").unwrap();

    let mut bytes_iter = bytes.into_iter();

    let header = bytes_iter
        .by_ref()
        .take(data_offset + 8)
        .collect::<Vec<u8>>();

    let (left, right) = bytes_iter
        .array_chunks::<4>()
        .map(i32::from_le_bytes)
        // (l, r), (l, r), (l, r), ...
        .tuples::<(i32, i32)>()
        // Vec<(l, r)> into (Vec<l>, Vec<r>)
        .unzip::<i32, i32, Vec<i32>, Vec<i32>>();

    Ok(Data {
        header,
        data_offset,
        left,
        right,
    })
}

fn write_32_bit_stereo_samples_as_pcm_wav(
    output_file: impl AsRef<Path>,
    mut header: Vec<u8>,
    data_offset: usize,
    left: Vec<i32>,
    right: Vec<i32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut data = left
        .into_iter()
        .interleave(right)
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<u8>>();

    // update data length
    header[data_offset + 4..data_offset + 8]
        .copy_from_slice(&u32::to_le_bytes(u32::try_from(data.len())?));

    header.append(&mut data);

    fs::write(output_file, header)?;

    Ok(())
}

#[inline(always)]
fn retain_top_n_magnitudes(fft_real_signal: &mut [Complex<f64>], n: usize) {
    // you'd be keeping the entire signal
    if n >= fft_real_signal.len() {
        return;
    }

    let mut indexed = fft_real_signal
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<(usize, Complex<f64>)>>();

    let target = fft_real_signal.len() - n;
    indexed.select_nth_unstable_by(target, |(_, c0), (_, c1)| c0.norm().total_cmp(&c1.norm()));

    let top_indices = indexed[target..]
        .iter()
        .map(|&(i, _)| i)
        .collect::<HashSet<usize>>();

    for (i, val) in fft_real_signal.iter_mut().enumerate() {
        if !top_indices.contains(&i) {
            *val = Complex::ZERO;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let now = Instant::now();

    let Data {
        header,
        data_offset,
        left,
        right,
    } = read_32_bit_stereo_pcm_wav("signal.wav")?;

    let mut fft = ChunkedRealFft::new(32768);

    let mut left = fft.forward(ChunkedRealFft::i32_to_f64(left));

    for chunk in &mut left {
        retain_top_n_magnitudes(chunk, args.order);
    }

    let left = fft.inverse(left);

    let mut right = fft.forward(ChunkedRealFft::i32_to_f64(right));

    for chunk in &mut right {
        retain_top_n_magnitudes(chunk, args.order);
    }

    let right = fft.inverse(right);

    write_32_bit_stereo_samples_as_pcm_wav(
        "output_signal.wav",
        header,
        data_offset,
        ChunkedRealFft::f64_to_i32(left),
        ChunkedRealFft::f64_to_i32(right),
    )?;

    println!("Finished in {:?}", now.elapsed());

    Ok(())
}
