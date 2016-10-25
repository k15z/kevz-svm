mod core;
use core::*;

use std::io;
use std::io::prelude::*;
use std::io::BufReader;
use std::fs::File;

fn main() {
	let mut x = vec![];
	let mut y = vec![];

	let f = File::open("data/mnist_digit_1.csv").unwrap();
	let f = BufReader::new(f);
	for line in f.lines() {
		let line = line.unwrap();
		let line = line.split(" ");
		let line = line.map(|s| s.parse::<f64>().unwrap()).collect::<Vec<_>>();
		x.push(line.clone());
		y.push(1.0);
	}

	let f = File::open("data/mnist_digit_7.csv").unwrap();
	let f = BufReader::new(f);
	for line in f.lines() {
		let line = line.unwrap();
		let line = line.split(" ");
		let line = line.map(|s| s.parse::<f64>().unwrap()).collect::<Vec<_>>();
		x.push(line.clone());
		y.push(-1.0);
	}

	println!("Ready!");
	let mut svm = KevzSVM::new(0.1);
	svm.fit(&x, &y);
}
