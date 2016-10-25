fn kernel(x1 : &Vec<f64>, x2 : &Vec<f64>) -> f64 {
	return linear_kernel(x1, x2);
}

fn dot_product(x1 : &Vec<f64>, x2 : &Vec<f64>) -> f64 {
	return linear_kernel(x1, x2);
}

fn linear_kernel(x1 : &Vec<f64>, x2 : &Vec<f64>) -> f64 {
	assert_eq!(x1.len(), x2.len());
	let mut result = 0.0;
	for i in 0..x1.len() {
		result += x1[i] * x2[i];
	}
	return result;
}

fn gaussian_kernel(x1 : &Vec<f64>, x2 : &Vec<f64>) -> f64 {
	assert_eq!(x1.len(), x2.len());
	let gamma = 1.0;
	let mut result = 0.0;
	for i in 0..x1.len() {
		result += (x1[i] - x2[i]).powf(2.0);
	}
	return (-gamma*result).exp();
}

pub struct KevzSVM {
	pub lambda: f64,
	pub alpha: Vec<f64>,
	pub vector: Vec<Vec<f64>>
}

impl KevzSVM {
	pub fn new(lambda: f64) -> KevzSVM {
		assert!(lambda >= 0.0);

		KevzSVM {
			lambda: lambda,
			alpha: Vec::new(),
			vector: Vec::new()
		}
	}

	pub fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<f64>) {
		assert_eq!(x.len(), y.len());
		for x_i in x {
			assert_eq!(x[0].len(), x_i.len())
		}

		let num_samples = x.len();
		let mut gram = vec![vec![0.0; num_samples]; num_samples];
		for i in 0..num_samples {
			for j in 0..num_samples {
				gram[i][j] = kernel(&x[i], &x[j]);
			}
		}
		println!("Kernelized!");

		let mut t = 0.0;
		let mut alpha = vec![0.0; num_samples];
		for epoch in 0..1000 {
			let mut right = 0.0;
			let mut total = 0.0;
			for i in 0..num_samples {
				t += 1.0;
				let lr = 1.0 / (t * self.lambda);

				let result = y[i] * lr * dot_product(&alpha, &gram[i]);
				if result > 0.0 {
					right += 1.0;
				}
				total += 1.0;

				if lr * result < 1.0 {
					alpha[i] = (1.0 - lr * self.lambda) * alpha[i] + lr * y[i];
				} else {
					alpha[i] = (1.0 - lr * self.lambda) * alpha[i];
				}
			}
			if epoch % 10 == 0 {
				println!("Epoch {0}: {1}", epoch, right / total);
			}
		}

		self.alpha = alpha;
		self.vector = x.clone();
	}
	
	pub fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<f64> {
		let num_samples = x.len();
		let mut y = vec![0.0; num_samples];
		for i in 0..x.len() {
			for j in 0..self.vector.len() {
				y[i] += self.alpha[j] * kernel(&x[i], &self.vector[j]);
			}
		}
		return y;
    }
}
