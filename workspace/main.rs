use std::thread;
use std::time::Instant;

/// Ultra-fast XorShift64 PRNG implementation for high-performance Monte Carlo simulations.
/// This replaces the need for external crates like `rand`.
struct Rng(u64);

impl Rng {
    #[inline(always)]
    fn new(seed: u64) -> Self {
        // Ensure seed is not zero
        let s = if seed == 0 { 0xACE1u64 } else { seed };
        Rng(s)
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    #[inline(always)]
    fn next_f64(&mut self) -> f64 {
        // Convert u64 to f64 in range [0, 1)
        // Using the standard method of dividing by max u64 value
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}

fn main() {
    let total_samples: u64 = 15_000_000;
    
    // Determine hardware concurrency for optimal scaling
    // On a 12th Gen i7-12700H, this will likely be 20 (logical cores)
    let num_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let samples_per_thread = total_samples / num_threads as u64;
    let remainder = total_samples % num_threads as u64;

    let start = Instant::now();

    // Vector to hold thread handles
    let mut handles = Vec::with_capacity(num_threads);

    for i in 0..num_threads {
        // Distribute the remainder to the first thread
        let count = if i == 0 {
            samples_per_thread + remainder
        } else {
            samples_per_thread
        };

        // Seed the PRNG with a combination of time and thread index to ensure
        // different sequences per thread without external dependencies.
        let seed = (Instant::now().elapsed().as_nanos() as u64) ^ (i as u64);

        let handle = thread::spawn(move || {
            let mut rng = Rng::new(seed);
            let mut inside_count: u64 = 0;
            
            for _ in 0..count {
                let x = rng.next_f64();
                let y = rng.next_f64();
                if x * x + y * y <= 1.0 {
                    inside_count += 1;
                }
            }
            inside_count
        });
        handles.push(handle);
    }

    // Aggregate results from all threads
    let mut total_inside: u64 = 0;
    for handle in handles {
        match handle.join() {
            Ok(count) => total_inside += count,
            Err(_) => eprintln!("Thread panicked!"),
        }
    }

    let pi = (total_inside as f64 / total_samples as f64) * 4.0;
    let duration = start.elapsed();

    // Output formatting matches Python's f-string precision
    println!("Estimated Pi: {:.8}", pi);
    println!("Execution Time: {:.4} seconds", duration.as_secs_f64());
}