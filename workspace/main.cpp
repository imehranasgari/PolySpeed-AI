#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <cstdint>

/**
 * @brief Ultra-fast XorShift64 PRNG.
 * Designed for high-performance Monte Carlo simulations.
 * Note: Bitwise operations are performed on uint64_t to avoid the 
 * "invalid operands of types 'double' and 'int'" compiler error.
 */
struct XorShift64 {
    uint64_t state;

    explicit XorShift64(uint64_t seed) : state(seed == 0 ? 0xACE1u : seed) {}

    // Generates next 64-bit integer
    inline uint64_t next() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }

    // Generates a double in the range [0, 1)
    // We use the standard method of shifting and multiplying to ensure 
    // high precision and avoid bit-shifting a double.
    inline double next_double() {
        // Use the top 53 bits to ensure precision within a double's mantissa
        return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0);
    }
};

/**
 * @brief Estimates Pi using Monte Carlo method.
 * Parallelized using OpenMP for maximum throughput on multi-core architectures.
 * 
 * @param samples Total number of random points to sample.
 * @return double The estimated value of Pi.
 */
double estimate_pi(uint64_t samples) {
    uint64_t total_inside = 0;

    // Use OpenMP to distribute work across physical cores.
    // Each thread gets its own PRNG state to avoid race conditions and false sharing.
    #pragma omp parallel
    {
        // Seed with a combination of time and thread ID for uniqueness
        uint64_t thread_seed = static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count()) + omp_get_thread_num();
        XorShift64 rng(thread_seed);
        uint64_t local_inside = 0;

        #pragma omp for nowait
        for (uint64_t i = 0; i < samples; ++i) {
            double x = rng.next_double();
            double y = rng.next_double();
            if (x * x + y * y <= 1.0) {
                local_inside++;
            }
        }

        // Atomic reduction of the local count to the global count
        #pragma omp atomic
        total_inside += local_inside;
    }

    return (static_cast<double>(total_inside) / static_cast<double>(samples)) * 4.0;
}

int main() {
    const uint64_t samples = 15000000;

    // High-resolution timing
    auto start = std::chrono::high_resolution_clock::now();
    
    double pi = estimate_pi(samples);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Output formatting matching Python's f-string precision
    std::cout << std::fixed << std::setprecision(8) << "Estimated Pi: " << pi << std::endl;
    std::cout << std::fixed << std::setprecision(4) << "Execution Time: " << duration.count() << " seconds" << std::endl;

    return 0;
}