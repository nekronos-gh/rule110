#pragma GCC target("popcnt") // Enable the POPCNT assembly instruction
#pragma GCC target("avx2")   // Enable the AVX2

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <omp.h>

typedef struct packed_buffer {
  uint64_t *current_buffer = nullptr;
  uint64_t *next_buffer = nullptr;
  size_t real_words = 0;   // number of uint64_t
  size_t groups = 0;       // number of __m256i groups (4 words each)
  size_t padded_words = 0; // groups * 4
  size_t ghost_offset = 0; // offset until actual data
  uint64_t seed_bits = 0;
  uint64_t last_mask = ~0ULL; // mask for last real word
  uint64_t firstbit_mask = ~1ULL; // Mask for first bit
  uint64_t lastbit_mask = 0ULL; // Mask for last bit
  size_t first_real_word = 0;
  size_t last_real_word = 0;
} packed_buffer_t;

// ----------------------------------
// Lane 0 | Lane 1 | Lane 2 | Lane 3
// ----------------------------------
#define LOAD_CENTER(buf, group_idx)                                            \
  _mm256_load_si256((__m256i *)(&(buf)[(group_idx) * 4]))
#define STORE_CENTER(buf, group_idx, v)                                        \
  _mm256_store_si256((__m256i *)(&(buf)[(group_idx) * 4]), (v))
#define LOAD_LEFT(buf, group_idx)                                              \
  _mm256_loadu_si256((__m256i *)(&(buf)[(group_idx) * 4 - 1]))
#define LOAD_RIGHT(buf, group_idx)                                             \
  _mm256_loadu_si256((__m256i *)(&(buf)[(group_idx) * 4 + 1]))

#define MAKE_LEFT(center, left_src)                                            \
  _mm256_or_si256(_mm256_slli_epi64(center, 1),                                \
                  _mm256_srli_epi64(left_src, 63));

#define MAKE_RIGHT(center, right_src)                                          \
  _mm256_or_si256(_mm256_srli_epi64(center, 1),                                \
                  _mm256_slli_epi64(right_src, 63));

// By applying boolean algebra to the truth table and Karnaugh Maps we obtain:
// (pack ^ right) | ((~left) & pack);
#define RULE110(left, center, right)                                           \
  _mm256_or_si256(_mm256_xor_si256((center), (right)),                         \
                  _mm256_andnot_si256((left), (center)))

void transform110_packed_avx(const uint64_t *current_buffer,
                             uint64_t *next_buffer, size_t groups,
                             size_t ghost_offset) {
  // The strategy is to shift the lane left and right, and then XOR it.
  // That is exacly what we do for each group.
#pragma omp for schedule(static)
  for (size_t group_idx = ghost_offset; group_idx < ghost_offset + groups;
       ++group_idx) {
    __m256i center = LOAD_CENTER(current_buffer, group_idx);
    __m256i left_src =
        LOAD_LEFT(current_buffer, group_idx); // starts at word group_idx*4-1
    __m256i right_src =
        LOAD_RIGHT(current_buffer, group_idx); // starts at word group_idx*4+1
    __m256i left = MAKE_LEFT(center, left_src);
    __m256i right = MAKE_RIGHT(center, right_src);
    STORE_CENTER(next_buffer, group_idx, RULE110(left, center, right));
  }
}

void simulate(uint64_t steps, packed_buffer_t &packed_sol) {

  if (packed_sol.real_words == 0) {
    std::cout << 0 << '\n';
    return;
  }

  for (uint64_t s = 0; s < steps; ++s) {
    transform110_packed_avx(packed_sol.current_buffer, packed_sol.next_buffer,
                            packed_sol.groups, packed_sol.ghost_offset);
    // Clear boundary bits (only on real data)
    packed_sol.next_buffer[packed_sol.first_real_word] &= packed_sol.firstbit_mask;
    packed_sol.next_buffer[packed_sol.last_real_word] &= packed_sol.lastbit_mask;
    std::swap(packed_sol.current_buffer, packed_sol.next_buffer);
  }

  // Clear excess calculations
  packed_sol.next_buffer[packed_sol.last_real_word] &= packed_sol.last_mask;

  uint64_t one_count = 0;
#pragma omp parallel for reduction(+ : one_count)
  for (size_t i = 0; i < packed_sol.real_words; ++i) {
    one_count += std::popcount(packed_sol.current_buffer[packed_sol.first_real_word + i]);
  }

  std::cout << one_count;
}

static bool read_bits_from_file_packed(const char *filename,
                                       packed_buffer_t &out) {
  std::ifstream file(filename);
  std::string config;
  file >> out.seed_bits >> config;

  out.real_words =
      static_cast<size_t>((out.seed_bits + 63) >> 6); // Faster / 64
  out.groups = (out.real_words + 3) >> 2;             // Fatster / 4
  out.padded_words = out.groups << 2;                 // Faster * 4

  // Compute mask for the last real word.
  uint64_t bits_in_last_pack = out.seed_bits & 63; // Faster % 64
  out.last_mask = ((1ULL << bits_in_last_pack) - 1) | -(bits_in_last_pack == 0);

  // Compute masks for first and last bit
  uint64_t last_bit_idx =
      (out.seed_bits == 0) ? 0 : (out.seed_bits - 1);
  out.lastbit_mask = ~(1ULL << last_bit_idx & 63); // Mask for last bit

  // Add 2 ghost groups (1 on each side) = 8 extra words
  out.ghost_offset = 2;                     // 2 groups padding before data
  out.padded_words = (out.groups + 4) << 2; // +4 groups (2 on each side)

  // Adjust word index for ghost offset
  out.first_real_word = out.ghost_offset * 4;
  size_t last_word = static_cast<size_t>(last_bit_idx >> 6);
  out.last_real_word = out.first_real_word + last_word;

  // Allocate aligned and zero-initialize
  out.current_buffer = new (std::align_val_t(32)) uint64_t[out.padded_words]();
  out.next_buffer = new (std::align_val_t(32)) uint64_t[out.padded_words]();
  // We will not free memory, we are here for speed :)))

  // Pack bits (LSB-first in each 64-bit word)
  size_t data_offset = out.ghost_offset * 4; // To skip ghost zone
  for (size_t i = 0; i < config.size(); ++i) {
    if (config[i] == '1') {
      size_t word_idx = (i >> 6) + data_offset; // Fast i / 64
      size_t bit_pos = i & 63;                  // Fast i % 64
      out.current_buffer[word_idx] |= (1ULL << bit_pos);
    }
  }
  return true;
}

int main(int argc, char **argv) {
  uint64_t steps = 10;
  char *init_path = nullptr;
  packed_buffer_t pack_buf;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--iter" && i + 1 < argc) {
      steps = std::stoull(argv[++i]);
    } else if (std::string(argv[i]) == "--init" && i + 1 < argc) {
      init_path = argv[++i];
    }
  }

  if (!init_path) {
    return 1;
  }
  if (!read_bits_from_file_packed(init_path, pack_buf))
    return 1;

  // After setup, begin simulation
  simulate(steps, pack_buf);
  return 0;
}
