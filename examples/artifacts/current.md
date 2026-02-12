# Benchbro Results

- started_at: `2026-02-12T23:33:26.580779+00:00`
- finished_at: `2026-02-12T23:33:26.717250+00:00`
- python_version: `3.13.11`
- platform: `macOS-15.7.3-arm64-arm-64bit-Mach-O`

| case | benchmark | metric_type | mean_s | median_s | p95_s | stddev_s | ops_per_sec | peak_alloc_bytes | net_alloc_bytes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hashing | sha1_digest | time | 1.10536e-06 | 1.105e-06 | 1.105e-06 | 9.49522e-09 | 904681 | - | - |
| hashing | sha256_digest | time | 1.11353e-06 | 1.106e-06 | 1.106e-06 | 2.37057e-08 | 898047 | - | - |
| allocations | list_allocation | memory | - | - | - | - | - | 52560 | 0 |
| allocations | dict_allocation | memory | - | - | - | - | - | 146232 | 0 |
