**Table 3.4 — GPU speedups over the CPU (RTX 4080 Laptop, 12 GB)**

| kernel | conditions | CPU | GPU | exactness vs CPU | kind |
|---|---|---|---|---|---|
| Boruvka MST (device) | N=4,000, float64, matrix resident | 1.0× (worst) | 5.4× | order 1.000 (bit-identical) | estimate (10 seeds) |
| VAT front end | N=4,000, float64, order only | 1.0× (worst) | 2.4× | order 1.000 (bit-identical) | estimate (10 seeds) |
| VAT front end | N=4,000, float64, UNMATCHED work | 1.0× (worst) | 5.6× | order 1.000 (bit-identical) | estimate (10 seeds) |
| Boruvka MST (device) | N=8,000, float64, matrix resident | 1.0× (worst) | 6.2× | order 1.000 (bit-identical) | estimate (10 seeds) |
| VAT front end | N=8,000, float64, order only | 1.0× (worst) | 3.0× | order 1.000 (bit-identical) | estimate (10 seeds) |
| VAT front end | N=8,000, float64, UNMATCHED work | 1.0× (worst) | 7.3× | order 1.000 (bit-identical) | estimate (10 seeds) |
| Boruvka MST (device) | N=16,000, float64, matrix resident | 1.0× (worst) | 5.4× | order 1.000 (bit-identical) | estimate (10 seeds) |
| VAT front end | N=16,000, float64, order only | 1.0× (worst) | 3.3× | order 1.000 (bit-identical) | estimate (10 seeds) |
| VAT front end | N=16,000, float64, UNMATCHED work | 1.0× (worst) | 8.1× | order 1.000 (bit-identical) | estimate (10 seeds) |
| Boruvka MST (device) | N=32,000, float64, matrix resident | 1.0× (worst) | 4.8× | order 1.000 (bit-identical) | estimate (10 seeds) |
| VAT front end | N=32,000, float64, order only | 1.0× (worst) | 4.3× | order 1.000 (bit-identical) | estimate (10 seeds) |
| VAT front end | N=32,000, float64, UNMATCHED work | 1.0× (worst) | 10.2× | order 1.000 (bit-identical) | estimate (10 seeds) |
| Fuzzy C-Means | N=50,000, k=10, d=20 | 1.0× (worst) | 5.1× | labels 1.000 ± 0.001; max abs Δcentre 4.9e-05 | estimate (10 seeds) |
| Fuzzy C-Means | N=50,000, k=10, d=20, MATCHED formulation | 1.0× (worst) | 2.6× | labels 1.000; max abs Δcentre 1.6e-13 | estimate (10 seeds) |
| Fuzzy C-Means | N=200,000, k=10, d=20 | 1.0× (worst) | 5.9× | labels 1.000; max abs Δcentre 4.8e-06 | estimate (10 seeds) |
| Fuzzy C-Means | N=200,000, k=10, d=20, MATCHED formulation | 1.0× (worst) | 3.3× | labels 1.000; max abs Δcentre 3.2e-13 | estimate (10 seeds) |
| Fuzzy C-Means | N=500,000, k=10, d=20 | 1.0× (worst) | 9.4× | labels 1.000; max abs Δcentre 9.8e-06 | estimate (10 seeds) |
| Fuzzy C-Means | N=500,000, k=10, d=20, MATCHED formulation | 1.0× (worst) | 5.1× | labels 1.000; max abs Δcentre 6.0e-13 | estimate (10 seeds) |
| Pairwise distances | N=16,000, d=10, float64, high_precision | 3.6× | 1.0× (worst) | max abs Δ = 7.1e-15 | estimate (10 seeds) |
| Pairwise distances | N=16,000, d=50, float64, high_precision | 3.2× | 1.0× (worst) | max abs Δ = 2.8e-14 | estimate (10 seeds) |
| Pairwise distances | N=16,000, d=200, float64, high_precision | 1.8× | 1.0× (worst) | max abs Δ = 7.1e-14 | estimate (10 seeds) |
| Pairwise distances | N=16,000, d=784, float64, high_precision | 2.6× | 1.0× (worst) | max abs Δ = 2.8e-13 | estimate (10 seeds) |
| Pairwise distances | N=16,000, d=10, float32, high_precision | 3.2× | 1.0× (worst) | max abs Δ = 0.0e+00 | estimate (10 seeds) |
| Pairwise distances | N=16,000, d=10, float32, fast (native acc) | 2.9× | 1.0× (worst) | max abs Δ = 3.8e-06 | estimate (10 seeds) |
| Pairwise distances | N=16,000, d=50, float32, high_precision | 3.0× | 1.0× (worst) | max abs Δ = 1.9e-06 | estimate (10 seeds) |
| Pairwise distances | N=16,000, d=50, float32, fast (native acc) | 2.2× | 1.0× (worst) | max abs Δ = 1.1e-05 | estimate (10 seeds) |
| Pairwise distances | N=16,000, d=200, float32, high_precision | 2.0× | 1.0× (worst) | max abs Δ = 3.8e-06 | estimate (10 seeds) |
| Pairwise distances | N=16,000, d=200, float32, fast (native acc) | 1.1× | 1.0× (worst) | max abs Δ = 3.8e-05 | estimate (10 seeds) |
| Pairwise distances | N=16,000, d=784, float32, high_precision | 1.5× | 1.0× (worst) | max abs Δ = 7.6e-06 | estimate (10 seeds) |
| Pairwise distances | N=16,000, d=784, float32, fast (native acc) | 1.0× (worst) | 1.4× | max abs Δ = 1.4e-04 | estimate (10 seeds) |
| VAT front end | N=48,000, float32, 9.22 GB resident | 1.0× (worst) | 3.5× | order 0.99992; Prim total CPU 42023.180315 vs GPU 42023.180315 (rel 0.00e+00) | demonstration (1 shot) |

> **Device.** NVIDIA GeForce RTX 4080 Laptop GPU, 12282 MiB, 610.88; compute capability 89; VRAM free 11.6/12.9 GB at start; CuPy 14.1.1; CUDA runtime 12.9. **The Markdown arms are normalized against the slower arm in each row**: the loser is the 1.0x baseline and the winner reads as 'this many times faster', so a row where **the GPU is the 1.0x baseline is a row the GPU loses**. Absolute seconds and their per-seed spreads are in the companion CSV; ratios survive a change of machine and seconds do not. 
>
> **Exactness is a column, not an assumption.** A speedup on a different answer is not a speedup, so every VAT/MST row compares its ordering elementwise against the compiled Cython serial reference on identical points, every FCM row compares hard labels and centres from identical initial centres, and every distance row reports the max absolute deviation from the CPU kernel. 
>
> **The Fuzzy C-Means row is quoted twice on purpose, and the chapter's version of it overstates the device.** `fcm.fuzzy_c_means` is a NumPy broadcasting implementation that materialises (n, k, d) and (n, k, k) temporaries; `gpu.fuzzy_c_means_gpu` uses the gram identity and two GEMMs. Those are different algorithms, so the ratio between them measures a rewrite as much as a card. The MATCHED rows run the GPU's own formulation in NumPy/BLAS on the CPU, and that is the device speedup. The FCM seconds carry a very large spread by construction: with the same initial centres and the same convergence test, the number of iterations to the fixed point varies from about 11 to the 100-iteration cap across the ten seeds, so a seed that runs nine times as long appears in both arms. All three arms see the same seeds and the reported figure is a ratio of means over them, but read the per-seed spread in the CSV before quoting any single FCM number. 
>
> **The VAT front end is also quoted twice.** The matched pair has both arms produce only an ordering. The UNMATCHED pair additionally has the CPU arm materialise the reordered n x n matrix (`compute_vat_c`), which the GPU arm never does; it is included because that is the comparison behind the chapter's cell, and it reads roughly three times higher. 
>
> **Two arms differing in kind is a real hazard here.** Ratios in this project are not machine-invariant when the arms differ in kind rather than in device -- a 40% cross-host move was measured in a ratio whose two arms were interpreted Python versus compiled Cython. The MST, front end and distance rows compare compiled CPU (Cython / C+OpenMP) against compiled CUDA and are safe on that count; the unmatched FCM row is not, which is why the matched one exists. 
>
> **The pairwise-distance rows are a NEGATIVE RESULT and are meant to be.** This card's FP64 throughput is a small fraction of its FP32, and the O(n^2) result must come back across PCIe, so the GPU loses at low dimension and at float64 -- the regime VAT actually lives in. The chapter predicts a datacenter card with full-rate FP64 would flip this. ** That prediction is UNTESTED and untestable on this host **; no cell here estimates it. 
>
> **kind.** Swept rows are ESTIMATES: ten seeds, spread in the CSV. The reachable-scale row is a single-shot DEMONSTRATION recorded with its hardware, precision and resident footprint instead of a spread (Chapter 7 §7.2), and at the VRAM edge its wall clock is volatile -- do not read it as an estimate. 
>
> **What is timed.** The MST row's matrix is already device-resident, so it times the kernel and not the transfer. Every device timing includes an explicit stream synchronise; without one a CUDA launch returns immediately and the arm would time as zero. RawModule JIT and cuBLAS handle creation are warmed before any measurement (the first `boruvka_mst_device` call spends ~0.4 s compiling, 13x the N=16,000 kernel time).

> Generated by `reproduce/`; seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]. `N/A` marks a cell whose method/dataset was unavailable.

> **Machine.** host: NEX-210200 · os: Windows-11 · cpu: Intel(R) Core(TM) i9-14900HX · cores: 32 physical, 32 logical · ram: 95.6 GiB · gpu: NVIDIA GeForce RTX 4080 Laptop GPU, 12282 MiB · python: 3.13.7
>
> Wall-clock times are machine-dependent; ratios are not. Markdown tables report normalized ratios where a timing is involved, and the companion CSV carries the absolute seconds.
