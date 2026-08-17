# Which problems are actually slow to converge?

Minibatch updates for a He-initialized 128-unit ReLU network (batch 128, Adam, best of [0.001, 0.003, 0.01]) to first reach R2 >= 0.9 on a 20% holdout, mean of 2 seeds, capped at 20,000.

Updates, not seconds and not epochs: updates are what an initialization can skip, and unlike epochs they are comparable across dataset sizes. **PhiUSIIL needed 25.** A warm start costing a 2 s FIS fit needs a problem in the thousands before it can repay itself.

| problem | rows x features | updates to R2>=0.9 | best R2 | what it is |
|---|---|---|---|---|
| `chirp-k8` | 4,000 x 1 | never (best R2 0.62) | 0.624 | sin(2*pi*8*x^2), 1 input |
| `chirp-k16` | 4,000 x 1 | never (best R2 0.34) | 0.341 | sin(2*pi*16*x^2), 1 input |
| `chirp-k40` | 4,000 x 1 | never (best R2 0.18) | 0.182 | sin(2*pi*40*x^2), 1 input |
| `sine2d-k4` | 8,000 x 2 | never (best R2 0.73) | 0.732 | sin(4x)sin(4y), 2 inputs |
| `sine2d-k8` | 8,000 x 2 | never (best R2 0.01) | 0.013 | sin(8x)sin(8y), 2 inputs |
| `pendulum-n2` | 62,000 x 3 | never (best R2 0.76) | 0.764 | n=2 time-step operator |
| `pendulum-n3` | 62,000 x 4 | never (best R2 0.69) | 0.688 | n=3 time-step operator |
| `sine2d-k2` | 8,000 x 2 | **8,699** | 0.995 | sin(2x)sin(2y), 2 inputs |
| `chirp-k4` | 4,000 x 1 | **8,018** | 0.959 | sin(2*pi*4*x^2), 1 input |
| `pendulum-n2-fric` | 62,000 x 3 | **3,444** | 0.979 | n=2 + friction time-step operator |
| `concrete` | 1,030 x 8 | **386** | 0.941 | reference (Part 1 workhorse) |
| `illcond` | 8,000 x 20 | **36** | 1.000 | 20 inputs, condition number 10000 |

