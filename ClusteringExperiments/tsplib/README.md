# TSPLIB
Library of Traveling Salesman problems from http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/ <sup>[1]</sup>

This is just the Symmetric TSP data.
Currently known best solutions are in the `solutions` file.

<sup>[1]</sup> Reinelt, G. "TSPLIB--A Traveling Salesman Problem Library." ORSA Journal on Computing, Vol. 3, No. 4, pp. 376-384. Fall 1991.

## Getting the `.tsp` files

The instance files are **not tracked** (standard TSPLIB95 data, ~9 MB). Fetch
them into this directory with:

```bash
python ClusteringExperiments/tsplib/download.py
```

`instances.txt` pins the exact set the experiments use; `solutions` holds the
known-optimum tour lengths. See `download.py` for the source and options.
