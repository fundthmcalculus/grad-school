# Paper 2: MergeVAT: $58K \times 58K$ in 60 seconds

---

## Motivation: Go, Fuzzy! - faster!

<div style="display: flex;">
<div style="flex: 1; padding: 10px;">

* UC Irvine NASA Dataset
    * Space Shuttle reentry
    * 80% of data in condition-1
    * 58,000 rows
* Can we visualize and confirm that?
    * This image is 1% linear scale, 1/10,000 in area
    * 8-bit grey-scale PNG is >400 MB
</div>
<div style="flex: 1; padding: 10px;">

![alt text](img/paper2/image-11.png)

</div>
</div>



---

## Patient Data

<div style="display: flex;">
<div style="flex: 1; padding: 10px;">

* 135K rows - 30% clustered, 70% sparse - mostly binary values
* Can we visualize and confirm that?
    * 8 minutes for VAT, 15 minutes for distance matrix calculation
    * At 32-bit floating point, this is 73GB

</div>
<div style="flex: 1; padding: 10px;">

![alt text](img/paper2/image-12.png)
</div>
</div>

---

## Scaling Time Complexity

<div style="display: flex;">
<div style="flex: 1; padding: 10px;">

* VAT gets the arg-min of the remainder of the current column
* This sorting operation is typically BubbleSort, $O(N)=N^2$
* This is applied on every column, so overall $O(N)=N^3$
> At 135K rows, my improved method is ~8000 times faster
</div>
<div style="flex: 1; padding: 10px;">

![image13](img/paper2/image-13.png)
</div>
</div>

---

## The First Insight - Sort Algorithm

<div style="display: flex;">
<div style="flex: 1; padding: 10px;">

* MergeSort is the asymptotically fastest algorithm which can exist: $O(N)=N \log N$
* Over $N$ columns, we have $O(N)=N^2 \log N$
* N-scaling=24 is a 16K element dataset
* Utilize a priority queue (fibbonacci heap) to extract the remainder index as $O(N)=1$ operation

> Professor Kreinovich pointed out this method is more akin to HeapSort, which is also $O(N)=N \log N$. The original
> name came from a failed experiment to implement what amounts to a 2D MergeSort.

</div>
<div style="flex: 1; padding: 10px;">

![alt text](img/paper2/image-14.png) 

</div>
</div>

---

## Scaling Comparison

<div style="display: flex;">
<div style="flex: 2; padding: 10px;">

![alt text](img/paper2/image-15.png) 

</div>
</div>


> For a 4096 element dataset, 124 seconds vs 2.56 seconds


---

## Sorting Algorithm details

<div style="display: flex;">
<div style="flex: 2; padding: 10px; max-height: 1000px; overflow-y: auto;">

![alt text](img/vat_prim_mst_block_diagram_v2.svg)

</div>
</div>

---

## TSP Optimization

<div style="display: flex;">
<div style="flex: 1; padding: 10px;">

![alt text](img/paper2/image-17.png)
</div>
<div style="flex: 1; padding: 10px;">

![alt text](img/paper2/image-16.png)
</div>
<div style="flex: 1; padding: 10px;">

![alt text](img/paper2/image-18.png)
</div>
</div>

> Unfortunately, the commonly used 2-OPT local optimization method breaks the cluster organization

---

## The Second Insight -- Memory

* VAT often caches the entire dissimilarity matrix $D$
* This doubles the memory consumption to save on compute costs, but since mergeVAT scales so much better, we need to
  reduce memory consumption
    * Why not compute only the requested distance $D_{i,j}$ as needed?
    * This reduces memory to one copy of $D$ plus working space, approximately $O(N) = {{N^2+N}\over{16}}$
      vs $O(N)=2N^2$

---

## The Third Insight - Loop-Walking

<div style="display: flex;">
<div style="flex: 1; padding: 10px;">

* VAT sequence, paired with the original sequence, creates a collection of loops: _directed, cyclic graphs_
* We can start at any point on any loop, and follow the loop until we reach our starting point again.
* If we mask which loop entries have been visited, we can simply increment until we find another loop

</div>
<div style="flex: 1; padding: 10px;">

![alt text](img/paper2/image-19.png)

</div>
</div>

---

## Conclusions and Future Work

* mergeVAT:
    * Expands the usable size from 5K to 130K+ elements
    * Provides a good initial guess for TSP applications
    * Loop-walking cuts the memory requirement in half
* **Active** Work: Identify VAT-clusters to change 2-Opt check points
* Future Work: Distributed mergeVAT for 500K elements
* Future Work: InsertionSort for building up to 500K elements
