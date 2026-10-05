# Backwards Construction of Fuzzy Inference Systems
Scott Phillips (University of Cincinnati)
Vladik Kreinovich PhD (University of Texas - El Paso)
Kelly Cohen PhD (University of Cincinnati)

### Introduction
Large-scale fuzzy inference system (FIS) construction is of significant interest and utility in the explainable and certifiable AI communities. At the scale of dozens or hundreds of inputs, traditional approaches to FIS construction and tuning show performance limitations. The time to construct and tune the dozens of membership functions and potentially thousands of rules also limit the ease of resulting model interpretability. In this paper, we propose that an uncommon reversed approach provides a far superior starting point. By construction, it produces a model with the provably minimal number of resultant rules. In addition, the resulting model can be easily fine tuned by existing methods such as genetic algorithms (GA) or gradient descent (GD).

By focusing on TSK-formulation problems with gaussian membership functions and probability-formulation t-norm/t-conorm operators, additional optimizations can be realized for the consequent construction. The uniform continuity and differentiability of this style of model provides the ability to globally optimize the consequent functions in regression models.

### Background
The common steps for Fuzzy Inference System (FIS) construction are as follows:
1. Define membership functions for each input
2. Define rule base for each output utilizing `AND` rules
3. Optimize the resulting system by varied approaches
There are several obvious limitations with this methodology which have been discussed in the literature. The most obvious of which is _rulebase explosion_, wherein the number of required resultant rules scales with the power of the number of inputs:
$$N_rules = \prod_1^{N_{input}} N_{mf,i} \approx N_\mu ^ {N_{input}}$$
Assuming that the number of membership functions is equal for all inputs. While this is often not the case, it still illustrates the scale of the problem. Existing methods have been designed to address this (eg ANFIS), but these can introduce limitations of the interpretability of the model. The ANFIS approach also introduces another parameter to tune: the number of resultant rules.

There exists in the literature mention **TODO citations** of consequent-first model, but these are limited and clustering-based. This approach still constrains the model construction. We will leverage a slightly different approach to also provide an anomaly-detection score which can be used to identify if a sample matches to the known model input domains. This is important for the certifiable AI community, since it provides a confidence-/familiarity-score which can be used to determine the trustworthiness of the model output, or if the decision needs to be escalated for further human review.

### Our Approach - Classification
We propose a shift in the construction to an output-first clustering. The clearest example is for single- or multi-label classification problems, so we will start there. The key distinction from previous approaches is to switch away from holistic per-output clustering to per-variable clustering. This provides additional input controls at the cost of some possible unsupported admissibility which will be address later.
1. For each output class, select all training samples that belong to that output class.
2. For each input variable, compute the membership functions using Gaussian Mixture Modeling (GMM) to identify not only the number of clusters, but also the exact parameters for this collection of membership functions. The weight parameter is discarded so that the resultant membership function follows the common form.
3. These per-input membership functions are then `t-conorm`d together because they all belong to the same output class.
4. Once all membership functions for each variable are computed for a given output class, `t-norm` the variables together as is common practice.
5. An optional $O(N)=N^2$ post-processing step is to deduplicate the repeated membership functions since the membership functions for each variable are computed individually and independently for each output label. This is not required, but it does increase interpretability, as well as return the model to a more common form.

**Advantages**
1. Fixed number of rules: $N_{output}=N_{class}$ thereby _eliminating_ all possible rulebase explosion. Because of this construction method, it is not possible to simplify the FIS to have fewer rules.
2. Individual membership extraction: By extracting memberships and utilizing `t-conorm` to combine them, the model can easily handle disjoint membership in a given input variable.
3. Gaussian membership: The gaussian membership, due to its infinite support, continuity, and differentiability, provides additional convenience in optimization. For classification, this is less important, but it will be extremely useful in regression later.

**Drawbacks**
1. Because the admissibility of any rule is defined as the cartesian product of the input variable regions, it is possible to construct a region that will be admitted without training support.
2. Membership function and t-norm/t-conorm pairs are fixed, which reduces flexibility. **TODO: Future work to show that we can shift from this model to a Ruspini partition**
3. Training data requirement. Unlike a forward training approach, which only requires a fitness/scoring function, this approach _requires_ a large amount of sample data to partition. For instance, it cannot be used to tune a fuzzy controller.

### Our Approach - Feature Engineering
For large scale problems with numerous variables, it is important to reduce the number contributing input variables as much as possible. This not only reduces the model size, but also increases the model interpretability. Various methods can be applied, but the method we have chosen is as follows:
1. For each output class (from above):
2. Compute the gaussian correlation between a given input variable and the output class using a combination of `wasserstein` and `bhattacharyya`.
  1. This is done because `bhattacharya` is parametric and works for approximately gaussian data while `wasserstein` is nonparametric.
  2. The composition is done by averaging the arithmetic _and_ geometric mean of the two metrics. This provides a stronger test that an input is correlated to the output.
3. Sort the variables by output correlation in descending order
4. Take up to the top `n` variables, skipping those input variables which have a Pearson correlation coefficient less than a defined threshold (usually $<0.85$).

This allows input feature reduction while preserving output discrimination.

### Our Approach - Anomaly Detection
Anomaly detection is specifically important in certifiable AI. If the model is making a decision, the model needs to also have clear and accurate repoerted confidence in the answer. Triangular/trapezoidal membership functions solve this by having finite support. Gaussian membership functions have infinite support, and will therefore cause the output rules to fire, albeit at a low level, even in regions where no training input data exists. For example, with the **TODO: PhiURII dataset** phishing dataset, it is important to know not only if the sample is valid or malicious, but also if model has seen a sample like that before. We propose an anomaly detection rule construction as follows.

$$ \mu_{anomaly} = complement(tconorm(\forall rules)) + boost$$

This approach provides a general definition for anomaly detection independent of the choice of `t-norm`/`t-conorm` pair provided that the choice is a **De Morgan Triplet**. The $boost$ parameter is a sensitivity parameter that can be used to adjust the threshold for anomaly detection. It is formulated this way so that the anomaly rule can be directly compared with the classification rules. The anomaly sensitivity parameter can be tuned independently of model training, but our experience **TODO evidence**) indicates a value $0.95 \in [0.9,0.99]$ is a good starting point. Because anomaly is a strict classification, the degree of membership can exceed unity. This is not a problem, since defuzzification for classification is just an `argmax` operator. This same anomaly rule can be used with regression to identify if the test data is well outside the training set.

### Our Approach - Regression
Similar to classification, regression is a natural extension by assigning a cardinal order to the output labels. To select the output labels, an additional parameter is set, the number of output bins: $N_{output}$. The range of each bin can be selected by multiple different means, but our testing has indicated that a simple uniform binning is sufficient. Quantile binning is prone to bias error by over-paritioning common outputs. Once the output bins have been selected, bin labels are assigned to each variable. After the bin labels have been assigned, the same approach to membership function selection and combination applies to this method.

As mentioned earlier, the approach is designed around a Takagi Sugeno Kang (TSK) formulation using gaussian memberships and probability `t-norm`/`t-cnorm` pairs. This formulation means that the output consequent is continuous and differentiable in all rules everywhere. For regression in TSK models, the defuzzification formula (a weighted average) is as follows:
$$ output_{defuzzified} = {{\sum_{1}^{N_{rules}} z_i \times w_i}\over{\sum_{1}^{N_{rules}} w_i}}$$
where $z_i$ is the crisp consequent rule and $w_i$ is the firing-weight of the $i$th rule.

Since this is a type-1 model, the consequent rules are typically polynomials. These polynomials can be fit using common least-squares techniques. Since the firing weights $w_i$ are already known, the defuzzification step can be expanded to be linear in the consequent terms.

### Derivation
For simplicity, define $W_{all}=\sum w_i$. From there, define $z_i = a_0 + a_1 x_1 + ... a_n x_n$ for linear (TSK order-1) consequent equations. Define $y$ is the crisp output variable, $a_{rule-num,coeff-num}$ is the coefficient of the $n$th crisp consequent rule and $x_i$ is the $i$th crisp input variable. The other polynomial orders follow naturally. The consequent equations then become as follows:

$$ y = {1 \over W_{sum}} \left ( w_1 (a_{0,0}+a_{0,1}x_1+...+a_{0,N}x_N) + w_2 (a_{1,0}+a_{1,1}x_1+...+a_{1,N}x_N)+... \right ) $$

Define normalized weights, with $x_0=1$ for convenience:
$$ W_{sum} = \sum_{j=1}^{R} w_j, \qquad \bar w_j = \frac{w_j}{W_{sum}}, \qquad x_0 \equiv 1 $$

Pull $1 / W_{sum}$ inside, write as double-sum
$$ y = \sum_{j=1}^{R} \bar w_j \sum_{i=0}^{N} a_{j-1,i}\, x_i $$

Group known terms ($\bar w_j w_i$) from unknowns ($a$)
$$ y = \sum_{j=1}^{R}\sum_{i=0}^{N} \left(\bar w_j\, x_i\right) a_{j-1,i} $$

Dot-product form for sample $m$
$$ y^{(m)} = {\boldsymbol\varphi^{(m)}}^{\!\top} \boldsymbol\theta,
\quad
\boldsymbol\varphi^{(m)} = \begin{bmatrix}
\bar w_1^{(m)} \\ \bar w_1^{(m)}x_1^{(m)} \\ \vdots \\ \bar w_R^{(m)}x_N^{(m)}
\end{bmatrix},
\quad
\boldsymbol\theta = \begin{bmatrix}
a_{0,0} \\ a_{0,1} \\ \vdots \\ a_{R-1,N}
\end{bmatrix}
$$

Stack $M$ samples (where $ M \gte R(N+1) $):
$$
\underbrace{\begin{bmatrix}
{\boldsymbol\varphi^{(1)}}^{\!\top} \\
{\boldsymbol\varphi^{(2)}}^{\!\top} \\
\vdots \\
{\boldsymbol\varphi^{(M)}}^{\!\top}
\end{bmatrix}}_{\Phi \in \mathbb{R}^{M \times R(N+1)}}
\boldsymbol\theta
=
\underbrace{\begin{bmatrix}
y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(M)}
\end{bmatrix}}_{\mathbf{y}}
##

Expand $\phi$ explicitly:
$$
\Phi =
\begin{bmatrix}
\bar w_1^{(1)} & \bar w_1^{(1)}x_1^{(1)} & \cdots & \bar w_1^{(1)}x_N^{(1)} & \bar w_2^{(1)} & \cdots & \bar w_R^{(1)}x_N^{(1)} \\
\bar w_1^{(2)} & \bar w_1^{(2)}x_1^{(2)} & \cdots & \bar w_1^{(2)}x_N^{(2)} & \bar w_2^{(2)} & \cdots & \bar w_R^{(2)}x_N^{(2)} \\
\vdots & \vdots & & \vdots & \vdots & & \vdots \\
\bar w_1^{(M)} & \bar w_1^{(M)}x_1^{(M)} & \cdots & \bar w_1^{(M)}x_N^{(M)} & \bar w_2^{(M)} & \cdots & \bar w_R^{(M)}x_N^{(M)}
\end{bmatrix}
$$

Least-squares solution (cost->gradient->normal equations)
$$
J(\boldsymbol\theta) = \lVert \Phi\boldsymbol\theta - \mathbf{y} \rVert_2^2
\;\;\Rightarrow\;\;
\nabla_{\boldsymbol\theta} J = 2\Phi^\top(\Phi\boldsymbol\theta - \mathbf{y}) = 0
\;\;\Rightarrow\;\;
\Phi^\top\Phi\,\boldsymbol\theta = \Phi^\top\mathbf{y}
$$
$$ \boldsymbol\theta = (\Phi^\top\Phi)^{-1}\Phi^\top\mathbf{y} = \Phi^{+}\mathbf{y} $$

Ridge variant (rank-deficient $\phi$)
$$ \boldsymbol\theta = (\Phi^\top\Phi + \lambda I)^{-1}\Phi^\top\mathbf{y} $$

As is clearly evident, this is a system of linear equations. The solution for the optimal consequent coefficients can be solved using any number of common numerical techniques. Our testing has found that regularization is helpful, since some of the training data can be numerically ill-conditioned. Pre-scaling all data to be approximately zero mean and unity variance or approximately $[0, 1]$ reduces this issue. This is not a specific limitation of this training method, but rather a limitation of FIS's in general. Other machine learning techniques also benefit from similar variable pre-conditioning.

### Key Results

**TODO: Show the example data results and timing**

### Conclusions and Future Work
This approach has substantial performance improvements for large-scale FIS construction. It creates an excellent initial guess for the model to be further refined by existing techniques while simultaneously eliminating rulebase explosion by construction. It does not sacrifice interpretability like ANFIS approaches in doing so.

Future work consists of extending this to handle Fuzzy Trees (**TODO: develop additional paper with Hugo**) as well as type-2 FISs. T2-FIS have achieved common use, and so would benefit from a similar performance enhancements. Fuzzy Trees are used in the engineering community, and would likewise benefit from faster construction and evaluation vs common Genetic Algorithm based approaches.

**TODO Show quantile bin issues**

**TODO Address admissibility consideration**


### Appendix

A _De Morgan Triplet_ is a triple $(\top,\perp\,\neg)$ where
1. $\top$ is a t-norm
2. $\neg$ is a strong negation operator
3. $\perp$ is a t-conorm such that $\perp(a,b) = \neg\top(a,b)$ such that $\perp(a,b) = 1-\top(1-a,1-b)$ 
