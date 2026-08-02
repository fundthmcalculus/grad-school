# Bibliography (working — to be consolidated into BibTeX)

**Status:** Outline · aggregated from repo docs. Existing partial `.bib`: `tribble-fis/tribble-tree/hfis_review.bib` (35 entries). Consolidate all into one `references.bib` for the final document.
**Action:** verify every DOI / page number before submission (flagged repeatedly in repo self-reviews). Drop the ungrounded "pqVAT six-orders-of-magnitude" web claim. Fix Zhang-2023 author attribution.

---

## Fuzzy Inference Systems
- Takagi & Sugeno 1985 (TSK); Jang 1993 (ANFIS); Wang & Mendel 1992; Sugeno & Yasukawa 1993 (output-first); Chiu 1994; Jang & Sun 1993 (RBF↔TSK); Wu et al. 2020 (TSK optimization taxonomy / MBGD-RDA); Wang 1998/1999, Joo & Lee 2002 (universal approximation); Ruspini 1969; Cordón et al. 2001 (genetic fuzzy systems); Herrera 2008; Alcalá et al. 2007.

## VAT / iVAT / cluster tendency
- Bezdek & Hathaway 2002 (VAT); Wang, Nguyen, Bezdek et al. 2010 (iVAT / aVAT); Havens & Bezdek 2012 (fast O(n²) iVAT); Wang, Leckie et al. 2009/2010 (SpecVAT / DBE); Kumar et al. 2013/2016 (clusiVAT); Kumar & Bezdek 2020 (VAT survey); Rathore et al. 2020 (ConiVAT); Hathaway/Huband 2006 (sVAT/bigVAT).

## Fast / GPU VAT (competitors to differentiate)
- Meng & Yuan 2018 (eVAT — exact GPU VAT); Avinash & Lachheb 2025 (Fast-VAT); *Information Sciences* 2024 (BB-VAT / kdT-VAT / TkdT-VAT); Vineet et al. 2009 (GPU Borůvka); Cate & Twigg 1977, Catanzaro 2014 (in-place permutation); Jin et al. (DiSC distributed single-linkage).

## MST / single-linkage / persistence / TDA
- Prim 1957; Gower & Ross 1969 (MST≡single-linkage); Zahn 1971 (inconsistent-edge cut); Tibshirani et al. 2001 (gap statistic); Chazal et al. 2013 (ToMATo); Bonis & Oudot 2014/2018 (beta-plateau — nearest precedent for Ch 5); AuToMATo (arXiv:2408.06958, bottleneck-bootstrap).

## Fuzzy C-Means / relational fuzzy
- Dunn 1973; Bezdek 1981; Stetco et al. 2015 (FCM++); Arthur & Vassilvitskii 2007 (k-means++); Hathaway & Bezdek 1994 (NERFCM); Kaufman & Rousseeuw (FANNY); Bien & Tibshirani 2011 (minimax-linkage prototypes); Chehreghani 2019/2020 (minimax embedding); Xing et al. (metric learning).

## Fuzzy trees / hierarchy / HME
- Janikow 1998 (fuzzy ID3); Yuan & Shaw 1995 (ambiguity); Suárez & Lutsko 1999; Olaru & Wehenkel 2003 (soft splits); Medina-Chico et al. 2001 (soft CART + linear leaves — key competitor); Fumanal-Idocin et al. 2025 (arXiv:2512.11616); Jordan & Jacobs 1994 (HME + EM); Raju et al. 1991 (hierarchical fuzzy); Zhou-Chung-Wang 2017 (D-TSK-FC); Zhang et al. 2023 (Information Fusion survey — FIX attribution); Nanfack et al. 2022; de Oliveira 1999; Guillaume & Charnomordic 2004/2006/2011 (FisPro); Magdalena 2018 (hierarchy≠interpretability — mandatory rebuttal); Higashi & Klir 1983 (nonspecificity); Cavalin & Oliveira 2019, Viola & Jones 2001 (cascades).

## Optimization / TSP / quality-diversity
- Lin & Kernighan 1973; Helsgaun 2000/2009 (LKH); Croes 1958 (2-opt); Johnson & McGeoch 1997; Lenstra 1974 (seriation≡TSP); Climer & Zhang 2006; Hahsler 2008 (seriation); Chisman 1975 (CTSP); Guttmann-Beck et al. 2000 (endpoint stitch); Ding 2007; Dorigo et al. 1996/1997 (ACO/ACS); Dai, Ji & Liu 2009 (MST-seeded ACO); Taillard & Helsgaun 2019 (POPMUSIC); Vassiliades et al. 2018 (CVT-MAP-Elites); Mouret & Clune 2015 (MAP-Elites); Deb et al. 2002 (NSGA-II); Zitzler et al. 2001 (SPEA2).

## Interpretability / XAI (secondary framing)
- Lundberg & Lee 2017 (SHAP); Molnar (interpretable ML) — optional.
- Pickering 2024 & Arnett 2018: **structural/template references only** (same lab, proposal format). The author's methods and interpretability framing were developed independently, without knowledge of Pickering's work — cite only as parallel work if at all, never as an intellectual source.

---

### Open items
- Merge `hfis_review.bib` + these into one `references.bib`.
- Verify DOIs / pages / the Wang et al. TKDE 2009/2010 pairing.
- Add exact NAFIPS self-citations once venues confirmed (Ch 9).
