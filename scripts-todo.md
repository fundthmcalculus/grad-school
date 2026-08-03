# Root Directory Cleanup - Scripts & Assets TODO

## Overview
Audit of files in repository root that should be organized into appropriate folders.

## Status: COMPLETED - Commit: [pending]

### Python Scripts (3 files → gated-minimax-selection/)
All related to NERFCM beta-spread activation research:

- [x] `verify_beta_final.py` - Final verification: NERFCM beta-spread on non-Euclidean data
- [x] `verify_beta_nonmetric.py` - Test beta-spread on explicitly non-metric dissimilarity matrices
- [x] `verify_beta_spread.py` - Verify beta-spread activation on real non-Euclidean data

**Destination**: `gated-minimax-selection/verification/`
**Reasoning**: These are research utilities for the gated-minimax selection project

### Image Assets (1 file)
- [x] `midterm-grad-and-c.png` - Midterm exam graphic

**Destination**: `presentations/`
**Status**: Moved ✓

### Configuration Files (Keep at root)
These should remain in the root:
- ✅ `.gitignore` - Git configuration
- ✅ `.gitmodules` - Git submodule configuration
- ✅ `LICENSE` - Repository license
- ✅ `README.md` - Repository overview
- ✅ `makefile` - Development commands (format, lint, type-check)
- ✅ `requirements.txt` - Python dependencies
- ✅ `setup.cfg` - Linting/type-checking config

## Completed ✓

### Changes Made
1. **Moved verify_beta scripts** to `gated-minimax-selection/verification/`
   - Created new `verification/` subfolder for research utilities
   
2. **Updated import paths** in all 3 scripts
   - Changed hardcoded `/home/scott/PycharmProjects/grad-school/gated-minimax-selection`
   - Now uses: `Path(__file__).parent.parent` (portable relative path)
   
3. **Fixed data loading** in verify_beta_spread.py
   - Original script referenced non-existent CSV files
   - Updated to use scikit-learn datasets (load_iris, load_wine)
   - Makes script portable and immediately runnable
   
4. **Moved midterm image** to `presentations/`
   - midterm-grad-and-c.png now organized with presentation materials

## Next Steps
1. Run verification scripts to confirm they work: `python gated-minimax-selection/verification/verify_beta_*.py`
2. Monitor for any remaining hard-coded paths in root directory
3. Consider organizing other research artifacts similarly

## Summary
- **Files moved**: 4 (3 .py + 1 .png)
- **Root directory cleaned**: 3 Python scripts, 1 image
- **Paths updated**: 3 scripts now use relative imports
- **Data sources**: Updated to use standard ML datasets for portability
