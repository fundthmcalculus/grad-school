# Proposal Defense Slides

This directory contains the scaffolding for your graduate research proposal defense presentation.

## Directory Structure

```
proposal-defense/
├── slides/          # Individual markdown slide files
├── scripts/         # Build scripts for generating presentations
├── output/          # Generated HTML and PDF artifacts
└── README.md        # This file
```

## Getting Started

### 1. Edit Sample Slides

Replace the sample content in `slides/` with your actual proposal defense slides:

- `01-title.md` - Title slide with presentation overview
- `02-motivation.md` - Research motivation and background
- `03-problem-statement.md` - Problem statement and research questions
- `04-methodology.md` - Proposed research methodology
- `05-timeline.md` - Timeline and expected outcomes

### 2. Add More Slides

To add additional slides:
1. Create a new markdown file in `slides/` with a numeric prefix (e.g., `06-results.md`)
2. Files are combined in alphabetical order, so numbering determines slide order

### 3. Build Locally

Run the build script to generate HTML and PDF presentations:

```bash
python3 proposal-defense/scripts/build.py
```

This generates:
- `proposal-defense/output/proposal-defense.html` - Interactive reveal.js presentation
- `proposal-defense/output/proposal-defense.pdf` - PDF slide deck

### 4. Automatic Build on Merge

When you push to `main`, the GitHub Actions workflow automatically:
- Combines all markdown slides
- Builds an interactive HTML presentation
- Generates a PDF for printing/sharing
- Uploads artifacts as GitHub Actions artifacts

## Requirements

To build locally, install:

```bash
# Ubuntu/Debian
sudo apt-get install pandoc texlive-latex-base texlive-latex-extra

# macOS
brew install pandoc basictex
```

## Markdown Syntax for Slides

Each slide is separated by `---` on its own line. Standard markdown syntax is supported:

```markdown
# Slide Title

## Subtitle or Section

- Bullet point 1
- Bullet point 2

---

# Next Slide
```

## Viewing Presentations

- **HTML:** Open `proposal-defense/output/proposal-defense.html` in a web browser
  - Navigate with arrow keys or click arrows
  - Press `?` for keyboard shortcuts
  - Press `s` for speaker view

- **PDF:** Open in any PDF viewer

## Customization

Edit `.github/workflows/proposal-defense.yml` to customize:
- Build triggers (when to automatically build)
- LaTeX theme (currently: Madrid)
- Reveal.js theme (currently: black)
- Output formats

Edit `proposal-defense/scripts/build.py` to:
- Change slide ordering
- Add new output formats
- Modify pandoc options
