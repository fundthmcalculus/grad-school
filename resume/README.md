# Resume / CV

`resume.md` is the editable Markdown source for a résumé or longer CV. It is a
starter template: replace all bracketed fields with your own verified content;
no résumé or CV was supplied when this folder was created.

## Build a PDF

From the repository root, run:

```bash
python resume/build_pdf.py
```

The script also works when invoked by path from another working directory. It
writes `outputs/resume.pdf` beside this README and keeps LaTeX intermediates in
a temporary directory that is removed after the build. A failed build leaves an
existing PDF untouched.

The build requires Pandoc and one of `xelatex`, `lualatex`, `pdflatex`, or
`tectonic`. The script can also use Pandoc bundled by an installed `pypandoc`
package. It does not download or install build dependencies. Unicode résumé
content is best supported by `xelatex` or `lualatex`.

Generated files in `outputs/` are ignored by Git; commit the Markdown source,
not the regenerable PDF. Before committing, review the document for personal
information: this repository may be public, so include contact details only if
you intend to publish them.