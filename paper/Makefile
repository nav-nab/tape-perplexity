# Build the paper:  make            (needs a TeX Live installation with latexmk)
# Rebuild figures:  make figures    (needs the outputs of `python run.py`)
all: main.pdf

main.pdf: main.tex refs.bib $(wildcard generated/*.tex) $(wildcard figures/*.pdf)
	latexmk -pdf -interaction=nonstopmode main.tex

figures:
	python make_figures.py

clean:
	latexmk -C
