%.pdf:%.tex
	pdflatex --shell-escape $<
	bibtex $(basename $<)
	pdflatex --shell-escape $<
	pdflatex --shell-escape $<


clean:
	rm -rf *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg *.brf *.idx *.ilg *.ind *.loa *fls *nav *snm *synctex.gz *fdb_latexmk _minted* *vrb
