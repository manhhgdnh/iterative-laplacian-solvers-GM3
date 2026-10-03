.PHONY: install run figures test clean

install:
	python3 -m pip install -r requirements.txt

run:
	python3 main.py --dim 2 --method pcg

figures:
	python3 scripts/generate_figures.py

test:
	python3 -m unittest discover -s tests -v

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ scripts/__pycache__
	rm -f results/vtk/*.vtk
