#!/bin/bash

# Full LaTeX compilation sequence for thesis with bibliography
echo "Step 1: First lualatex compilation..."
lualatex -synctex=1 -interaction=nonstopmode -file-line-error thesis.tex

echo "Step 2: Running biber..."
biber thesis

echo "Step 3: Second lualatex compilation..."
lualatex -synctex=1 -interaction=nonstopmode -file-line-error thesis.tex

echo "Step 4: Third lualatex compilation (final cross-references)..."
lualatex -synctex=1 -interaction=nonstopmode -file-line-error thesis.tex

echo "Compilation complete!"

# Keep PDF viewer at current page by touching the PDF file
# This prevents the viewer from jumping back to page 1
touch thesis.pdf