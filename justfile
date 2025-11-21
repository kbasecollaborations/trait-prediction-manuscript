# Watch and auto-compile, continuing past errors
watch:
    latexmk -pdf -pvc -f

# Clean build artifacts first, then watch
watch-clean:
    latexmk -C
    latexmk -pdf -pvc -f

# Force clean including build directory
clean:
    latexmk -C
    rm -rf build/
