help:	## Show this help and exit.
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-30s\033[0m %s\n", $$1, $$2}'


.PHONY: report
report:	## Compile the report into a PDF.
	@if ! command -v typst &> /dev/null; then \
		echo "Error: The Typst CLI is needed to compile the report locally, please install it from https://github.com/typst/typst?tab=readme-ov-file#installation"; \
		exit 1; \
	fi
	typst c -f pdf report/main.typ
	@echo "Report compiled successfully to ./report/main.pdf"

requirements-txt:	## Generate requirements.txt from uv lockfile.
	uv export --no-hashes --format requirements-txt > requirements.txt