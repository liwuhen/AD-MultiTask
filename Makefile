# Makefile for Datachain project

# Default target
help:
	@echo "Available commands:"
	@echo "  lint           - Run linting checks"
	@echo "  clean          - Clean up generated files"

# Code quality
lint:
	pre-commit run --all-files

# Cleanup
clean:
	rm -rf build_nv_x86/
	rm -rf install_nvidia/
