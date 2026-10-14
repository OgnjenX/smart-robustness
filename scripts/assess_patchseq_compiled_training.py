"""Explicit compiled-table assessment; no automatic source route substitution."""

from pathlib import Path

from assess_patchseq_training import main

REG = Path('docs/validation-results/post2008-patchseq-compiled-training-assessment-registration-1461.yaml')

if __name__ == '__main__':
    main(REG)
