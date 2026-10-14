"""Explicit compiled-table input; selected comparison still uses original engines."""

from pathlib import Path

from verify_patchseq_selected_fit import main

REG = Path('docs/validation-results/post2008-patchseq-selected-compiled-table-registration-1459.yaml')

if __name__ == '__main__':
    main(REG)
