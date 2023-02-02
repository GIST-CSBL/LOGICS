
"""
This is the code implementation for:
LOGICS: Learning optimal generative distribution for designing de novo chemical structures

The implementation of generative LSTM is hugely influenced by REINVENT,
https://github.com/MarcusOlivecrona/REINVENT

For comments and bug reports, please send an email to bsbae402@gmail.com.
"""

## Here, we disable the logging option of RDKit.
## If you don't disable the logging of the rdkit,
## you will see tons of error and warning messages when you try to MolFromSmiles().
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')