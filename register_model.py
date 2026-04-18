"""Register TabTransformer into drevalpy's MODEL_FACTORY.

Import this module before running experiments to make TabTransformer
available alongside built-in models.
"""

import os
import sys

# Add project root to path so models package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.TabTransformer.tab_transformer import TabTransformer

# Register into drevalpy's factories
from drevalpy.models import MODEL_FACTORY, MULTI_DRUG_MODEL_FACTORY

MODEL_FACTORY["TabTransformer"] = TabTransformer
MULTI_DRUG_MODEL_FACTORY["TabTransformer"] = TabTransformer

print(f"Registered TabTransformer into MODEL_FACTORY. Available models: {len(MODEL_FACTORY)}")
