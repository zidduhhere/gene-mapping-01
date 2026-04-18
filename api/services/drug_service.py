"""Drug service — lookup known drugs and generate fingerprints."""

import os

import numpy as np
import pandas as pd


class DrugService:
    def __init__(self):
        self.drug_list = []
        self.drug_fingerprints = {}
        self._loaded = False

    def load(self):
        """Load drug names and fingerprints from GDSC2 data."""
        project_root = os.path.join(os.path.dirname(__file__), "..", "..")

        # Data may be in data/GDSC2 or results/data/GDSC2
        candidates = [
            os.path.join(project_root, "data", "GDSC2"),
            os.path.join(project_root, "results", "data", "GDSC2"),
        ]
        data_path = None
        for c in candidates:
            if os.path.isdir(c):
                data_path = c
                break

        if data_path is None:
            print("WARNING: No GDSC2 data directory found")
            self._loaded = True
            return

        # Load drug names (columns: pubchem_id, drug_name)
        names_file = os.path.join(data_path, "drug_names.csv")
        if os.path.exists(names_file):
            df_names = pd.read_csv(names_file, dtype=str)
            for _, row in df_names.iterrows():
                drug_id = str(row.iloc[0]).strip()
                drug_name = str(row.iloc[1]).strip() if len(row) > 1 else drug_id
                self.drug_list.append({"id": drug_id, "name": drug_name})

        # Load fingerprints (2048-bit Morgan fingerprints)
        # Format: 2048 rows (bits) x N columns (drug IDs) — each column is a drug
        fp_file = os.path.join(data_path, "drug_fingerprints", "pubchem_id_to_demorgan_2048_map.csv")
        if os.path.exists(fp_file):
            df_fp = pd.read_csv(fp_file)
            # Columns are drug IDs, rows are fingerprint bits
            for col in df_fp.columns:
                drug_id = str(col).strip()
                fp_values = df_fp[col].values.astype(np.float32)
                self.drug_fingerprints[drug_id] = fp_values

        self._loaded = True
        print(f"Loaded {len(self.drug_list)} drugs, {len(self.drug_fingerprints)} fingerprints")

    def get_drug_list(self) -> list[dict]:
        if not self._loaded:
            self.load()
        return self.drug_list

    def get_fingerprint(self, drug_id: str) -> np.ndarray | None:
        """Get fingerprint for a known drug by ID."""
        if not self._loaded:
            self.load()
        return self.drug_fingerprints.get(drug_id.strip())

    def generate_fingerprint_from_smiles(self, smiles: str) -> np.ndarray | None:
        """Generate 2048-bit Morgan fingerprint from a SMILES string."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            return np.array(fp, dtype=np.float32)
        except ImportError:
            return None

    def get_fingerprint_for_request(self, drug_id: str | None, smiles: str | None) -> np.ndarray | None:
        """Get fingerprint from drug_id or SMILES, whichever is provided."""
        if drug_id:
            fp = self.get_fingerprint(drug_id)
            if fp is not None:
                return fp

        if smiles:
            return self.generate_fingerprint_from_smiles(smiles)

        return None

    def get_fingerprint_dim(self) -> int:
        """Return the dimensionality of fingerprints."""
        if self.drug_fingerprints:
            return len(next(iter(self.drug_fingerprints.values())))
        return 2048


drug_service = DrugService()
