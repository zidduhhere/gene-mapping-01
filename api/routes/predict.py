"""Prediction route."""

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException

from api.routes.auth import verify_token
from api.services.model_service import model_service
from api.services.drug_service import drug_service

router = APIRouter()


@router.post("")
async def predict(
    file: UploadFile = File(...),
    drug_id: str = Form(default=None),
    smiles: str = Form(default=None),
    user: dict = Depends(verify_token),
):
    """Predict drug response from uploaded gene expression CSV."""
    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not drug_id and not smiles:
        raise HTTPException(status_code=422, detail="Provide either drug_id or smiles")

    # Get drug fingerprint
    fingerprint = drug_service.get_fingerprint_for_request(drug_id, smiles)
    if fingerprint is None:
        if smiles:
            raise HTTPException(
                status_code=422,
                detail="Could not generate fingerprint from SMILES. Install rdkit or check the SMILES string.",
            )
        raise HTTPException(status_code=404, detail=f"Drug '{drug_id}' not found")

    # Parse gene expression CSV
    csv_bytes = await file.read()
    try:
        gene_expression = model_service.parse_gene_csv(csv_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error parsing CSV: {e}")

    # Run prediction
    result = model_service.predict(gene_expression, fingerprint)
    return result
