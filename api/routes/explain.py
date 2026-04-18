"""SHAP explanation route."""

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException

from api.routes.auth import verify_token
from api.services.model_service import model_service
from api.services.drug_service import drug_service
from api.services.shap_service import shap_service

router = APIRouter()


@router.post("")
async def explain(
    file: UploadFile = File(...),
    drug_id: str = Form(default=None),
    smiles: str = Form(default=None),
    user: dict = Depends(verify_token),
):
    """Compute SHAP explanation for a prediction."""
    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not drug_id and not smiles:
        raise HTTPException(status_code=422, detail="Provide either drug_id or smiles")

    fingerprint = drug_service.get_fingerprint_for_request(drug_id, smiles)
    if fingerprint is None:
        raise HTTPException(status_code=422, detail="Could not get drug fingerprint")

    csv_bytes = await file.read()
    try:
        gene_expression = model_service.parse_gene_csv(csv_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error parsing CSV: {e}")

    # Ensure network is initialized by running a prediction first
    network = model_service.get_network()
    if network is None:
        model_service.predict(gene_expression, fingerprint)
        network = model_service.get_network()

    if network is None:
        raise HTTPException(status_code=503, detail="Model network not available")

    gene_names = model_service.get_landmark_genes()

    try:
        result = shap_service.explain(
            gene_expression=gene_expression,
            fingerprint=fingerprint,
            network=network,
            gene_names=gene_names,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP explanation failed: {e}")

    return result
