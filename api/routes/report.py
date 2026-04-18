"""PDF report generation route."""

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from fastapi.responses import Response

from api.routes.auth import verify_token
from api.services.model_service import model_service
from api.services.drug_service import drug_service
from api.services.shap_service import shap_service
from api.services.pdf_service import generate_report_pdf

router = APIRouter()


@router.post("")
async def generate_report(
    file: UploadFile = File(...),
    drug_id: str = Form(default=None),
    drug_name: str = Form(default="Unknown Drug"),
    smiles: str = Form(default=None),
    user: dict = Depends(verify_token),
):
    """Generate a PDF clinical report with prediction and SHAP explanation."""
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

    # Prediction
    prediction = model_service.predict(gene_expression, fingerprint)
    if not prediction.get("predictions"):
        raise HTTPException(status_code=500, detail="Prediction returned no results")

    # SHAP explanation
    network = model_service.get_network()
    if network is None:
        raise HTTPException(status_code=503, detail="Model network not available after prediction")

    gene_names = model_service.get_landmark_genes()
    try:
        explanation = shap_service.explain(
            gene_expression=gene_expression,
            fingerprint=fingerprint,
            network=network,
            gene_names=gene_names,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP explanation failed: {e}")

    # Generate PDF
    try:
        pdf_bytes = generate_report_pdf(prediction, explanation, drug_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=pharmaai_report.pdf"},
    )
