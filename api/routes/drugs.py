"""Drug listing route."""

from fastapi import APIRouter, Depends

from api.routes.auth import verify_token
from api.services.drug_service import drug_service

router = APIRouter()


@router.get("")
async def list_drugs(user: dict = Depends(verify_token)):
    """Return list of available drugs."""
    drugs = drug_service.get_drug_list()
    if not drugs:
        drug_service.load()
        drugs = drug_service.get_drug_list()
    return {"drugs": drugs, "count": len(drugs)}
