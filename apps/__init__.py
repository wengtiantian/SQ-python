from fastapi.routing import APIRouter

from .sq import sq_api

api_router = APIRouter()

api_router.include_router(sq_api, prefix="/sq")


__all__ = ['api_router']