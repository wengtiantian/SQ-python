

# try:
#     from redis.asyncio import Redis as asyncRedis
# except ImportError:
#     from aioredis import Redis as asyncRedis
from common.curd_base import CRUDBase
from ..models.sq import sqModel
class CURDsq(CRUDBase):
    def fun(self):
        pass

curd_sq=CURDsq(sqModel)