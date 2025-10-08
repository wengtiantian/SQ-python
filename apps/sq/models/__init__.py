from db.base_class import Base
from db.session import engine
from .sq import sqModel


__all__ = ['sqModel']


Base.metadata.create_all(engine)