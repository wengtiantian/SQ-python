

from sqlalchemy import Column, Integer, BigInteger, Float,Text

from db.base_class import Base


class sqModel(Base):
    __tablename__ = "es_reviews"
    user_id = Column(BigInteger, nullable=False, comment="发表评论的用户ID")
    servicer_id = Column(BigInteger, nullable=False, comment="被评价的服务主体ID")
    content = Column(Text, nullable=True, comment="评价内容")
    rating = Column(Integer, nullable=False, comment="评分（1-5星）")
    SQdim = Column(Integer, nullable=True, comment="这条评论所属的服务质量维度：有形性、可靠性、响应性")
    SenScore = Column(Float, nullable=True, comment="该评论的情感评分")