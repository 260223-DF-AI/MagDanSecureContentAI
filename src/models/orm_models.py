from sqlalchemy import Column, String, Integer, Float, Boolean, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base
from dotenv import load_dotenv
import os

Base = declarative_base()

# =========================
# Dimension Tables
# =========================

class DimUser(Base):
    __tablename__ = "dim_user"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, nullable=False)
    num_of_posts = Column(Integer, nullable=False)
    num_of_violations = Column(Integer, nullable=False)


class DimImage(Base):
    __tablename__ = "dim_image"

    image_id = Column(Integer, primary_key=True, autoincrement=True)
    image_path = Column(String, nullable=False)
    correct_cat = Column(String, nullable=True)


class DimDescription(Base):
    __tablename__ = "dim_description"

    description_id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(String, nullable=False)
    is_safe_content = Column(Boolean, nullable=False)


class DimPost(Base):
    __tablename__ = "dim_post"

    post_id = Column(Integer, primary_key=True, autoincrement=True)
    status = Column(String, nullable=False)

    # Foreign keys
    user_key = Column(Integer, ForeignKey("dim_user.user_id"), nullable=False)
    image_key = Column(Integer, ForeignKey("dim_image.image_id"), nullable=False)
    description_key = Column(Integer, ForeignKey("dim_description.description_id"), nullable=False)


# =========================
# Training / Model Output Tables
# =========================

class CNNTraining(Base):
    __tablename__ = "cnn_training"

    cnn_train_id = Column(Integer, primary_key=True, autoincrement=True)
    confidence_score = Column(Float, nullable=False)
    classification_cat = Column(String, nullable=False)
    is_correct = Column(Boolean, nullable=True)
    accuracy = Column(Float, nullable=True)

    # FK to image
    image_key = Column(Integer, ForeignKey("dim_image.image_id"), nullable=False)


class LLMTraining(Base):
    __tablename__ = "llm_training"

    llm_train_id = Column(Integer, primary_key=True, autoincrement=True)
    output = Column(String, nullable=False)
    is_correct = Column(Boolean, nullable=True)
    accuracy = Column(Float, nullable=True)

    # FK to description
    description_key = Column(Integer, ForeignKey("dim_description.description_id"), nullable=False)


class FinalModelLog(Base):
    __tablename__ = "final_model_logs"

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    model_output = Column(String, nullable=False)
    is_correct_class = Column(Boolean, nullable=True)
    is_correct_prompt = Column(Boolean, nullable=True)
    is_post_allowed = Column(Boolean, nullable=False)
    policy_check_accuracy = Column(Float, nullable=True)

    # FK to post
    post_key = Column(Integer, ForeignKey("dim_post.post_id"), nullable=False)

def create_tables_in_db():
    # gather connection string
    load_dotenv()
    CS = os.getenv("CS")
    
    engine = create_engine(CS)
    Base.metadata.create_all(engine) # adds all tables to db

if __name__ == "__main__":
    create_tables_in_db()
    print(f"Created tables in database: {Base.metadata.tables.keys()}") # TODO: change this to a logger
