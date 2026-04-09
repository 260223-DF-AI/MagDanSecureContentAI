from sqlalchemy import Column, String, Integer, Float, Boolean, ForeignKey, DateTime, func
from sqlalchemy.orm import declarative_base
from src.models.instances import get_engine

Base = declarative_base()
# TODO: 
# add grains to table, scd DocStrings

# =========================
# Dimension Tables
# =========================

class DimUser(Base):
    """
    Grain: Each row represents a unique user.
    SCD: Type 1 - Overwrites existing data when changes occur.
    """
    __tablename__ = "dim_user"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, nullable=False)
    num_of_posts = Column(Integer, nullable=False)
    num_of_violations = Column(Integer, nullable=False)


class DimImage(Base):
    """
    Grain: Each row represents a unique image.
    SCD: Type 1 - Overwrites existing data when changes occur.
    """
    __tablename__ = "dim_image"

    image_id = Column(Integer, primary_key=True, autoincrement=True)
    image_path = Column(String, nullable=False)
    label = Column(String, nullable=True)


class DimDescription(Base):
    """
    Grain: Each row represents a unique description.
    SCD: Type 1 - Overwrites existing data when changes occur.
    """
    __tablename__ = "dim_description"

    description_id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(String, nullable=False)
    is_safe_content = Column(Boolean, nullable=True)


class DimPost(Base):
    """
    Grain: Each row represents a unique post.
    SCD: Type 1 - Overwrites existing data when changes occur.
    """
    __tablename__ = "dim_post"

    post_id = Column(Integer, primary_key=True, autoincrement=True)
    status = Column(String, nullable=True)

    # Foreign keys
    user_key = Column(Integer, ForeignKey("dim_user.user_id"), nullable=False)
    image_key = Column(Integer, ForeignKey("dim_image.image_id"), nullable=False)
    description_key = Column(Integer, ForeignKey("dim_description.description_id"), nullable=False)


# =========================
# Training / Model Output Tables
# =========================

class CNNTraining(Base):
    """
    Grain: Each row represents one image classified by the CNN_model in a training session.
    """
    __tablename__ = "cnn_training"

    cnn_train_id = Column(Integer, primary_key=True, autoincrement=True)
    confidence_score = Column(Float, nullable=False)
    predicted_class = Column(String, nullable=False)
    is_correct = Column(Boolean, nullable=True)

    # FK to image
    run_key = Column(Integer, ForeignKey("cnn_training_runs.run_id"), nullable=False)
    image_key = Column(Integer, ForeignKey("dim_image.image_id"), nullable=False)


class LLMTraining(Base):
    """
    Grain: Each row represents one description processed by the LLM model in a training session.
    """
    __tablename__ = "llm_training"

    llm_train_id = Column(Integer, primary_key=True, autoincrement=True)
    output = Column(String, nullable=False)
    is_correct = Column(Boolean, nullable=True)
    accuracy = Column(Float, nullable=True)

    # FK to description
    description_key = Column(Integer, ForeignKey("dim_description.description_id"), nullable=False)
    
class CNNTrainingRun(Base):
    """
    Grain: Each row represents a unique training run of the CNN model.
    """
    __tablename__ = "cnn_training_runs"

    run_id = Column(Integer, primary_key=True, autoincrement=True)
    started_at  = Column(DateTime, server_default=func.now())

class FinalModelLog(Base):
    """
    Grain: Each row represents the output of the final model for a given post.
    """
    __tablename__ = "final_model_logs"

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    model_output = Column(String, nullable=False)
    is_correct_class = Column(Boolean, nullable=True)
    is_correct_prompt = Column(Boolean, nullable=True)
    is_post_allowed = Column(Boolean, nullable=False)
    policy_check_accuracy = Column(Float, nullable=True)

    # FK to post
    post_key = Column(Integer, ForeignKey("dim_post.post_id"), nullable=False)

# Run file to create all tables in DB
if __name__ == "__main__":
    engine = get_engine()
    Base.metadata.create_all(engine) # adds all tables to db
    print(f"Created tables in database: {Base.metadata.tables.keys()}") # TODO: change this to a logger
