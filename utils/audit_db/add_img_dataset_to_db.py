import os
from sqlalchemy.orm import Session 
from src.models.orm_models import DimImage
from src.models.schemas import DimImageSchema
from src.models.instances import get_engine

BASE_DIR = "utils/data/humans"
engine = get_engine()

def load_images_from_directory(root_dir: str, engine: Engine) -> None: # type: ignore
    """
    Walks the dataset directory and inserts images into dim_image.
    """
    with Session(engine) as session:
        for split in ["training", "test"]:
            split_path = os.path.join(root_dir, split)

            for label in os.listdir(split_path):
                label_dir = os.path.join(split_path, label)

                if not os.path.isdir(label_dir):
                    continue

                for filename in os.listdir(label_dir):
                    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue

                    full_path = os.path.join(label_dir, filename)

                    check_img_exists_in_db(full_path, label, session)
        session.commit()

def preload_existing_images(engine: Engine) -> set: # type: ignore
    """Load all image paths from the DB into a fast lookup set."""
    with Session(engine) as session:
        rows = session.query(DimImage.image_path).all()
        return {row[0] for row in rows}

def check_img_exists_in_db(file_path: str, name: str, session: Session) -> DimImage:
    """Helper method to verify if an image exists in the dim _image table of DB.

    Args:
        file_path (str): file_path to image
        name (str): class/label of image
        session (Session): orm session for DML

    Raises:
        ValueError: if img_row not found or created raise error

    Returns:
        DimImage: the corresponding table row for the provided file_path
    """
    img_row = None
    existing_imgs = preload_existing_images(engine)
    
    if file_path not in existing_imgs:
        # Create Pydantic schema
        img_schema = DimImageSchema(
            image_path=file_path,
            label=str(name)
        )

        # Convert to ORM
        img_row = DimImage(**img_schema.model_dump())
        session.add(img_row)
        session.commit()
        session.refresh(img_row)

        # Add to in-memory set
        existing_imgs.add(file_path)
        print(f"New image added to DB: {file_path}") #TODO: logger

    else:
        # Fetch existing image row
        img_row = session.query(DimImage).filter_by(image_path=file_path).first()
    
    if not img_row: raise ValueError(f"An entry was not found and could not be made for: {file_path}")
    
    return img_row

# Run file to add all images of /data directory to din_image table of DB
if __name__ == "__main__":
    try: 
        load_images_from_directory(BASE_DIR, engine)
        print(f"Images have been added successfully to DB!") # TODO: change this to a logger
    except:
        raise ConnectionError("Unable to add images to dim_images table in DB.")
