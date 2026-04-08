import os
from sqlalchemy.orm import Session 
from src.models.orm_models import DimImage
from src.models.schemas import DimImageSchema
from src.models.instances import get_engine

BASE_DIR = "utils/data/humans"
engine = get_engine()

def load_images_from_directory(root_dir: str, engine):
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

                    # Build Pydantic schema
                    img_schema = DimImageSchema(
                        image_path=full_path,
                        label=label
                    )

                    # Convert to ORM row
                    img_row = DimImage(**img_schema.model_dump())
                    session.add(img_row)
        session.commit()
        
# Run file to add all images of /data directory to din_image table of DB
if __name__ == "__main__":
    try: 
        load_images_from_directory(BASE_DIR, engine)
        print(f"Images have been added successfully to DB!") # TODO: change this to a logger
    except:
        raise ConnectionError("Unable to add images to dim_images table in DB.")

