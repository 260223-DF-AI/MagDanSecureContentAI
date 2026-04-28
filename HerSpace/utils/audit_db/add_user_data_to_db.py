import os
import json
from sqlalchemy.orm import Session 
from src.models.orm_models import DimDescription, DimPost, DimUser
from src.models.schemas import DimDescriptionSchema, DimPostSchema, DimUserSchema
from src.models.instances import get_engine
from utils.audit_db.add_img_dataset_to_db import check_img_exists_in_db

BASE_DIR = "utils/users"
engine = get_engine()

# NOTE: Script is not checking for duplicate descriptions, users, or posts. User data is crafted specifically for demo purposes and future users and posts are not expected.
def load_users_from_directory(root_dir: str, engine: Engine) -> None: # type: ignore
    """
    Walks the dataset directory and inserts images into dim_image.
    """
    with Session(engine) as session:
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            
            try:
                username = folder
                img_ids = []
                desc_ids = []
                
                for filename in os.listdir(folder_path):
                    # add user images to dim_images
                    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        full_path = os.path.join(folder_path, filename)
                        img_row = check_img_exists_in_db(full_path, None, session) # if img isn't in db, add it
                        if not filename.startswith("pfp"):
                            img_ids.append(img_row.image_id)
                    
                    # add descriptions to dim_descriptions
                    if filename == 'post_descriptions.json':
                        with open(os.path.join(folder_path, filename)) as f:
                            description_dict = json.load(f)
                        
                        for key, value in description_dict.items():
                            # Create Pydantic schema
                            desc_schema = DimDescriptionSchema(
                                text=str(value)
                            )

                            # Convert to ORM
                            desc_row = DimDescription(**desc_schema.model_dump())
                            session.add(desc_row)
                            session.flush() 
                            desc_ids.append(desc_row.description_id)
                            
                # add user to dim_user
                user_schema = DimUserSchema(
                    username=username,
                    num_of_posts=len(img_ids),
                    num_of_violations=0
                )
                
                user_row = DimUser(**user_schema.model_dump())
                session.add(user_row)
                session.flush()
                user_id = user_row.user_id
                
                # add posts to dim_posts table - using user, imgs, and decs FKs
                for i in range(len(img_ids)):
                    post_schema = DimPostSchema(
                        status=None, # None meaning, post content still needs to be reviewed
                        user_key=user_id,
                        image_key=img_ids[i],
                        description_key=desc_ids[i]
                    )
                    
                    post_row = DimPost(**post_schema.model_dump())
                    session.add(post_row)
                session.commit()
            except:
                session.rollback()
                raise

# Run file to add all data in /users directory to din_image table of DB
if __name__ == "__main__":
    try: 
        load_users_from_directory(BASE_DIR, engine)
        print(f"User have been added successfully to DB!")
    except Exception as e:
        raise ConnectionError(f"Unable to add user data from /users in DB: {e}")
