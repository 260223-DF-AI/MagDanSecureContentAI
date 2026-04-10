# GroupLDD - Ramen Ratings Data Analysis
## Set-up Guide:
In terminal:
    python -m venv venv
    ./venv/Scripts/activate

    pip install -r requirements.txt

Create your .env  file:
1. add line ->   CS = "postgresql://postgres:password123@localhost:5432/magdan"
2. replace "password123" with your postgresql password
3. create the tables ->   run: python -m src.models.orm_models 
4. add imgset to db ->    run: python -m utils.audit_db.add_img_dataset_to_db
5. add user data to db -> run: python -m utils.audit_db.add_user_data_to_db (**not needed to train model**)
