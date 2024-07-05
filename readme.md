# Knowledge Test

end-to-end project made to do face recognition using features extracted from a picture with KNN (due to data limitation), fastAPI, and postgreSQL by Syahrul Maulana Wijaya.

this project consisting of 3 files with main.py is the most important file where I put most of the work there.

# preparation
first prepare the postgreSQL server and the database will be used under the file [databse.py](https://github.com/jay76-a11y/AI_engineer_knowledgeTest/blob/main/database.py "Named link title")

```
DATABASE_URL = "postgresql://postgres:maulana@localhost:5432/pydb"
```

#  How To Run
there are 4 main call to use this program

##  Initiation
first, need to run the app
```python
uvicorn main:app --reload
```
when running this, the app will be initiated and will automatically pick the data inside the database in postgreSQL

### [GET] face
this function will pull all of the registered face on the database as shown in this example
```
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/face" -Method Get -ContentType "application/json"
```

### [POST] register
this function will register the face inputted, I also do a simple image augmentation to add datas on the same label

To register, provide the path of the picture and the label of the picture as shown in this example

```
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/face/register?path=jenna.jpg&tag=jenna" -Method Post -ContentType "application/json"
```

### [POST] recognize
this function will detect (predict) name of the person in the given picture with KNN model. the reason for using KNN is the limitation of the data available to train a better model, so for this small project, KNN is chosen
this call will only take Path of the picture as shown in this example
```
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/face/recognize?path=jenna.jpg" -Method Post -ContentType "application/json"
```

### [DELETE] face
this call will delete the given name on the database and will also retrain the model with the updated data.
the input for this call is the "label" or the name that need to be deleted from the database as shown in this example

```
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/face?label=jenna" -Method Delete -ContentType "application/json"
```




