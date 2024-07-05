# Knowledge Test

end-to-end project made to do face recognition using features extracted from a picture with KNN (due to data limitation), fastAPI, and postgreSQL by Syahrul Maulana Wijaya.

this project consisting of 3 files with main.py is the most important file where I put most of the work there.

# preparation
first prepare the postgreSQL server and the database will be used under the file databse.py

```
DATABASE_URL = "postgresql://postgres:maulana@localhost:5432/pydb"
```

#  How To Run
there are 4 main call to use this program

first, need to run the app
```python
uvicorn main:app --reload
```
when running this, the app will be initiated and will automatically pick the data inside the database in postgreSQL


