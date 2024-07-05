from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse

from PIL import Image, ImageOps
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models.utils.detect_face import extract_face
import torch

import pandas as pd
import xgboost as xgb
import io
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from typing import Annotated

from pydantic import BaseModel
from database import engine, SessionLocal
from sqlalchemy.orm import Session
import models
from models import FaceData

app = FastAPI()
models.Base.metadata.create_all(bind = engine)


# Initialize global variables
base_data = {}
base_data_df = None
base_data['label'] = []
j = 0

# initiate others
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, thresholds=[0.6, 0.7, 0.9], device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize XGBoost model
model = xgb.XGBClassifier(
    max_depth=5, learning_rate=0.01, n_estimators=100, gamma=0,
    min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005, n_estimator=1000
)

knn = KNeighborsClassifier(n_neighbors=3)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
db_dependency = Annotated[Session, Depends(get_db)]

class FaceDataBase(BaseModel):

    label : str
    
    feature_0 : float
    feature_1 : float
    feature_2 : float
    feature_3 : float
    feature_4 : float
    feature_5 : float
    feature_6 : float
    feature_7 : float
    feature_8 : float
    feature_9 : float
    feature_10 : float
    feature_11 : float
    feature_12 : float
    feature_13 : float
    feature_14 : float
    feature_15 : float
    feature_16 : float
    feature_17 : float
    feature_18 : float
    feature_19 : float
    feature_20 : float
    feature_21 : float
    feature_22 : float
    feature_23 : float
    feature_24 : float
    feature_25 : float
    feature_26 : float
    feature_27 : float
    feature_28 : float
    feature_29 : float
    feature_30 : float
    feature_31 : float
    feature_32 : float
    feature_33 : float
    feature_34 : float
    feature_35 : float
    feature_36 : float
    feature_37 : float
    feature_38 : float
    feature_39 : float
    feature_40 : float
    feature_41 : float
    feature_42 : float
    feature_43 : float
    feature_44 : float
    feature_45 : float
    feature_46 : float
    feature_47 : float
    feature_48 : float
    feature_49 : float
    feature_50 : float
    feature_51 : float
    feature_52 : float
    feature_53 : float
    feature_54 : float
    feature_55 : float
    feature_56 : float
    feature_57 : float
    feature_58 : float
    feature_59 : float
    feature_60 : float
    feature_61 : float
    feature_62 : float
    feature_63 : float
    feature_64 : float
    feature_65 : float
    feature_66 : float
    feature_67 : float
    feature_68 : float
    feature_69 : float
    feature_70 : float
    feature_71 : float
    feature_72 : float
    feature_73 : float
    feature_74 : float
    feature_75 : float
    feature_76 : float
    feature_77 : float
    feature_78 : float
    feature_79 : float
    feature_80 : float
    feature_81 : float
    feature_82 : float
    feature_83 : float
    feature_84 : float
    feature_85 : float
    feature_86 : float
    feature_87 : float
    feature_88 : float
    feature_89 : float
    feature_90 : float
    feature_91 : float
    feature_92 : float
    feature_93 : float
    feature_94 : float
    feature_95 : float
    feature_96 : float
    feature_97 : float
    feature_98 : float
    feature_99 : float
    feature_10: float
    feature_101 : float
    feature_102 : float
    feature_103 : float
    feature_104 : float
    feature_105 : float
    feature_106 : float
    feature_107 : float
    feature_108 : float
    feature_109 : float
    feature_110 : float
    feature_111 : float
    feature_112 : float
    feature_113 : float
    feature_114 : float
    feature_115 : float
    feature_116 : float
    feature_117 : float
    feature_118 : float
    feature_119 : float
    feature_120 : float
    feature_121 : float
    feature_122 : float
    feature_123 : float
    feature_124 : float
    feature_125 : float
    feature_126 : float
    feature_127 : float
    feature_128 : float
    feature_129 : float
    feature_130 : float
    feature_131 : float
    feature_132 : float
    feature_133 : float
    feature_134 : float
    feature_135 : float
    feature_136 : float
    feature_137 : float
    feature_138 : float
    feature_139 : float
    feature_140 : float
    feature_141 : float
    feature_142 : float
    feature_143 : float
    feature_144 : float
    feature_145 : float
    feature_146 : float
    feature_147 : float
    feature_148 : float
    feature_149 : float
    feature_150 : float
    feature_151 : float
    feature_152 : float
    feature_153 : float
    feature_154 : float
    feature_155 : float
    feature_156 : float
    feature_157 : float
    feature_158 : float
    feature_159 : float
    feature_160 : float
    feature_161 : float
    feature_162 : float
    feature_163 : float
    feature_164 : float
    feature_165 : float
    feature_166 : float
    feature_167 : float
    feature_168 : float
    feature_169 : float
    feature_170 : float
    feature_171 : float
    feature_172 : float
    feature_173 : float
    feature_174 : float
    feature_175 : float
    feature_176 : float
    feature_177 : float
    feature_178 : float
    feature_179 : float
    feature_180 : float
    feature_181 : float
    feature_182 : float
    feature_183 : float
    feature_184 : float
    feature_185 : float
    feature_186 : float
    feature_187 : float
    feature_188 : float
    feature_189 : float
    feature_190 : float
    feature_191 : float
    feature_192 : float
    feature_193 : float
    feature_194 : float
    feature_195 : float
    feature_196 : float
    feature_197 : float
    feature_198 : float
    feature_199 : float
    feature_200 : float
    feature_201 : float
    feature_202 : float
    feature_203 : float
    feature_204 : float
    feature_205 : float
    feature_206 : float
    feature_207 : float
    feature_208 : float
    feature_209 : float
    feature_210 : float
    feature_211 : float
    feature_212 : float
    feature_213 : float
    feature_214 : float
    feature_215 : float
    feature_216 : float
    feature_217 : float
    feature_218 : float
    feature_219 : float
    feature_220 : float
    feature_221 : float
    feature_222 : float
    feature_223 : float
    feature_224 : float
    feature_225 : float
    feature_226 : float
    feature_227 : float
    feature_228 : float
    feature_229 : float
    feature_230 : float
    feature_231 : float
    feature_232 : float
    feature_233 : float
    feature_234 : float
    feature_235 : float
    feature_236 : float
    feature_237 : float
    feature_238 : float
    feature_239 : float
    feature_240 : float
    feature_241 : float
    feature_242 : float
    feature_243 : float
    feature_244 : float
    feature_245 : float
    feature_246 : float
    feature_247 : float
    feature_248 : float
    feature_249 : float
    feature_250 : float
    feature_251 : float
    feature_252 : float
    feature_253 : float
    feature_254 : float
    feature_255 : float
    feature_256 : float
    feature_257 : float
    feature_258 : float
    feature_259 : float
    feature_260 : float
    feature_261 : float
    feature_262 : float
    feature_263 : float
    feature_264 : float
    feature_265 : float
    feature_266 : float
    feature_267 : float
    feature_268 : float
    feature_269 : float
    feature_270 : float
    feature_271 : float
    feature_272 : float
    feature_273 : float
    feature_274 : float
    feature_275 : float
    feature_276 : float
    feature_277 : float
    feature_278 : float
    feature_279 : float
    feature_280 : float
    feature_281 : float
    feature_282 : float
    feature_283 : float
    feature_284 : float
    feature_285 : float
    feature_286 : float
    feature_287 : float
    feature_288 : float
    feature_289 : float
    feature_290 : float
    feature_291 : float
    feature_292 : float
    feature_293 : float
    feature_294 : float
    feature_295 : float
    feature_296 : float
    feature_297 : float
    feature_298 : float
    feature_299 : float
    feature_300 : float
    feature_301 : float
    feature_302 : float
    feature_303 : float
    feature_304 : float
    feature_305 : float
    feature_306 : float
    feature_307 : float
    feature_308 : float
    feature_309 : float
    feature_310 : float
    feature_311 : float
    feature_312 : float
    feature_313 : float
    feature_314 : float
    feature_315 : float
    feature_316 : float
    feature_317 : float
    feature_318 : float
    feature_319 : float
    feature_320 : float
    feature_321 : float
    feature_322 : float
    feature_323 : float
    feature_324 : float
    feature_325 : float
    feature_326 : float
    feature_327 : float
    feature_328 : float
    feature_329 : float
    feature_330 : float
    feature_331 : float
    feature_332 : float
    feature_333 : float
    feature_334 : float
    feature_335 : float
    feature_336 : float
    feature_337 : float
    feature_338 : float
    feature_339 : float
    feature_340 : float
    feature_341 : float
    feature_342 : float
    feature_343 : float
    feature_344 : float
    feature_345 : float
    feature_346 : float
    feature_347 : float
    feature_348 : float
    feature_349 : float
    feature_350 : float
    feature_351 : float
    feature_352 : float
    feature_353 : float
    feature_354 : float
    feature_355 : float
    feature_356 : float
    feature_357 : float
    feature_358 : float
    feature_359 : float
    feature_360 : float
    feature_361 : float
    feature_362 : float
    feature_363 : float
    feature_364 : float
    feature_365 : float
    feature_366 : float
    feature_367 : float
    feature_368 : float
    feature_369 : float
    feature_370 : float
    feature_371 : float
    feature_372 : float
    feature_373 : float
    feature_374 : float
    feature_375 : float
    feature_376 : float
    feature_377 : float
    feature_378 : float
    feature_379 : float
    feature_380 : float
    feature_381 : float
    feature_382 : float
    feature_383 : float
    feature_384 : float
    feature_385 : float
    feature_386 : float
    feature_387 : float
    feature_388 : float
    feature_389 : float
    feature_390 : float
    feature_391 : float
    feature_392 : float
    feature_393 : float
    feature_394 : float
    feature_395 : float
    feature_396 : float
    feature_397 : float
    feature_398 : float
    feature_399 : float
    feature_400 : float
    feature_401 : float
    feature_402 : float
    feature_403 : float
    feature_404 : float
    feature_405 : float
    feature_406 : float
    feature_407 : float
    feature_408 : float
    feature_409 : float
    feature_410 : float
    feature_411 : float
    feature_412 : float
    feature_413 : float
    feature_414 : float
    feature_415 : float
    feature_416 : float
    feature_417 : float
    feature_418 : float
    feature_419 : float
    feature_420 : float
    feature_421 : float
    feature_422 : float
    feature_423 : float
    feature_424 : float
    feature_425 : float
    feature_426 : float
    feature_427 : float
    feature_428 : float
    feature_429 : float
    feature_430 : float
    feature_431 : float
    feature_432 : float
    feature_433 : float
    feature_434 : float
    feature_435 : float
    feature_436 : float
    feature_437 : float
    feature_438 : float
    feature_439 : float
    feature_440 : float
    feature_441 : float
    feature_442 : float
    feature_443 : float
    feature_444 : float
    feature_445 : float
    feature_446 : float
    feature_447 : float
    feature_448 : float
    feature_449 : float
    feature_450 : float
    feature_451 : float
    feature_452 : float
    feature_453 : float
    feature_454 : float
    feature_455 : float
    feature_456 : float
    feature_457 : float
    feature_458 : float
    feature_459 : float
    feature_460 : float
    feature_461 : float
    feature_462 : float
    feature_463 : float
    feature_464 : float
    feature_465 : float
    feature_466 : float
    feature_467 : float
    feature_468 : float
    feature_469 : float
    feature_470 : float
    feature_471 : float
    feature_472 : float
    feature_473 : float
    feature_474 : float
    feature_475 : float
    feature_476 : float
    feature_477 : float
    feature_478 : float
    feature_479 : float
    feature_480 : float
    feature_481 : float
    feature_482 : float
    feature_483 : float
    feature_484 : float
    feature_485 : float
    feature_486 : float
    feature_487 : float
    feature_488 : float
    feature_489 : float
    feature_490 : float
    feature_491 : float
    feature_492 : float
    feature_493 : float
    feature_494 : float
    feature_495 : float
    feature_496 : float
    feature_497 : float
    feature_498 : float
    feature_499 : float
    feature_500 : float
    feature_501 : float
    feature_502 : float
    feature_503 : float
    feature_504 : float
    feature_505 : float
    feature_506 : float
    feature_507 : float
    feature_508 : float
    feature_509 : float
    feature_510 : float
    feature_511 : float
    feature_512 : float



# Function to detect faces and extract features
def detect_and_extract_features(image: Image.Image):
    try:
        boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
        if landmarks is None:
            return None
        faces = torch.stack([extract_face(image, bb) for bb in boxes])
        embeddings = facenet(faces).detach().numpy()
        return embeddings
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting faces: {str(e)}")

# Function to preprocess images and extract features
def preprocess_maker(image: Image.Image, tag: str, db: db_dependency):
    global base_data_df, j
    
    # Flip and rotate images
    flipped_image = ImageOps.mirror(image)
    img_90 = image.rotate(90)
    img_180 = image.rotate(180)
    img_270 = image.rotate(270)
    
    list_preprocessed = [image, flipped_image, img_90, img_180, img_270]
    for pre_image in (list_preprocessed):
        try:
            landmarks = detect_and_extract_features(pre_image)
            if landmarks is None:
                continue
            
            landmarks_list = [landmark.tolist() for landmark in landmarks]
            for i, landmark in enumerate(landmarks_list[0]):
                if f'feature_{i}' not in base_data:
                    base_data[f'feature_{i}'] = []
                base_data[f'feature_{i}'].append(landmark)      
                              
            base_data['label'].append(tag)
            
            
            if base_data_df is None:
                base_data_df = pd.DataFrame(base_data).drop_duplicates()
                len_ori = len(base_data_df)
            else:
                temp_df = pd.DataFrame(base_data).drop_duplicates()   
                base_data_df = pd.concat([temp_df]) 
                len_ori = 0
            
            features = {f'feature_{i}': float(landmark) for i, landmark in enumerate(landmarks_list[0])}
            features['label'] = tag
            j = j +1+len_ori
            features['id'] = j
            print( j+len(base_data_df))
            face_data = models.FaceData(**features)
            db.add(face_data)
            db.commit()
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
    
    print(base_data_df)
    return base_data_df
def load_data_from_db(db: Session):
    try:

        face_data_records = db.query(FaceData).all()
        data_dict = [record.__dict__ for record in face_data_records]
        base_data_df = pd.DataFrame(data_dict)
        if '_sa_instance_state' in base_data_df.columns:
            base_data_df = base_data_df.drop(columns=['_sa_instance_state'])
        return base_data_df
    except Exception as e:
        print(f"Error loading data from the database: {e}")
        return pd.DataFrame()

#### FastAPI ENDPOINTS #######


@app.on_event('startup')
def startup_event():
    global base_data_df
    db = SessionLocal()
    try:
        base_data_df = load_data_from_db(db)
        #   always train the model
        try:
            train_model()
        except Exception:
            pass
        print("Data loaded successfully on startup.")
    except Exception as e:
        print(f"Error during startup data load: {e}")
    finally:
        db.close()

@app.get('/')
def root():
    return {'hello': 'world'}

#   delete face
@app.delete('/api/face')
def delete_by_face(label: str, db: Session = Depends(get_db)):
    global base_data_df
    try:
        records_to_delete = db.query(models.FaceData).filter(models.FaceData.label == label).all()
        base_data_df = base_data_df[base_data_df['label'] != label].copy()
        
        if records_to_delete:
            for record in records_to_delete:
                db.delete(record)
            db.commit()
            return {"message": f"Deleted {len(records_to_delete)} records with label '{label}'"}
        else:
            raise HTTPException(status_code=404, detail=f"No records found with label '{label}'")
    except Exception as e:
        db.rollback()  
        raise HTTPException(status_code=500, detail=f"Error deleting records: {str(e)}")

#   show all faces
@app.get('/api/face')  
def show_faces():
    global base_data_df
    list_face = list(base_data_df['label'].unique())    
    print(list_face)
    return list_face

#   recognize + input data inside the db
@app.post('/api/face/register')
def register(path: str, tag: str, db: Session = Depends(get_db)):
    global base_data_df
    
    img = Image.open(path).convert('RGB')
    base_data_df = preprocess_maker(img, tag, db)
    
    #   always train the model
    try:
        train_model()
    except Exception:
        pass
    return {"message": "Images registered and processed."}

@app.post('/train')
def train_model():
    global base_data_df
    print(base_data_df)
    if base_data_df is None or base_data_df.empty:
        raise HTTPException(status_code=400, detail="No data available to train the model.")
    
    features = base_data_df[[column for column in base_data_df.columns if column not in ['label', 'id']]]
    try:
        knn.fit(np.array(features), np.array(base_data_df['label']))
        print("Model trained successfully.")
        return {"message": "Model trained successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@app.post('/api/face/recognize')
def predict(path: str):
    try:

        img = Image.open(path).convert('RGB')
        landmarks = detect_and_extract_features(img)
        if landmarks is None or len(landmarks) == 0:
            raise HTTPException(status_code=400, detail="No faces detected")
        prediction = knn.predict(landmarks)
        predicted_label = prediction[0]
        
        return JSONResponse(content={"name": predicted_label})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
