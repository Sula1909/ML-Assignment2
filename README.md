# ML-Assignment2: IEEE-CIS Fraud Detection

## პროექტის სათაური

Kaggle IEEE-CIS Fraud Detection 

## Kaggle-ის კონკურსის მოკლე მიმოხილვა

კონკურსის მიზანია თითოეული ტრანზაქციისთვის `isFraud` კლასის, ანუ სიყალბის პროგნოზირება. ამოცანაში მოცემული დატა არის საკმაოდ დაუბალანსებელი, რადგან ცოტაა ყალბი ტრანზაქციების პროცენტულობა და ამ competition-ის მთავარი მეტრიკა არის ROC-AUC.  

## ჩემი მიდგომა პრობლემის გადასაჭრელად

მთავარი იდეა იყო, რომ დაუბალანსებელ დატასთან მის შესაბამისად მემუშავა და არ ამერჩია ისეთი მოდელები, რომლებსაც მსგავს დატასთან მუშაობა უჭირს. ვცადე 5 სხვადასხვა მოდელის არქიტექტურა, XGBoost, RandomForest, DecisionTree, CatBoost და LightGBM.



## რეპოზიტორიის სტრუქტურა

ML-Assignment2/
├── model-experiment-randomforest.ipynb
├── model_experiment_DecisionTree.ipynb
├── model_experiment_CatBoost.ipynb
├── model-experiment-xgboost.ipynb
├── model_experiment_LGBM.ipynb
└── README.md


## ყველა ფაილის განმარტება

- `model-experiment-randomforest.ipynb` — pipeline RandomForest მოდელის pipeline.
- `model_experiment_DecisionTree.ipynb` — Decision Tree pipeline (baseline + tuning).
- `model_experiment_CatBoost.ipynb` — CatBoost pipeline 
- `model-experiment-xgboost.ipynb` — XGBoost pipeline 
- `model_experiment_LGBM.ipynb` — LightGBM pipeline
- `README.md` — პროექტის დოკუმენტაცია.

## FEATURE ENGINEERING

რაც გავლილი გვაქვს, თითქმის ყველაფერი ვცადე. ერთადერთი არ ვცადე RFE, რადგან ამხელა დატაზე წარმოუდგენელი იქნებოდა RFE-ს გამოყენება.

Feature engineering და preprocessing pipeline :

- Train დატაში id სვეტების: `-` → `_`.
- Memory reduction (`reduce_mem_usage`) downcast-ებით.
- outlier removal: `TransactionAmt > 30000` მოცილება (მხოლოდ train-ში).
 

- დროითი features: `day`, `hour`, `hour_alertFeature`.
- რელაციური feature: `TransactionAmt_to_mean_card1`.

## კატეგორიული ცვლადების რიცხვითში გადაყვანა

გამოყენებული encoding მიდგომები:

- **WOE & IV** — WOE mapping-ები ითვლება მხოლოდ train-ზე target-ის გამოყენებით.
- **One-Hot Encoding** — დაბალი უნიკალური მნიშვნელობის სვეტებისთვის (`nunique <= 5`), რათა კატეგორიული ცვლადები ისე მოვიშოროთ, რომ დატაც ძალიან არ გავზარდოთ. 
- **Frequency Encoding** — მაღალი სიხშირის კატეგორიული სვეტებისთვის.
- **Label Encoding** — მაღალი სიხშირის სვეტებისთვის (`nunique > 5`), რომლებზე OHE-ც ძალიან გაგვიდიდებდა.

## NAN მნიშვნელობების დამუშავება

- RandomForest / Decision Tree / CatBoost pipeline-ებში გამოიყენება median imputation (train-ზე ნასწავლი მედიანებით).
- XGBoost / LightGBM pipeline-ებში გამოიყენება constant fill: `-999`.
- ყველა imputer transformer ინახავს fit-ზე ნასწავლ პარამეტრებს და transform-ზე კონსისტენტურად იყენებს.

## CLEANING მიდგომები

- Raw data merge: `train_transaction` + `train_identity`.
- `TransactionID` ამოღებულია features-დან.
- ქრონოლოგიური split (`70%/15%/15%`) `TransactionDT`-ს მიხედვით.
- dtype optimization და feature-name sanitization (LightGBM compatibility).

## FEATURE SELECTION

ამ Assignment-ში გამოყენებულია შემდეგი feature selection / importance-based მიდგომები:

- **Correlation Filter** — `C*` და `D*` ჯგუფებში მაღალი კორელაციის სვეტების მოცილება (`> 0.95`).
- **RFE-like შემცირება LGBM importance-ით** — `V*` ჯგუფში დაბალი მნიშვნელობის სვეტების მოცილება.
- **WOE & IV** — კატეგორიული სვეტების ინფორმაციულობის გაძლიერება target-aware encoding-ით.
- **SHAP** — მოდელის ინტერპრეტაციისთვის გამოყენებული ანალიზის მიდგომა.
- **One-Hot / Frequency / Label Encoding** — feature representation-ის გაუმჯობესება მოდელების მოთხოვნების მიხედვით.

## გამოყენებული მიდგომები და მათი შეფასება

- Pipeline-first მიდგომამ გაამარტივა ექსპერიმენტების გამეორება და მოდელებს შორის სამართლიანი შედარება.
- Chronological split-მა უფრო რეალისტური generalization სურათი აჩვენა, ვიდრე random split.
- ძლიერი recall-ის მიღწევა ხშირად იწვევდა precision-ის შემცირებას; საბოლოო არჩევანი გაკეთდა AUC/Recall ბალანსით.

## TRAINING

### ტესტირებული მოდელები

- RandomForest
- Decision Tree
- CatBoost
- XGBoost
- LightGBM

### Hyperparameter ოპტიმიზაციის მიდგომა

- iterative/manual tuning MLflow ექსპერიმენტებით;
- train/validation რეჟიმი tuning-ისას;
- test holdout-ის გამოყენება მხოლოდ საბოლოო pipeline შეფასებისთვის;
- overfit კონტროლი რეგულარიზაციით, depth/leaf შეზღუდვებით და sampling პარამეტრებით.

### საბოლოო მოდელის შერჩევის დასაბუთება

მიმდინარე ექსპერიმენტების მიხედვით ორი ყველაზე ძლიერი კანდიდატია:

- **CatBoost_Pipeline** — მაღალი Validation AUC და სტაბილური Test AUC.
- **XGBoost_Pipeline_Final** — ძლიერი AUC + უკეთესი precision/recall ბალანსი validation-ზე.

ბოლო LGBM რანმაც აჩვენა ძლიერი recall (≈0.70) შედარებით დაბალი overfit-ით (Train AUC ≈0.966 vs Validation AUC ≈0.913), რის გამოც LGBM საბოლოო submission-ის ძლიერი კანდიდატია.

## MLFLOW TRACKING

### MLflow ექსპერიმენტების ბმული

- [ML-Assignment2 MLflow (DagsHub)](https://dagshub.com/Sula1909/ML-Assignment2.mlflow)

### ჩაწერილი მეტრიკების აღწერა

ექსპერიმენტებში ლოგდება:

- `Train_AUC`, `Validation_AUC` (და final run-ებში `Test_AUC`)
- `Precision`, `Recall`, `F1`
- final run-ებისთვის `Test_Precision`, `Test_Recall`, `Test_F1`
- მოდელის ჰიპერპარამეტრები და pipeline-ის საკვანძო პარამეტრები

### საუკეთესო მოდელის შედეგები

ამჟამად საუკეთესო დაფიქსირებული შედეგები:

- **CatBoost_Pipeline**  
  - Train AUC: `0.974251`  
  - Validation AUC: `0.916905`  
  - Test AUC: `0.904117`  
  - Validation Recall: `0.690664`

- **XGBoost (strong tuned run)**  
  - Train AUC: `0.964933`  
  - Validation AUC: `0.916620`  
  - Validation Precision: `0.599346`  
  - Validation Recall: `0.481920`

- **LightGBM (latest regularized run before finalization)**  
  - Train AUC: `0.965964`  
  - Validation AUC: `0.913248`  
  - Validation Recall: `0.708416`
