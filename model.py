from sklearn.model_selection import train_test_split
from  sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
df = pd.read_csv("/kaggle/input/titanic/train.csv")
df = df.copy()

num_tf = SimpleImputer(strategy="median", add_indicator = True)





target = "Survived"

ids = df["PassengerId"].copy()

X = df.drop(columns= [target,"PassengerId","Name"])

y = df[target]

cat_cols = [col for col in X if X[col].dtypes=="object"]
print(cat_cols)
num_cols = X.select_dtypes(include=["number"]).columns.tolist()

cat_tf= Pipeline ([
    ("imp", SimpleImputer(strategy="most_frequent",add_indicator= True)),
    ("ohe",OneHotEncoder(
        handle_unknown="ignore",
        drop ="if_binary",
        min_frequency = 0.01,
        sparse_output =False
    ))
])

pre = ColumnTransformer([
    ("num",num_tf,num_cols),
    ("cat",cat_tf,cat_cols),
    
])


pipe  = Pipeline([
    ("pre",pre),
    ("clf",XGBClassifier(n_estimators = 2 , max_depth =2, learning_rate = 1))])


X_train , X_valid , y_train, y_valid = train_test_split(X,y,test_size= .2)

pipe.fit(X_train,y_train)

y_pred_tr=  pipe.predict(X_train)
y_pred_v = pipe.predict(X_valid)

# accuracies (ratio + count)
print(f"Train acc: {accuracy_score(y_train, y_pred_tr):.3f} "
      f"({accuracy_score(y_train, y_pred_tr, normalize=False)}/{len(y_train)})")

print(f"Valid acc: {accuracy_score(y_valid, y_pred_v):.3f} "
      f"({accuracy_score(y_valid, y_pred_v, normalize=False)}/{len(y_valid)})")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

id = test["PassengerId"].copy()
X_test= test.drop(columns=["Name"])

# y_test= test[target]

test_pred  = pipe.predict(X_test).astype(int)


sub = pd.DataFrame({
    "PassengerId":test["PassengerId"].values,
    "Survived":test_pred
})
print("a")
sub.to_csv("/kaggle/working/submission.csv",index=False)
