from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
def train_model(df):
    x = df.drop(['price','mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus'], axis=1)

    y = df['price']
    X_train,X_test,y_train,y_test =train_test_split(x,y, test_size=0.25,random_state=60)
    model = LinearRegression()
    model.fit(X_train,y_train)
    with open('model/house_price_model.pkl',"wb") as f:
        pickle.dump(model, f)
    return model, X_test, y_test    
