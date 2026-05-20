from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))