def Stacking_Ensemble(models, X_train, y_train, X_test, n_folds):
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    
    Stacking_train = np.zeros((X_train.shape[0], len(models)))
    Stacking_test = np.zeros((X_test.shape[0], len(models)))

    Strat_KFold = model_selection.StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 42)
    skf = list(Strat_KFold.split(X_train, y_train))

    ### Loop all the Base Models
    for model_counter, model in enumerate(models):
        print("Model_Counter" + str(model_counter))
        S_test_temp = np.zeros((titanic_test_data_X.shape[0], n_folds))
        
        ### Loop across the Folds
        for fold_counter, (tr_index, te_index) in enumerate(skf):
            print("Fold : " + str(fold_counter))
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            
            model.fit(X_tr, y_tr)
            print("----------------------")
            print("Model : " + str(model_counter) + "Fold : " + str(fold_counter) + "Score is - " + str(model.score(X_tr, y_tr)))
            print("----------------------")
            Stacking_train[te_index, model_counter] = model.predict(X_te)
            
            S_test_temp[:, fold_counter] = model.predict(X_test)
            
        Stacking_test[:, model_counter] = st.mode(S_test_temp, axis = 1)[0].ravel()
    
    return (Stacking_train, Stacking_test)
