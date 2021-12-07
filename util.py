# /Users/sekhar/Documents/python_programs/ML/pandas_notes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def cat_to_cat_plot(df, 
                    dependent_var=None, 
                    independent_var=None, 
                    normalize=True, 
                    figsize=(8,6),
                    title=None
                    ):
    '''
    This function will plot a bar chart
    It accepts the following inputs:
    df: A pd.DataFrame object
    dependent_var: dependent categorical variable or target variable
    independent_var: independent categorical variable
    normalize: if True, will normalize the values (or gets proportions)
    figsize: figure size. Default is (8,6)
    title: The title of the graph

    Logic:
    -----
    1. Create another data frame with just dependent_var and independent_var columns
    2. Add another column called new_dummy_col with values of 1
    3. Pivot the data frame with index as the dependent var and columns as the independent var
    4  While pivoting use the aggregate function as sum or count
    5. If normalize is set to True, then get the sums of columns for each row, 
       and divide the row values by the corresponding columns sum.
    6. Then plot the brachart   
    '''
    # Check the presence of columns

    first_call = 0

    if dependent_var in df.columns and independent_var in df.columns:
        if title == None:
            title = f"{independent_var} vs {dependent_var} plot"        
        # Create a new data frame with just dependent and independent cols
        df_viz = df.loc[:, [dependent_var, independent_var]]            
        # Create a dummy col with a value of 1
        df_viz['new_dummy_col'] = 1
        # Pivot the new data frame with index as the dependent variables
        df_viz = df_viz.pivot_table(index=dependent_var, columns=independent_var, aggfunc="sum")

        # Rename the columns of the pivoted data frame.
        df_viz.columns = df_viz.columns.levels[1]

        # perform normalization, if set
        if normalize:
            for i in range(len(df_viz)):
                df_viz.iloc[i] = df_viz.iloc[i]/df_viz.iloc[i].sum()
        # Plot the graph        
        df_viz.plot.bar(figsize=figsize)
    # If independent var is set to None, then just plot the dependent var    
    elif independent_var == None and dependent_var!= None:
        # Just plot the dependent var 
        df[dependent_var].value_counts(normalize=normalize).plot.bar(legend=True, figsize=figsize)

    # Set the title of the plot    
    if title:
        plt.title(title)
    plt.show()
    plt.close()    


def perform_cat_vs_cat_analysis(df, var_1, var_2):
    if var_1 in df.columns and var_2 in df.columns:
        #if is_string_dtype(df[var_1]) and is_string_dtype(df[var_2]):
        cat_to_cat_plot(df, 
                    dependent_var=var_1, 
                    independent_var=var_2, 
                    normalize=True, 
                    figsize=(8,6),
                    title=f"Probability of {var_2} given {var_1}"
                    )

        cat_to_cat_plot(df, 
                    dependent_var=var_2, 
                    independent_var=var_1, 
                    normalize=True, 
                    figsize=(8,6),
                    title=f"Probability of {var_1} given {var_2}"            
                    )
        # print("Summary:")
        # var_1_nulls_perc = (df[var_1].isnull())/len(df)*100
        # var_2_nulls_perc = (df[var_2].isnull())/len(df)*100
        # print(f'\nThe variable "{var_1}" contains {var_1_nulls_perc}\% of nulls')
        # print(f'\nThe variable "{var_2}" contains {var_2_nulls_perc}\% of nulls')
    else:
        print("Check the column names. The given column names are not present in the data frame")   


def get_cat_col_details(df):
    line = "-" * 80
    cat_cols = df.select_dtypes(object).columns
    print(f"The data frame contains {df.shape[0]} rows and {df.shape[1]} rows.")
    print(f"Out of {df.shape[1]} rows, {len(cat_cols)} are categorical columns." )
    print("Here is the summary of categorical columns:")
    for col in cat_cols:
        print(line)
        print(f'Summary of "{col}":\n')
        unique_levels = len(set(df[col]))
        nulls_perc = round(sum(df[col].isnull())/len(df)*100, 4)
        print(f'The categorical variable: {col} contains {unique_levels} distinct values')
        print(f'It contains {nulls_perc}% of nulls')
        dupe_perc = sum(df[col].duplicated())/len(df)*100
        print(f'It contains {dupe_perc}% of duplicate values\n')


# Lets automate the plot of Categorical (dependent) and numeric (ind) variables
def cat_to_num_plot(df, 
                    independent_var = None,
                    dependent_var=None,
                    title=None,
                    bins=None,
                    x_label=None,
                    y_label=None,
                    alpha=0.7,
                    figsize=(8,6)
                    ):
    '''
    This function will plot histograms of a numeric var (independent) based on the values 
    of the dependent var (categorical).

    It accepts the following inputs:
    df: A pd.DataFrame object
    independent_var: Numeric independent variable    
    dependent_var: dependent categorical variable or target variable
    title: title o fthe graph
    bins: Number of bins
    x_label: x-axis label
    y_label: y-axis label
    alpha: the transparency level
    figsize: figure size. Default is (8,6)

    Logic:
    -----
    1. Declare an empty list. This will be modified by inserting a data frame (see step 2)
    2. For each level in the target var (which is a categorical val), create a data frame
       with the independnet variable's values. Name the column name of the data frame as 
       the level of the dependent categorical var
       Repeat the step 2 for each level in the categorical var (dependnet var).
       At the end of this step you should get a list ofdata frames
    3. For each data frame in the list, plot a histogram on the common axes.   
    '''

    if dependent_var in df.columns and independent_var in df.columns:
        if title == None:
            title = f"{independent_var} vs {dependent_var} plot"
        # Get the levels in dep cat var
        levels = list(set(df[dependent_var]))
        # Declare a list
        df_viz = []
        # For each level, create a data frame with the target numerical values
        for i in levels:
            # Create a temp data frame
            temp_df = df.loc[df[dependent_var] == i, [independent_var]]
            # Rename the columns of the temp data frame
            temp_df.columns = [str(i)]
            # Append the data frame to the list
            df_viz.append(temp_df)

        # Now plot each data frame
        if len(df_viz) > 0:
            if bins:
                ax = df_viz[0].plot.hist(alpha=alpha, bins=bins)
                for i in range(1, len(df_viz) - 1):
                    ax = df_viz[i].plot.hist(alpha=alpha, ax=ax, bins=bins)
                if len(df_viz) > 1:
                    df_viz[-1].plot.hist(alpha=alpha, ax=ax, bins=bins)
            else:                
                ax = df_viz[0].plot.hist(alpha=alpha)
                for i in range(1, len(df_viz) - 1):
                    ax = df_viz[i].plot.hist(alpha=alpha, ax=ax)
                if len(df_viz) > 1:
                    df_viz[-1].plot.hist(alpha=alpha, ax=ax)
            if x_label:
                plt.xlabel(x_label)
            if y_label:
                plt.ylabel(y_label)   
            if title:
                plt.title(title)
            plt.show()  
            plt.close()


def perform_cat_vs_num_analysis(df, cat_var, num_var, bins=None, alpha=0.7, 
                                figsize=(8,6),
                                log_scale=False,
                                sqrt_scale=False):
    if cat_var in df.columns and num_var in df.columns:
        # Plot the numeric hist:
        if bins:
            df[num_var].plot.hist(bins=bins, title=f"Histogram of {num_var}")      
            plt.show()
            plt.close()  
        else:    
            df[num_var].plot.hist(title=f"Histogram of {num_var}")      
            plt.show()
            plt.close()  

        cat_to_num_plot(df, 
                    independent_var = num_var,
                    dependent_var=cat_var,
                    title=f"{cat_var} vs {num_var} histogram",
                    bins=bins,
                    x_label=num_var,
                    y_label=None,
                    alpha=alpha,
                    figsize=figsize
                    )

        if log_scale:
            df_viz = df.loc[:, [cat_var, num_var]]
            num_var_log = num_var + '_log'
            df_viz[num_var_log] = np.log(df[num_var] + 1)
            cat_to_num_plot(df_viz, 
                    independent_var = num_var_log,
                    dependent_var=cat_var,
                    title=f"{cat_var} vs {num_var_log} histogram",
                    bins=bins,
                    x_label=num_var_log,
                    y_label=None,
                    alpha=alpha,
                    figsize=figsize
                    )       
            return df_viz[num_var_log]
        elif sqrt_scale:
            df_viz = df.loc[:, [cat_var, num_var]]
            num_var_sqrt = num_var + '_sqrt'
            df_viz[num_var_sqrt] = np.log(df[num_var] + 1)
            cat_to_num_plot(df_viz, 
                    independent_var = num_var_sqrt,
                    dependent_var=cat_var,
                    title=f"{cat_var} vs {num_var_sqrt} histogram",
                    bins=bins,
                    x_label=num_var_sqrt,
                    y_label=None,
                    alpha=alpha,
                    figsize=figsize
                    )       
            return df_viz[num_var_sqrt]
                       
        # print(f"summary of {num_var}: ")
        # nulls_perc = sum(df[num_var].isnull())/len(df)*100
        # print(f'The numeric variable "{num_var}" contains {round(nulls_perc, 4)}% of nulls')
        # print(f"It's MEAN is {df[num_var].mean()} and MEDIAN is: {df[num_var].median()}")
        # print(f"The std. dev is {df[num_var].std()}")





def split_into_three_parts(df, test=0.2, validation=0.2, dependent_var=None, stratify_var=None):
    '''
    Split the data frame into 3 parts: test, train, validation
    INPUTS:
    ------
    df: The data frame
    test: The desired proportion of test data (0.2 is the default). The proportion of the 
          training data will be 1-test.
    validation: Out of the training data, what perc should be the validation data?
                Default value is 0.2
    dependent_var: What is the dependent var 
                   (must be a categorical variable, not necessarily a string. can be a int also)
    stratify_var: which var to use for stratification? Defaulyt will be the dependent var

    OUTPUT:
    ------
    Returns the following data frames (in the below order):
    X_train, X_test, X_validation, y_train, y_test, y_validation 
    '''
    if stratify_var == None:
        stratify_var = dependent_var

    if dependent_var in df.columns and stratify_var in df.columns:
        # Split the data int test and train
        X_train, X_test, y_train, y_test = train_test_split(
                        df.drop(dependent_var, axis=1), df[dependent_var], 
                        test_size=test, 
                        stratify=df[stratify_var],
                        random_state=42)    

        # Split X_train into validation
        if stratify_var in X_train.columns:
            X_train, X_validation, y_train, y_validation = train_test_split(
                            X_train, y_train, 
                            test_size=validation, 
                            stratify=X_train[stratify_var],
                            random_state=42)    
        else:
            X_train, X_validation, y_train, y_validation = train_test_split(
                            X_train, y_train, 
                            test_size=validation, 
                            stratify=y_train,
                            random_state=42)    
        print("Shapes of the objects:")
        print("----------------------")
        print(f"Shape of X_train: {X_train.shape}")    
        print(f"Shape of y_train: {y_train.shape}")    
        print(f"Shape of X_test: {X_test.shape}")    
        print(f"Shape of y_test: {y_test.shape}")    
        print(f"Shape of X_validation: {X_validation.shape}")    
        print(f"Shape of y_validation: {y_validation.shape}")    

        return X_train, X_test, X_validation, y_train, y_test, y_validation


class CatImputer(BaseEstimator, TransformerMixin):
    '''
    Define a categorical imputer, as there is no builtin categorical imputer
    The cunstructor will accept the following inputs:
    value: "Which value to substitute". Applicable only when strategy="value"
    strategy: can accept 2 values: "value" (default) and "frequent"
              If "frequent",then substitute the most frequent value in place of nulls
              If "value", use the value parameter value to substitute
    '''
    def __init__(self, value="Unknown", strategy="value"):
        self.value = value
        self.strategy = strategy
        if self.strategy.lower() not in ["frequent", "value"]:
            raise Exception("The strategy must be 'frequent' or 'value'.")

    def transform(self, X):
        if self.strategy == "value":
            cols = X.columns
            fill_values = {}
            for col in cols:
                fill_val = str(col) + "_" + str(self.value)
                fill_values[col] = fill_val
            for k, v in fill_values.items():
                X.loc[:, k] = X.loc[:, k].fillna(fill_values[k]) 
            return X
        elif self.strategy == "frequent":
            cols = X.columns
            freq_values= {}
            for col in cols:
                freq_values[col] = X[col].value_counts().sort_values(ascending=False)[0]

            for k, v in freq_values.items():
                X.loc[:,k] = X.loc[:,k].fillna(freq_values[k])
            return X    

    def fit_transform(self, X, y):
        return self.transform(X)

    def fit(self, X, y = None):
        return self        


def plot_confusion_matrix(model=None, predictions=None, true_labels=None):
    cm = confusion_matrix(predictions, true_labels)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # view with a heatmap
    sns.heatmap(cm, annot=True, annot_kws={"size":30}, 
            cmap='Blues', square=True, fmt='.3f')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if model:
        plt.title('Confusion matrix for:\n{}'.format(model.__class__.__name__))
    plt.show()         


def get_classifier_scores(actual_labels, predicted_labels, predicted_y_proba, pos_label=None):
    if pos_label:    
        accuracy = accuracy_score(actual_labels, predicted_labels)
        precision = precision_score(actual_labels, predicted_labels,pos_label=pos_label)
        recall = recall_score(actual_labels, predicted_labels,pos_label=pos_label)
        f1 = f1_score(actual_labels, predicted_labels,pos_label=pos_label)

        auc_roc = roc_auc_score(y_true = actual_labels, y_score = predicted_y_proba)

        auc_precision_recall = average_precision_score(y_true = actual_labels, 
                                                       y_score=predicted_y_proba,
                                                       pos_label=pos_label)

        return pd.Series({'Accuracy': accuracy,
                             'Precision': precision,
                             'Recall': recall,
                             'F1 Score': f1,
                             'ROC AUC': auc_roc,
                             'PR AUC': auc_precision_recall
                            })
    else:
        print("Yom must supply the positive label")   



def compare_classifiers(classifiers, X, y, pos_label=None):
    '''
    classifiers = List of classifiers
    X: Input variables
    y: True labels
    '''
    if pos_label:
        df = pd.DataFrame()
        for clf in classifiers:
            predicted_y = clf.predict(X)

            # Get the probs of pos_label
            predicted_y_proba = clf.predict_proba(X)

            # Get the index of positive label
            i = list(clf.classes_).index(pos_label)
            predicted_y_proba = predicted_y_proba[:, i]

            s = get_classifier_scores(y, predicted_y, predicted_y_proba, pos_label)

            col_name = clf.__class__.__name__
            df[col_name] = s

        df.plot.bar(figsize=(10,8))
        plt.title("Classifiers scores")
        plt.show()
        plt.close()

        return df
    else:
        print("Yom must supply the positive label")   


def get_all_value_counts(df):
    df_cat = df.select_dtypes(object)
    d = {}
    for col in df_cat.columns:
        d[col] = len(df_cat[col].value_counts())
    return pd.Series(d).sort_values(ascending=False)


def perform_mass_cat_vs_cat_analysis(df, dependent_var=None):
    cat_cols = df.select_dtypes(object).columns
    if dependent_var and dependent_var in cat_cols:
        for col in cat_cols:
            print(f"Analysis of {col} vs {dependent_var}")
            if col in cat_cols and col != dependent_var:
                perform_cat_vs_cat_analysis(df, var_1=col, var_2=dependent_var)


def perform_mass_cat_vs_num_analysis(df, dependent_var=None):
    num_cols = df.select_dtypes([float, int]).columns
    if dependent_var and dependent_var in df.columns:
        for col in num_cols:
            print(f"Analysis of {col} vs {dependent_var}")
            if col in num_cols and col != dependent_var:
                perform_cat_vs_num_analysis(df, num_var=col, cat_var=dependent_var)








