#!/home/femibyte/local/anaconda/bin/python
import pandas as pd
import argparse
import pprint as pp

parser = argparse.ArgumentParser(description='survival table implementation for Kaggle Titanic problem')
 
parser.add_argument('--train', dest='train_file', help='training file path',
                    required=True)

parser.add_argument('--test', dest='test_file', help='test file path',
                    required=True)

parser.add_argument('--results', dest='results_file', help='results file path',
                    required=True)

args=parser.parse_args()

train_file=args.train_file
test_file=args.test_file
results_file=args.results_file

def add_row(df, row):
    return df.append(pd.DataFrame(row), ignore_index=True)
 
def bucket_contains_fare(fare, fare_bracket_size, bucket):
    return (fare > bucket * fare_bracket_size) & (fare <= ((bucket+1) * fare_bracket_size))
 
def generate_survival_table(train_file):    
    fare_ceiling = 40
    train_df = pd.read_csv(train_file)
    train_df[train_df['Fare'] >= 39.0] = 39.0
    fare_bracket_size = 10  
    num_price_buckets = fare_ceiling / fare_bracket_size # Number of price buckets
    num_passenger_classes = 3 #1st, 2nd and 3rd class.
 
    survival_table = pd.DataFrame(columns=['Sex', 'Pclass', 'PriceBucket', 'Survived', 'NumberOfPeople'])
 
    for pclass in range(1, num_passenger_classes + 1): # add 1 to handle 0 start
        for bucket in range(0, num_price_buckets):
            for sex in ['female', 'male']:
                survival = train_df[(train_df['Sex'] == sex) 
                                    & (train_df['Pclass'] == pclass) 
                                    & bucket_contains_fare(train_df["Fare"], fare_bracket_size, bucket)]
 
                row = [dict(Pclass=pclass, Sex=sex, PriceBucket = bucket, 
                            Survived = round(survival['Survived'].mean()), 
                            NumberOfPeople = survival.count()[0]) ]
                survival_table = add_row(survival_table, row)
 
    return survival_table.fillna(0)
 



def bucket_fare(fare):
    if (fare >= 0 and fare < 10):
        return 0
    elif (fare >= 10 and fare < 20):
        return 1
    elif (fare >= 20 and fare < 30):
        return 2
    else:
        return 3
 
def compute_survival(survival_table, row):
    survival_row = survival_table[(survival_table["Sex"] == row["Sex"]) & (survival_table["Pclass"] == row["Pclass"]) & (survival_table["PriceBucket"] == bucket_fare(row["Fare"]))]
    return int(survival_row["Survived"].iat[0])    
 
def main():
    survival_table = generate_survival_table(train_file)
    test_df = pd.read_csv(test_file)             
    test_df["Survived"] = test_df.apply(lambda row: compute_survival(survival_table, row), axis=1)
    test_df.to_csv(results_file,cols=['PassengerId', 'Survived'], index=False)
 #   pp.pprint(test_df)
    print "S_TABLE:\n %s" % survival_table.to_string()

if __name__ == '__main__':
    main()
