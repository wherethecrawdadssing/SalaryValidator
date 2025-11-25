import pandas as pd
import argparse
import json
import datetime

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.report = {}
    

    def load_data(self):
        self.df = pd.read_csv(self.filepath)

    def validate_columns(self):
        self.report["missing_values"] = self.df.isna().sum().to_dict()
        #print(self.df.head())
        #print(self.df.isna().sum())

        missing = self.df.isna().sum()
        missing = missing[missing > 0]
        self.report["missing_values"] = missing.to_dict()
    
    def duplicates_rows(self):
    
        self.report["duplicates_rows"] = self.df.duplicated().sum()

    def save_report(self, filepath):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}.json"
        with open(filepath, 'w') as f:
            json.dump(self.report, f, indent = 4)
        print(f"Report saved as {filename}")

    
    def run(self, report_path = None):

        self.load_data()
        self.validate_columns()
        self.duplicates_rows()
        
        if report_path is not None:
            self.save_report(report_path)

        return self.report

if __name__ == "__main__":
    #create parser
    parser = argparse.ArgumentParser(description="DataValidator: validate CSV datasets.")

    parser.add_argument(
        "--file",
        "-f",
        required = True,
        help = "Path to the CSV file you want to validate."
    )

    args = parser.parse_args()

    validator = DataLoader(args.file)

    report = validator.run()

    print(report)







        
        
        




        

