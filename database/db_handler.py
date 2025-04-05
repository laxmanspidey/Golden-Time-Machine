import sqlite3
import pandas as pd
from datetime import datetime

class GoldPriceDB:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        
    def check_city_data(self, city):
        query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{city.lower()}_prices';"
        return pd.read_sql(query, self.conn).shape[0] > 0
    
    #def get_latest_date(self, city):
    #    return pd.read_sql(f"SELECT MAX(date) FROM {city.lower()}_prices;", self.conn).iloc[0,0]
    
    def get_latest_date(self, city):
        try:
            query = f"""
            SELECT Date
            FROM {city.lower()}_prices
            WHERE Date IS NOT NULL;
            """            
            result = pd.read_sql(query, self.conn)

            if not result.empty:
                # Function to convert the date string into 'YYYY-MM-DD' format
                def convert_date(date_str):
                    day, month_str, year = date_str.split('-')
                    
                    months = {
                        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                        'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                        'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
                    }
                    
                    month = months.get(month_str, '00')  # Default to '00' if month is invalid
                    
                    # Handle single digit day (e.g., '1' should become '01')
                    day = day.zfill(2) # zero fill for length 2
                    
                    # Ensure the year is in four digits (e.g., '22' -> '2022')
                    year = '20' + year
                    
                    return f"{year}-{month}-{day}"
                
                formatted_dates = [convert_date(date) for date in result['Date']]

                max_date = max(formatted_dates, key=lambda d: datetime.strptime(d, "%Y-%m-%d"))

                return max_date

            else:
                print("No valid dates found.")
                return None 
            
        except Exception as e:
            print(f"Error fetching latest date: {str(e)}")
            return None
        
    def update_data(self, city, new_df):
        new_df.to_sql(f"{city.lower()}_prices", self.conn, if_exists='append', index=False)
        
    def get_all_data(self, city):
        return pd.read_sql(f"SELECT * FROM {city.lower()}_prices ORDER BY date;", self.conn)
    
    def close(self):
        self.conn.close()
