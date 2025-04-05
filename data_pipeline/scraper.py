from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

from typing import Tuple, Optional

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, # level=logging.INFO: This sets the minimum logging level. It means only log messages with severity INFO and above (WARNING, ERROR, CRITICAL) will be shown.
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class WebDriverManager:
    """Manages the lifecycle of the Chrome WebDriver"""
    def __init__(self):
        self.chrome_options = Options()
        self._configure_chrome_options()
        self.service = Service("/usr/bin/chromedriver") # "/usr/bin/chromedriver" for linux Streamlit Cloud Deployment # "C:/Users/DELL/chromedriver-win64/chromedriver.exe" in Windows local

    def _configure_chrome_options(self):
        """Configure Chrome options for headless browsing"""
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-gpu") # not needed during Streamlit Cloud Deployment
        self.chrome_options.binary_location = "/usr/bin/chromium" # "/usr/bin/chromium" for linux Streamlit Cloud Deployment # "C:/Program Files/Google/Chrome/Application/chrome.exe" for Windows local

    def get_driver(self):
        """Returns a new WebDriver instance"""
        return webdriver.Chrome(service=self.service, options=self.chrome_options)

class GoldPriceScraper:
    """Scrapes gold price data from indgold.com for different time periods"""
    
    BASE_URL = "https://www.indgold.com/{city}-gold-rate{page_suffix}"
    
    def __init__(self, city: str):
        self.city = city.lower()
        self.driver_manager = WebDriverManager()
        self.logger = logging.getLogger(__name__)
        
    def _get_url_for_period(self, month: str, year: int) -> str:
        """Determine the correct URL format based on the date range"""
        target_date = datetime(year, self._month_to_number(month), 1) # The line creates a datetime object representing the first day of the specified month and year.
        
        #if datetime(2021, 8, 1) <= target_date <= datetime(2023, 7, 31):
        #    return self.BASE_URL.format(city=self.city, page_suffix=f"-{month}-{year}.htm")
        #elif datetime(2023, 8, 1) <= target_date < datetime.now().replace(day=1):
        #    return self.BASE_URL.format(city=self.city, page_suffix=f"-{month}-{year}.htm")

        if datetime(2021, 8, 1) <= target_date and target_date < datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0):
            return self.BASE_URL.format(city=self.city, page_suffix=f"-{month}-{year}.htm")
        elif self.city in ['mumbai','delhi','chennai','bangalore','hyderabad','cochin']:
            return self.BASE_URL.format(city=self.city, page_suffix="s.htm")
        else:    
            return self.BASE_URL.format(city=self.city, page_suffix=".htm")

    def _month_to_number(self, month: str) -> int:
        """Convert month name to its corresponding number"""
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                 'july', 'august', 'september', 'october', 'november', 'december']
        return months.index(month) + 1

    def _scrape_table_data(self, html_content: str, period_type: str) -> pd.DataFrame:
        """Scrape table data based on the website format"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        if period_type == "legacy":
            tables = soup.find_all('table')
            if len(tables) >= 2:
                return self._parse_table(tables[1])
        else:
            table_div = soup.find('div', id='table')
            if table_div:
                return self._parse_table(table_div)
                
        return pd.DataFrame(columns=['Date', 'Morning', 'Evening'])

    def _parse_table(self, table_element) -> pd.DataFrame:
        """Parse table data into a DataFrame"""
        rows = table_element.find_all('tr')
        data = []
        
        for row in rows[1:]:
            cols = row.find_all('td')
            data.append([col.get_text(strip=True) for col in cols])
            
        df = pd.DataFrame(data, columns=['Date', 'Morning', 'Evening'])

        #try:
        #    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        #except ValueError:
        #    df['Date'] = pd.to_datetime(df['Date'])
        
        return df

    def _fetch_page_content(self, url: str) -> Optional[str]:
        """Fetch HTML content using Selenium"""
        try:
            driver = self.driver_manager.get_driver()
            driver.get(url)
            content = driver.page_source
            driver.quit()
            return content
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def scrape_month(self, month: str, year: int) -> pd.DataFrame:
        """Scrape data for a specific month and year"""
        url = self._get_url_for_period(month, year)
        self.logger.info(f"Scraping data from: {url}")
        
        html_content = self._fetch_page_content(url)
        if not html_content:
            return pd.DataFrame()
            
        period_type = "legacy" if datetime(year, self._month_to_number(month), 1) <= datetime(2023, 7, 31) else "current"
        return self._scrape_table_data(html_content, period_type)

    def scrape_range(self, start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
        """Scrape data for a range of dates"""
        if end_date is None:
            today = datetime.today()
            end_date = today - timedelta(days=1)
            #end_date = datetime.now()
            
        all_data = pd.DataFrame()
        current_date = start_date
        
        while current_date <= end_date:
            month_name = current_date.strftime('%B').lower()
            year = current_date.year
            
            month_data = self.scrape_month(month_name, year)
            
            #today = datetime.now().strftime('%d-%b-%y')  # Today's date in 'd-Mon-yy' format
            #if month_data['Date'].iloc[-1] == today:
            #    month_data = month_data.drop(month_data.index[-1])
            #print(month_data)

            
            # Filter data within the exact date range
            #month_data['Date'] = pd.to_datetime(month_data['Date'])
            #month_data = month_data[(month_data['Date'] >= start_date) & (month_data['Date'] <= end_date)]
            
            def is_within_range(date_str, start_date, end_date):
            # Parse the 'd-Mon-yy' date string manually
                date_obj = datetime.strptime(date_str, '%d-%b-%y')
                return start_date <= date_obj <= end_date

            month_data = month_data[month_data['Date'].apply(lambda x: is_within_range(x, start_date, end_date))]

            all_data = pd.concat([all_data, month_data], ignore_index=True)
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
                
        return all_data

    def scrape_current_month(self) -> pd.DataFrame:
        """Scrape data for the current month"""
        today = datetime.now()
        return self.scrape_month(today.strftime('%B').lower(), today.year) # strftime stands for "string format time". It's a method in Python's datetime module used to format datetime objects as strings according to a specified format.

# Example usage:
if __name__ == "__main__":
    scraper = GoldPriceScraper("coimbatore")
    
    # Scrape historical data
    historical_data = scraper.scrape_range(
        start_date=datetime(2021, 8, 1),
        end_date=datetime(2023, 7, 31)
    )
    
    # Scrape recent data
    recent_data = scraper.scrape_range(
        start_date=datetime(2023, 8, 1),
        end_date=datetime.now()
    )
    
    # Scrape current month
    current_month_data = scraper.scrape_current_month()
    
    # Combine all data
    combined_data = pd.concat([historical_data, recent_data, current_month_data], ignore_index=True)
