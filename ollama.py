
import pandas as pd
import requests
import json
import numpy as np
from typing import Optional, Dict, Any
import time
import re

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"  # Adjust if your Docker container uses different port
MODEL_NAME = "llama3.1"

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate text using Ollama"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama: {e}")
            return ""
    
    def test_connection(self) -> bool:
        """Test if Ollama is accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

# Initialize Ollama client
ollama = OllamaClient(OLLAMA_BASE_URL, MODEL_NAME)

# Test connection
print("Testing Ollama connection...")
if ollama.test_connection():
    print("✅ Successfully connected to Ollama")
else:
    print("❌ Failed to connect to Ollama. Please check your Docker container is running.")

# Load and examine the data
print("\nLoading Divar.csv data...")
try:
    divar = pd.read_csv('Divar_clean.csv')
    df = divar.copy()
    print(f"✅ Data loaded successfully. Shape: {df.shape}")
    
    print("\nColumn names:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nMissing values per column:")
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            print(f"{col}: {count} missing values ({count/len(df)*100:.1f}%)")
            
except FileNotFoundError:
    print("❌ Divar.csv not found. Please ensure the file is in the current directory.")
    df = None

class DataExtractor:
    """Extract structured data from text using LLM"""
    
    def __init__(self, ollama_client: OllamaClient):
        self.ollama = ollama_client
    
    def extract_divar_data(self, title: str, description: str) -> Dict[str, Any]:
        """Extract Divar-specific data from title and description"""
        
        system_prompt = """You are a Persian/Farsi real estate data extraction expert for Divar.ir listings.
        Extract information from the Persian text and return ONLY a JSON object with these exact fields:
        
        REQUIRED FIELDS (extract these with high priority):
        - "rooms_count": number of bedrooms/rooms (خواب، اتاق) - extract as integer
        - "construction_year": year building was built (ساخت، سال ساخت) - extract as integer 
        - "building_size": size in square meters (متر، متراژ) - extract as integer
        - "has_parking": "دارد" if parking mentioned, "ندارد" if explicitly no parking, null otherwise
        - "has_warehouse": "دارد" if storage/warehouse mentioned, "ندارد" if explicitly none, null otherwise
        - "price_mode": "توافقی" for negotiable, "قطعی" for fixed price, null if unclear
        - "price_value": numeric price in Toman (remove all commas and text)
        - "floor": floor number or description (طبقه)
        - "has_elevator": "دارد" if elevator mentioned, "ندارد" if explicitly none, null otherwise
        - "has_balcony": "دارد" if balcony mentioned (بالکن، تراس), "ندارد" if none, null otherwise
        - "user_type": "مشاور املاک" if real estate agent, "شخصی" if individual, null if unclear
        - "neighborhood_slug": neighborhood name in Persian if mentioned
        - "rent_mode": "ماهانه", "روزانه", "سالانه" etc. for rental period
        - "rent_value": numeric rental value in Toman
        
        IMPORTANT EXTRACTION RULES:
        - For numbers: Extract only digits, ignore commas and Persian text
        - For yes/no fields: Use "دارد"/"ندارد" format consistently
        - Use null (not empty string) for missing information
        - Look for common Persian real estate terms
        - Be precise with numeric extractions"""
        
        prompt = f"""Extract real estate data from this Divar listing:

TITLE: {title}

DESCRIPTION: {description}

Return ONLY valid JSON with the specified fields. Focus on accuracy over completeness:"""
        
        response = self.ollama.generate(prompt, system_prompt)
        
        try:
            # Clean response and extract JSON
            cleaned_response = response.strip()
            # Try to find JSON block
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL | re.MULTILINE)
            if json_match:
                json_str = json_match.group()
                extracted_data = json.loads(json_str)
                # Clean and validate the data
                return self._clean_extracted_data(extracted_data)
        except Exception as e:
            print(f"JSON parsing error for row: {e}")
            print(f"Raw response: {response[:200]}...")
        
        return {}
    
    def _clean_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate extracted data with robust Persian text handling"""
        cleaned = {}
        
        # Numeric fields that should be integers
        integer_fields = ['rooms_count', 'construction_year', 'building_size', 'price_value', 'rent_value']
        for field in integer_fields:
            if field in data and data[field] is not None:
                try:
                    # Handle Persian/Arabic numerals and clean text
                    value_str = str(data[field])
                    # Remove common Persian punctuation and text
                    value_str = re.sub(r'[^\d۰-۹]', '', value_str)
                    # Convert Persian numbers to English
                    persian_to_english = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')
                    value_str = value_str.translate(persian_to_english)
                    
                    if value_str:  # Only convert if we have digits
                        cleaned[field] = int(value_str)
                    else:
                        cleaned[field] = None
                except (ValueError, TypeError):
                    cleaned[field] = None
            else:
                cleaned[field] = None
        
        # String fields - keep as is but clean whitespace
        string_fields = ['price_mode', 'rent_mode', 'floor', 'user_type', 'neighborhood_slug']
        for field in string_fields:
            value = data.get(field)
            if value and str(value).strip():
                cleaned[field] = str(value).strip()
            else:
                cleaned[field] = None
        
        # Boolean-like fields (standardize to دارد/ندارد format)
        boolean_fields = ['has_parking', 'has_warehouse', 'has_elevator', 'has_balcony']
        for field in boolean_fields:
            value = data.get(field)
            if value:
                value_str = str(value).lower().strip()
                # Check for positive indicators
                if any(word in value_str for word in ['دارد', 'هست', 'موجود', 'yes', 'true', '1']):
                    cleaned[field] = 'دارد'
                # Check for negative indicators  
                elif any(word in value_str for word in ['ندارد', 'نیست', 'نداره', 'no', 'false', '0']):
                    cleaned[field] = 'ندارد'
                else:
                    cleaned[field] = None
            else:
                cleaned[field] = None
        
        return cleaned
    
    def extract_property_features(self, title: str, description: str) -> Dict[str, Any]:
        """Extract property features from text"""
        
        system_prompt = """You are a real estate data extraction expert. Extract property features from Persian/Farsi listings.
        Return ONLY a JSON object with these fields:
        - "rooms": number of rooms/bedrooms
        - "area": area in square meters
        - "floor": floor number
        - "age": building age in years
        - "parking": true/false for parking availability
        - "elevator": true/false for elevator
        - "balcony": true/false for balcony
        - "storage": true/false for storage room
        
        If information is not found, use null."""
        
        prompt = f"""Extract property features from this listing:
        Title: {title}
        Description: {description}
        
        Return only valid JSON:"""
        
        response = self.ollama.generate(prompt, system_prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {}
    
    def extract_location_info(self, title: str, description: str) -> Dict[str, Any]:
        """Extract location information"""
        
        system_prompt = """Extract location information from Persian/Farsi real estate listings.
        Return ONLY a JSON object with these fields:
        - "city": city name
        - "district": district/neighborhood name
        - "street": street name if mentioned
        - "metro_nearby": true/false if near metro/subway
        
        If information is not found, use null."""
        
        prompt = f"""Extract location information from this listing:
        Title: {title}
        Description: {description}
        
        Return only valid JSON:"""
        
        response = self.ollama.generate(prompt, system_prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {}

# Initialize data extractor
extractor = DataExtractor(ollama)

def process_row(row_data: pd.Series, target_columns: list) -> Dict[str, Any]:
    """Process a single row and extract data for target columns"""
    
    title = str(row_data.get('title', ''))
    description = str(row_data.get('description', ''))
    
    # Extract comprehensive Divar data
    extracted_data = extractor.extract_divar_data(title, description)
    
    # Filter results to only include target columns
    results = {col: extracted_data.get(col) for col in target_columns if col in extracted_data}
    
    return results

def fill_missing_data(df: pd.DataFrame, target_columns: list, sample_size: int = 5) -> pd.DataFrame:
    """Fill missing data using LLM extraction"""
    
    if df is None:
        print("No data to process")
        return None
    
    print(f"\nProcessing {sample_size} rows to fill missing data...")
    df_copy = df.copy()
    
    # Process a sample of rows
    sample_indices = df.index[:sample_size]  # Take first N rows
    
    for idx in sample_indices:
        print(f"Processing row {idx}...")
        
        row_data = df.iloc[idx]
        extracted_data = process_row(row_data, target_columns)
        
        # Fill missing values
        for col in target_columns:
            if pd.isna(df_copy.loc[idx, col]) and col in extracted_data:
                df_copy.loc[idx, col] = extracted_data[col]
                print(f"  Filled {col}: {extracted_data[col]}")
        
        # Add delay to avoid overwhelming the API
        time.sleep(1)
    
    return df_copy

# Strategic TARGET_COLUMNS based on your actual missing data percentages
# Prioritizing columns with moderate missing rates that are extractable from text

# HIGH PRIORITY - Most extractable and valuable
HIGH_PRIORITY_COLUMNS = [
    'rooms_count',         # 15.4% missing - highly extractable from text
    'construction_year',   # 18.4% missing - often mentioned in descriptions
    'building_size',       # 2.0% missing - but very valuable when missing
    'has_parking',         # 27.2% missing - commonly mentioned
    'has_warehouse',       # 27.2% missing - often in descriptions
]

# MEDIUM PRIORITY - More challenging but doable  
MEDIUM_PRIORITY_COLUMNS = [
    'price_mode',          # 42.6% missing - "توافقی" vs "قطعی"
    'price_value',         # 43.2% missing - numeric price extraction
    'floor',               # 45.8% missing - floor information
    'has_elevator',        # 45.8% missing - building amenities
    'has_balcony',         # 49.4% missing - apartment features
]

# CHALLENGING - High missing rates, complex extraction
CHALLENGING_COLUMNS = [
    'neighborhood_slug',   # 56.3% missing - location extraction
    'rent_mode',           # 64.7% missing - rental terms
    'rent_value',          # 64.9% missing - rental prices
    'user_type',           # 71.1% missing - "مشاور املاک" vs "شخصی"
]

# Start with HIGH_PRIORITY for testing
TARGET_COLUMNS = HIGH_PRIORITY_COLUMNS

print(f"\nTarget columns for filling: {TARGET_COLUMNS}")

if df is not None:
    # Check which target columns exist in the dataframe
    existing_columns = [col for col in TARGET_COLUMNS if col in df.columns]
    missing_columns = [col for col in TARGET_COLUMNS if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: These columns don't exist in the CSV: {missing_columns}")
    
    if existing_columns:
        print(f"Will process these existing columns: {existing_columns}")
        
        # Process a small sample first (change sample_size as needed)
        processed_df = fill_missing_data(df, existing_columns, sample_size=3)
        
        if processed_df is not None:
            print("\n" + "="*50)
            print("RESULTS COMPARISON")
            print("="*50)
            
            for col in existing_columns:
                if col in df.columns:
                    original_missing = df[col].isna().sum()
                    new_missing = processed_df[col].isna().sum()
                    filled_count = original_missing - new_missing
                    
                    print(f"\nColumn '{col}':")
                    print(f"  Originally missing: {original_missing}")
                    print(f"  Still missing: {new_missing}")
                    print(f"  Filled: {filled_count}")
                    
                    if filled_count > 0:
                        print("  Sample filled values:")
                        filled_rows = processed_df[df[col].isna() & processed_df[col].notna()]
                        for idx in filled_rows.index[:3]:  # Show first 3 filled values
                            print(f"    Row {idx}: {processed_df.loc[idx, col]}")
            
            # Save results
            processed_df.to_csv('Divar_processed.csv', index=False)
            print(f"\n✅ Processed data saved to 'Divar_processed.csv'")
    else:
        print("No target columns found in the dataset.")

# Function to analyze a specific row in detail
def analyze_single_row(df: pd.DataFrame, row_index: int):
    """Analyze a single row in detail"""
    
    if df is None or row_index >= len(df):
        print("Invalid row index or no data")
        return
    
    row = df.iloc[row_index]
    title = str(row.get('title', ''))
    description = str(row.get('description', ''))
    
    print(f"\n" + "="*60)
    print(f"DETAILED ANALYSIS FOR ROW {row_index}")
    print("="*60)
    
    print(f"\nTitle: {title[:100]}...")
    print(f"\nDescription: {description[:200]}...")
    
    # Extract comprehensive Divar data
    print("\n--- EXTRACTED DIVAR DATA ---")
    divar_data = extractor.extract_divar_data(title, description)
    for key, value in divar_data.items():
        print(f"{key}: {value}")
        
    # Show original values for comparison
    print("\n--- ORIGINAL VALUES IN DATASET ---")
    relevant_cols = ['price_value', 'price_mode', 'rent_value', 'rent_mode', 'rooms_count', 
                    'floor', 'has_parking', 'has_elevator', 'construction_year', 'user_type']
    for col in relevant_cols:
        if col in row.index:
            print(f"{col}: {row[col]}")

# Example: Analyze the first row in detail
if df is not None and len(df) > 0:
    print("\n" + "="*60)
    print("SAMPLE ANALYSIS OF FIRST ROW")
    print("="*60)
    analyze_single_row(df, 0)

# Additional utility functions for large dataset processing

def process_batch(df: pd.DataFrame, start_idx: int, batch_size: int, target_columns: list) -> pd.DataFrame:
    """Process a batch of rows for large datasets"""
    
    end_idx = min(start_idx + batch_size, len(df))
    batch_df = df.iloc[start_idx:end_idx].copy()
    
    print(f"Processing batch: rows {start_idx} to {end_idx-1}")
    
    for i, (idx, row) in enumerate(batch_df.iterrows()):
        if i % 10 == 0:  # Progress update every 10 rows
            print(f"  Processing row {idx} ({i+1}/{len(batch_df)})")
        
        extracted_data = process_row(row, target_columns)
        
        # Fill missing values
        for col in target_columns:
            if pd.isna(batch_df.loc[idx, col]) and col in extracted_data:
                batch_df.loc[idx, col] = extracted_data[col]
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.5)
    
    return batch_df

def smart_sampling_strategy(df: pd.DataFrame, target_columns: list, sample_size: int = 100):
    """Intelligent sampling strategy for large datasets"""
    
    print(f"\n--- SMART SAMPLING FOR {sample_size} ROWS ---")
    
    # Sample strategy: take rows with most missing target columns
    missing_counts = df[target_columns].isna().sum(axis=1)
    
    # Get indices of rows with most missing data
    high_missing_indices = missing_counts.nlargest(sample_size//2).index
    
    # Get some random indices for diversity
    random_indices = df.sample(sample_size//2).index
    
    # Combine both strategies
    selected_indices = list(high_missing_indices) + list(random_indices)
    selected_indices = list(set(selected_indices))[:sample_size]  # Remove duplicates and limit
    
    print(f"Selected {len(selected_indices)} rows for processing")
    print(f"Average missing columns per selected row: {missing_counts[selected_indices].mean():.1f}")
    
    return selected_indices

def estimate_processing_time(df: pd.DataFrame, sample_size: int):
    """Estimate processing time based on sample size"""
    
    time_per_row = 2  # seconds (conservative estimate including API delays)
    total_time = sample_size * time_per_row
    
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60
    
    print(f"\nEstimated processing time for {sample_size} rows:")
    if hours > 0:
        print(f"  {hours}h {minutes}m {seconds}s")
    elif minutes > 0:
        print(f"  {minutes}m {seconds}s")
    else:
        print(f"  {seconds}s")

print("\n" + "="*60)
print("DIVAR ANALYSIS TOOLKIT - READY")
print("="*60)

print(f"\nDataset Overview:")
print(f"- Total rows: {len(df):,}")
print(f"- Total columns: {len(df.columns)}")

print(f"\nMissing Data Analysis (Top 15 columns by missing percentage):")
missing_analysis = df.isna().mean().sort_values(ascending=False).head(15)
for col, missing_pct in missing_analysis.items():
    missing_count = int(missing_pct * len(df))
    print(f"  {col}: {missing_count:,} missing ({missing_pct:.1f}%)")

print(f"\nSTRATEGIC COLUMN PRIORITIES:")
print(f"\n🟢 HIGH PRIORITY (Most extractable):")
for col in HIGH_PRIORITY_COLUMNS:
    if col in df.columns:
        missing_pct = df[col].isna().mean() * 100
        print(f"  {col}: {missing_pct:.1f}% missing")

print(f"\n🟡 MEDIUM PRIORITY (Moderately extractable):")  
for col in MEDIUM_PRIORITY_COLUMNS:
    if col in df.columns:
        missing_pct = df[col].isna().mean() * 100
        print(f"  {col}: {missing_pct:.1f}% missing")

print(f"\n🔴 CHALLENGING (Complex extraction):")
for col in CHALLENGING_COLUMNS:
    if col in df.columns:
        missing_pct = df[col].isna().mean() * 100
        print(f"  {col}: {missing_pct:.1f}% missing")

print(f"\nCURRENT TARGET_COLUMNS: {TARGET_COLUMNS}")

# Show sample data structure
print(f"\n📋 SAMPLE DATA INSPECTION:")
print(f"First row title: {df['title'].iloc[0]}")
print(f"First row description (first 200 chars): {df['description'].iloc[0][:200]}...")

# User type analysis from sample
print(f"\nUSER_TYPE analysis from your sample:")
print(f"Row 0 user_type: {df['user_type'].iloc[0]}")
print(f"Row 1 user_type: {df['user_type'].iloc[1]}")

print(f"\nRecommended Workflow:")
print(f"1. 🧪 TEST: Start with 5-10 rows using HIGH_PRIORITY_COLUMNS")
print(f"2. 📊 EVALUATE: Check extraction accuracy") 
print(f"3. 🔧 OPTIMIZE: Adjust prompts based on results")
print(f"4. 📈 SCALE: Gradually increase to 50, 100, 1000+ rows")
print(f"5. 🎯 EXPAND: Add MEDIUM_PRIORITY_COLUMNS once HIGH works well")

print(f"\nRecommended Usage:")
print(f"1. Start with small sample: process_sample = 10-50 rows")
print(f"2. Use smart sampling: selected_indices = smart_sampling_strategy(df, TARGET_COLUMNS, 50)")
print(f"3. Process in batches: process_batch(df, start_idx=0, batch_size=10, target_columns=TARGET_COLUMNS)")
print(f"4. Scale up gradually based on results")

# Example execution for small sample
print(f"\nExecuting small sample analysis (10 rows)...")
estimate_processing_time(df, 10)

# Smart sampling for better results
selected_indices = smart_sampling_strategy(df, TARGET_COLUMNS, 10)
sample_df = df.loc[selected_indices].copy()

print(f"\nProcessing selected sample...")
processed_sample = fill_missing_data(sample_df, TARGET_COLUMNS, sample_size=10)