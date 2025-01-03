import gzip
import json
import os

def check_gzip_file(filepath: str, num_lines: int = 5):
    """
    Check contents of a gzipped JSON file
    
    Args:
        filepath: Path to the gzipped file
        num_lines: Number of lines to check
    """
    print(f"\nChecking file: {filepath}")
    print(f"File size: {os.path.getsize(filepath)} bytes")
    
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            print("\nFirst few lines:")
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                try:
                    # Try to parse JSON
                    data = json.loads(line.strip())
                    print(f"\nLine {i+1} (parsed JSON):")
                    print(json.dumps(data, indent=2)[:500] + "...")  # Print first 500 chars
                except json.JSONDecodeError as e:
                    print(f"\nLine {i+1} (raw, failed to parse):")
                    print(line.strip()[:500] + "...")
                    print(f"JSON Error: {str(e)}")
                except Exception as e:
                    print(f"Error on line {i+1}: {str(e)}")
                    
    except Exception as e:
        print(f"Error opening file: {str(e)}")

def main():
    # Check both review and metadata files
    files_to_check = [
        'data/meta_Beauty.json.gz',
        'data/reviews_Beauty_5.json.gz'
    ]
    
    for filepath in files_to_check:
        check_gzip_file(filepath)

if __name__ == "__main__":
    main()