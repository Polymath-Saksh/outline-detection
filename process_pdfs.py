import os
from enhanced_pdf_extractor import process_pdf_enhanced
import json

INPUT_DIR = '/app/input'
OUTPUT_DIR = '/app/output'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_pdf(filename):
    return filename.lower().endswith('.pdf')

def main():
    pdf_files = [f for f in os.listdir(INPUT_DIR) if is_pdf(f)]
    if not pdf_files:
        print('No PDF files found in input directory.')
        return
    for pdf_file in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, pdf_file)
        print(f'Processing: {pdf_file}')
        try:
            result = process_pdf_enhanced(pdf_path)
            output_filename = os.path.splitext(pdf_file)[0] + '.json'
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f'Output written to: {output_path}')
        except Exception as e:
            print(f'Error processing {pdf_file}: {e}')

if __name__ == '__main__':
    main()
