import os
import json
import time
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Assuming these imports are correct based on the original code structure
from round1a.outline_extractor_enhanced import OutlineExtractorEnhanced
from round1b.document_analyst_enhanced import DocumentAnalystEnhanced

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    challenge_1b_dir = os.path.join(base_dir, "Challenge_1b")

    if not os.path.exists(challenge_1b_dir):
        print(f"Error: Directory '{challenge_1b_dir}' not found.")
        print("Please ensure 'Challenge_1b' folder is in the same directory as main.py.")
        return

    # Find all collection folders (e.g., Collection1, Collection2)
    collection_dirs = [
        os.path.join(challenge_1b_dir, d)
        for d in os.listdir(challenge_1b_dir)
        if os.path.isdir(os.path.join(challenge_1b_dir, d)) and d.startswith("Collection")
    ]

    if not collection_dirs:
        print(f"No 'CollectionX' folders found in {challenge_1b_dir}. Exiting.")
        return

    for collection_path in sorted(collection_dirs):
        collection_name = os.path.basename(collection_path)
        print(f"\n--- Processing {collection_name} ---")

        pdf_input_dir = os.path.join(collection_path, "PDFs")
        json_input_file = os.path.join(collection_path, "challenge1b_input.json")
        output_dir = collection_path # Output JSON will be in the collection folder

        os.makedirs(pdf_input_dir, exist_ok=True) # Ensure PDF input directory exists
        os.makedirs(output_dir, exist_ok=True)    # Ensure output directory exists

        if not os.path.exists(json_input_file):
            print(f"Error: {json_input_file} not found. Skipping {collection_name}.")
            continue

        try:
            with open(json_input_file, 'r') as f:
                input_data = json.load(f)
            persona_file = input_data.get("persona")
            job_to_be_done = input_data.get("job_to_be_done")

            if not job_to_be_done:
                print(f"Error: 'job_to_be_done' not found in {json_input_file}. Skipping {collection_name}.")
                continue
            if not persona_file:
                print(f"Error: 'persona' not found in {json_input_file}. Skipping {collection_name}.")
                continue

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {json_input_file}. Skipping {collection_name}.")
            continue
        except KeyError as e:
            print(f"Error: Missing key in {json_input_file}: {e}. Skipping {collection_name}.")
            continue

        pdf_files = [f for f in os.listdir(pdf_input_dir) if f.lower().endswith(".pdf")]

        if not pdf_files:
            print(f"No PDF files found in {pdf_input_dir} for {collection_name}.")
            continue

        pdf_paths_full = [os.path.join(pdf_input_dir, f) for f in pdf_files]

        print(f"Executing Round 1B: Persona-Driven Document Analysis for PDFs in {pdf_input_dir}")

        extractor = OutlineExtractorEnhanced(enable_profiling=False) # Profiling disabled as per reduced scope
        analyst = DocumentAnalystEnhanced(persona_file, job_to_be_done)
        print("Using Enhanced Document Analyst with Enhanced Outline Extractor")

        print(f"Analyzing {len(pdf_paths_full)} documents for {collection_name}...")
        start_time = time.time()
        results = analyst.analyze_documents(pdf_paths_full, extractor)
        total_time = time.time() - start_time

        print(f"Total analysis time for {collection_name}: {total_time:.2f} seconds")

        output_filename = f"{collection_name}_analysis_enhanced.json"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Analysis saved to {output_path}")

        # Print summary
        if hasattr(analyst, 'get_analysis_summary'):
            summary = analyst.get_analysis_summary(results)
            print("Analysis Summary:")
            print(f"  Total sections analyzed: {summary['total_sections_analyzed']}")
            print(f"  Processing time: {summary['processing_time']:.2f} seconds")
            print(f"  Persona role: {summary['persona_role']}")
            print(f"  Top 3 relevant sections:")
            # Ensure top_5_sections is handled gracefully if fewer than 3 sections exist
            for i, section in enumerate(summary.get('top_5_sections', [])[:3]):
                print(f"    {i+1}. {section['title']} (Score: {section['relevance_score']})")

if __name__ == "__main__":
    main()
    sys.exit(0)