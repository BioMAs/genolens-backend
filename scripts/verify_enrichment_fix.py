import asyncio
import os
import sys

# Add . to path
sys.path.append('.')

from app.services.data_processor import data_processor
from app.services.storage import storage_service

async def main():
    dataset_id = '9d8c246f-2fd5-4611-b83e-01a571e33948'
    raw_file_path = 'projects/2eff81e8-256a-4acf-9d1b-27dfc28b5aaf/raw/funct.analysis_STRATEGY1_data_p0.05_r3_onlyenriched.xlsx'
    
    print("Downloading file...")
    try:
        raw_data = await storage_service.download_file(raw_file_path)
    except Exception as e:
        print(f"Failed to download: {e}")
        return

    print("Converting to parquet...")
    try:
        parquet_data = await data_processor.convert_to_parquet(raw_data, '.xlsx')
    except Exception as e:
        print(f"Failed to convert: {e}")
        return
        
    print("Extracting metadata...")
    metadata = await data_processor.get_file_metadata(raw_data, '.xlsx')
    print(f"Metadata enrichment_comparisons: {metadata.get('enrichment_comparisons')}")
    
    if metadata.get('enrichment_comparisons'):
        print("Extracting enrichment pathways...")
        pathways = await data_processor.extract_enrichment_pathways_for_db(
            parquet_data,
            {comp: {} for comp in metadata["enrichment_comparisons"]}
        )
        
        for comp, paths in pathways.items():
            print(f"Comparison: {comp}, Pathways count: {len(paths)}")
            if len(paths) > 0:
                print(f"First pathway keys: {paths[0].keys()}")
                # Print just name and id to be concise
                print(f"First pathway ID: {paths[0].get('pathway_id')}, Name: {paths[0].get('pathway_name')}, padj: {paths[0].get('padj')}")
    else:
        print("No enrichment_comparisons found in metadata!")

if __name__ == "__main__":
    asyncio.run(main())
