#!/usr/bin/env python3
"""
Extract text from images using Ollama Qwen3-VL
Processes all images in the Images folder
"""

import json
from pathlib import Path
from datetime import datetime
import ollama


def extract_text_from_image(image_path: str, model: str = "qwen3-vl:32b"):
    """Extract all text from an image using vision model"""
    
    print(f"\n🖼️  Analyzing image: {image_path}")
    print(f"🤖 Using model: {model}")
    
    # Query the vision model with verbose output
    response = ollama.chat(
        model=model,
        messages=[
            {
                'role': 'user',
                'content': 'Extract all text from this image. Provide the complete transcription of any text visible in the image.',
                'images': [image_path]
            }
        ],
        options={
            'num_predict': -1,  # No limit on tokens
        }
    )
    
    extracted_text = response['message']['content']
    
    # Extract token information
    token_info = {
        'prompt_eval_count': response.get('prompt_eval_count', 0),
        'eval_count': response.get('eval_count', 0),
        'total_duration': response.get('total_duration', 0),
        'load_duration': response.get('load_duration', 0),
        'prompt_eval_duration': response.get('prompt_eval_duration', 0),
        'eval_duration': response.get('eval_duration', 0),
    }
    
    print(f"✅ Text extracted successfully!")
    print(f"📏 Length: {len(extracted_text)} characters")
    print(f"🔢 Tokens - Prompt: {token_info['prompt_eval_count']}, Response: {token_info['eval_count']}")
    
    return extracted_text, token_info


def save_result(image_path: str, model: str, extracted_text: str, token_info: dict):
    """Save extraction result to query_logs folder"""
    
    # Create logs directory structure
    logs_dir = Path("query_logs") / model.replace(":", "_")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_name = Path(image_path).stem
    log_file = logs_dir / f"image_{image_name}_{timestamp}.json"
    
    # Prepare log data
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "task": "text_extraction",
        "image_path": str(image_path),
        "extracted_text": extracted_text,
        "metadata": {
            "text_length": len(extracted_text),
            "image_filename": Path(image_path).name
        },
        "token_details": {
            "prompt_tokens": token_info['prompt_eval_count'],
            "response_tokens": token_info['eval_count'],
            "total_tokens": token_info['prompt_eval_count'] + token_info['eval_count'],
            "timing": {
                "total_duration_ns": token_info['total_duration'],
                "load_duration_ns": token_info['load_duration'],
                "prompt_eval_duration_ns": token_info['prompt_eval_duration'],
                "eval_duration_ns": token_info['eval_duration'],
                "total_duration_sec": token_info['total_duration'] / 1e9,
                "eval_duration_sec": token_info['eval_duration'] / 1e9,
            }
        }
    }
    
    # Save to JSON
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    print(f"📝 Saved to: {log_file}")
    return log_file


def main():
    images_folder = Path("Images")
    model = "qwen3-vl:32b"
    
    # Check if images folder exists
    if not images_folder.exists():
        print(f"❌ Error: '{images_folder}' folder not found!")
        return
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    image_files = [f for f in images_folder.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"❌ No image files found in '{images_folder}'!")
        return
    
    print(f"📁 Found {len(image_files)} images to process")
    print("="*60)
    
    results = []
    for idx, image_path in enumerate(image_files, 1):
        try:
            print(f"\n[{idx}/{len(image_files)}]")
            
            # Extract text
            extracted_text, token_info = extract_text_from_image(str(image_path), model)
            
            # Save result
            log_file = save_result(image_path, model, extracted_text, token_info)
            
            results.append({
                'image': image_path.name,
                'status': 'success',
                'log_file': str(log_file)
            })
            
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {str(e)}")
            results.append({
                'image': image_path.name,
                'status': 'error',
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'error')
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(results)}")
    print("="*60)


if __name__ == "__main__":
    main()
