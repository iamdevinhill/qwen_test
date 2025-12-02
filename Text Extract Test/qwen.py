import fitz  # PyMuPDF
from PIL import Image
import ollama
import os


def pdf_page_to_png(pdf_path, page_num, output_png):
    """Convert a PDF page to PNG image file."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Render page to image at high resolution
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
    
    # Convert to PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Save as PNG
    img.save(output_png)
    
    doc.close()
    return output_png


def extract_text_from_pdf(pdf_path, output_path="output.txt"):
    """Extract text from PDF using Ollama qwen3-vl:8b model."""
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        return
    
    print(f"Processing PDF: {pdf_path}")
    
    # Open PDF to get page count
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    doc.close()
    
    print(f"Total pages: {num_pages}")
    
    all_text = []
    
    # Process each page
    for page_num in range(num_pages):
        print(f"\nProcessing page {page_num + 1}/{num_pages}...")
        
        # Convert page to PNG file
        png_path = f"page_{page_num + 1}.png"
        pdf_page_to_png(pdf_path, page_num, png_path)
        print(f"Saved {png_path}")
        
        # Call Ollama with qwen3-vl:8b model using the PNG file
        try:
            response = ollama.chat(
                model='qwen3-vl:8b',
                messages=[{
                    'role': 'user',
                    'content': 'Please perform OCR on this image and extract all visible text.',
                    'images': [png_path]
                }]
            )
            
            extracted_text = response['message']['content'].strip()
            
            # Debug: print response
            print(f"Response: {extracted_text[:200]}..." if len(extracted_text) > 200 else f"Response: {extracted_text}")
            
            if not extracted_text:
                print("Warning: No text extracted from this page")
                extracted_text = "[No text detected]"
            
            all_text.append(f"=== Page {page_num + 1} ===\n{extracted_text}\n")
            print(f"Extracted {len(extracted_text)} characters from page {page_num + 1}")
            
        except Exception as e:
            print(f"Error processing page {page_num + 1}: {e}")
            all_text.append(f"=== Page {page_num + 1} ===\n[Error extracting text]\n")
    
    # Save all text to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_text))
    
    print(f"\n✓ Text extraction complete!")
    print(f"✓ Saved to: {output_path}")
    print(f"✓ Total pages processed: {num_pages}")


if __name__ == "__main__":
    # Extract text from input.pdf and save to output.txt
    extract_text_from_pdf("input.pdf", "output.txt")
