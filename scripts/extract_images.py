import os
import glob
from pathlib import Path
from dotenv import load_dotenv
from google import genai
import time
import PIL.Image

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

images_dir = r"C:\Users\RADHAKRISHNA\Downloads\ilovepdf_pages-to-jpg"
output_file = Path(__file__).parent.parent / "data" / "raw_page.md"
output_file.parent.mkdir(parents=True, exist_ok=True)

image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
print(f"Found {len(image_paths)} images.")

prompt = """
You are a highly accurate OCR system. 
Please transcribe the following page of a document into clean Markdown format.
Preserve the headers (using ## or ###) and paragraphs. 
Do not add any conversational text, just output the transcribed markdown content.
"""

with open(output_file, "w", encoding="utf-8") as f:
    for i, p in enumerate(image_paths):
        print(f"Processing page {i+1}/{len(image_paths)}: {os.path.basename(p)}")
        img = PIL.Image.open(p)
        try:
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=[prompt, img]
            )
            f.write(response.text + "\n\n")
            print("Success.")
            time.sleep(2)  # avoid rate limit
        except Exception as e:
            print(f"Error on page {i+1}: {e}")
            time.sleep(10)
            # retry once
            try:
                response = client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=[prompt, img]
                )
                f.write(response.text + "\n\n")
                print("Success on retry.")
            except Exception as e2:
                print(f"Failed again: {e2}")

print(f"Done. Saved to {output_file}")
