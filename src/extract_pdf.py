import re
from pathlib import Path
from pypdf import PdfReader

pdf_path = Path(r"C:\Users\RADHAKRISHNA\Downloads\Press Release Page _ Press Information Bureau.pdf")
output_file = Path(__file__).parent.parent / "data" / "raw_page.md"
output_file.parent.mkdir(parents=True, exist_ok=True)

reader = PdfReader(pdf_path)
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

# Clean up newlines and add markdown headers to known sections
lines = text.split('\n')
out_lines = []

headers = [
    "Towards Universal Health Coverage",
    "Ayushman Bharat: Universal Health Coverage for Every Citizen",
    "Pillar 1: Public Health Insurance Through Ayushman Bharat",
    "Ayushman App",
    "Pillar 2: Primary Care Through Ayushman Arogya Mandirs (AAM)",
    "Improving Public Health at the Grassroot Level",
    "Pillar 3: Pandemic Preparedness Through PM-ABHIM",
    "Pillar 4: Digital Health Ecosystem Through Ayushman Bharat Digital Mission (ABDM)",
    "National Health Mission: Targeting Myriad Ailments and Diseases",
    "Maternal and Child Healthcare under National Health Mission",
    "Community Health Workers Ensure Safe Institutional Births",
    "Maternal Health",
    "Child Health",
    "Nutrition and Adolescent Health",
    "Mission Indradhanush",
    "U-WIN",
    "Eliminating Communicable Diseases",
    "Tuberculosis",
    "Malaria",
    "Other Communicable Diseases",
    "COVID-19 and Pandemic Response",
    "Prevention and Treatment of Non-Communicable Diseases",
    "Early Detection and Screening",
    "Treatment and Care",
    "Cancer Care",
    "Kidney Disease and Dialysis",
    "Prevention: Reducing NCD Risk Factors",
    "Eat Right India",
    "Fit India",
    "Tobacco Control",
    "Providing Affordable Medicines and Emergency Transport",
    "Free Diagnostics Initiative",
    "Reaching People Where They Are: Digital and Last-Mile Health Services",
    "eSanjeevani: National Telemedicine Service",
    "Tele-MANAS: Mental Health on the Phone",
    "i-DRONE for Medicine Delivery",
    "Transforming Healthcare Delivery Through Artificial Intelligence",
    "Medical Education and Workforce",
    "Alternative Healthcare",
    "Towards a Viksit Bharat 2047",
    "References",
]

for line in lines:
    cleaned = line.strip()
    if not cleaned:
        continue
        
    is_header = False
    for h in headers:
        if cleaned.startswith(h):
            if h == "References":
                # Stop processing when reaching the references section
                break
            out_lines.append(f"\n## {cleaned}\n")
            is_header = True
            break
            
    if cleaned.startswith("References"):
        break
            
    if not is_header:
        out_lines.append(cleaned)

with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))

print(f"Extracted PDF text to {output_file}")
