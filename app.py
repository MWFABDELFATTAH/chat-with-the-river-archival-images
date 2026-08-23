import os
import pandas as pd
import gradio as gr
import re
from google import genai
from google.genai import types
from PIL import Image
import io
import base64

# 1. Setup Google Gemini
client = genai.Client(api_key=os.environ.get("gemini_API_KEY"))

# 2. Load Excel Data
try:
    df = pd.read_excel("data.xlsx")
    df.columns = df.columns.str.strip()
except Exception as e:
    df = pd.DataFrame()

def get_image_path(archive_id):
    for filename in os.listdir("."):
        if filename.lower().startswith(f"{archive_id}.") and filename.split('.')[-1].lower() in ["jpg", "jpeg", "png", "webp"]:
            return filename
    return None

def compress_image_to_base64(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=65)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except:
        return None

def get_markdown_images(archive_id):
    """Converts the 3 images into Markdown image tags so they display BEFORE the text"""
    orig_path = get_image_path(archive_id)
    seg_path = f"seg_{archive_id}.jpg"
    col_path = f"colors_{archive_id}.jpg"
    
    md = ""
    if orig_path and os.path.exists(orig_path):
        with open(orig_path, "rb") as f: b64 = base64.b64encode(f.read()).decode('utf-8')
        md += f"![Original Archive](data:image/jpeg;base64,{b64})\n"
    if os.path.exists(seg_path):
        with open(seg_path, "rb") as f: b64 = base64.b64encode(f.read()).decode('utf-8')
        md += f"![Semantic Segmentation](data:image/jpeg;base64,{b64})\n"
    if os.path.exists(col_path):
        with open(col_path, "rb") as f: b64 = base64.b64encode(f.read()).decode('utf-8')
        md += f"![Dominant Colour Palette](data:image/jpeg;base64,{b64})\n"
        
    return md + "\n"

def answer_question(user_text, history):
    if df.empty: return "Error loading data."
    
    user_text = user_text.strip()
    nums = re.findall(r'\b(\d+)\b', user_text)
    requested_ids = [int(n) for n in nums if 1 <= int(n) <= 156]
    requested_ids = list(dict.fromkeys(requested_ids)) # Remove duplicates

    # SCENARIO A: Multiple Archives Requested (e.g., "Compare 6 and 13")
    if len(requested_ids) > 1:
        parts = [f"The user asked: '{user_text}'. Here are the requested archives for you to compare/analyze:"]
        md_images = ""
        
        for archive_id in requested_ids:
            match_df = df[df['ID'].astype(str).str.strip() == str(archive_id)]
            if not match_df.empty:
                row = match_df.iloc[0]
                title = str(row.get('TITLE', 'Unknown'))
                orig_path = get_image_path(archive_id)
                img_b64 = compress_image_to_base64(orig_path) if orig_path else None
                
                if img_b64:
                    parts.append(f"Archive ID {archive_id} ({title}):")
                    parts.append(types.Part.from_bytes(data=base64.b64decode(img_b64), mime_type="image/jpeg"))
                    md_images += f"![Archive {archive_id}](data:image/jpeg;base64,{img_b64})\n"

        prompt = f"""
        CRITICAL RULES:
        1. YOU MUST NOT HALLUCINATE. Use only the provided data and images.
        2. Provide EXACTLY FOUR concise paragraphs (about 80 words each).
        3. Your response must be STRAIGHTFORWARD, SYSTEMATIC, AND BENEFICIAL. STRICTLY NO REPETITION.
        4. Compare the archives based on their visual composition, colour, and historical context.
        """
        try:
            res = client.models.generate_content(model="gemini-3.6-flash", contents=parts + [prompt])
            res_text = res.text
            return f"{md_images}\n{res_text}"
        except Exception as e:
            return f"Error comparing archives: {str(e)}"

    # SCENARIO B: No number provided (Handles BOTH Follow-ups AND General Questions)
    if not requested_ids:
        if not user_text: return "Please enter a number 1-156."
        csv_data = df.to_string(index=False)
        is_followup = False
        if history:
            matches = re.findall(r'Archive ID (\d+)', str(history))
            if matches:
                requested_ids = [int(matches[-1])]
                is_followup = True

        if is_followup:
            archive_id = requested_ids[0]
            row = df[df['ID'].astype(str).str.strip() == str(archive_id)].iloc[0]
            title = str(row.get('TITLE', 'Unknown'))
            orig_path = get_image_path(archive_id)
            img_b64 = compress_image_to_base64(orig_path) if orig_path else None

            prompt = f"""
            The user asked: "{user_text}"
            You are currently analyzing Archive ID {archive_id} ({title}). The image is attached.
            The full database metadata for all 156 archives is provided below:
            {csv_data}
            INSTRUCTIONS:
            - If the user is asking a follow-up question or challenging your previous analysis about Archive {archive_id}, respond conversationally in under 150 words based on the attached image. Be STRAIGHTFORWARD and SYSTEMATIC.
            - If the user is asking a GENERAL question (e.g., "which ones have boats or animals?"), IGNORE the attached image and answer their question by searching the database metadata provided above. List the Archive IDs and Titles.
            - YOU MUST NOT HALLUCINATE. Use only the provided data.
            """
            try:
                contents = [prompt]
                if img_b64:
                    contents.append(types.Part.from_bytes(data=base64.b64decode(img_b64), mime_type="image/jpeg"))
                res = client.models.generate_content(model="gemini-3.6-flash", contents=contents)
                
                # Return 3 images for follow-up queries too
                md_images = get_markdown_images(archive_id)
                return f"{md_images}\n{res.text}"
            except Exception as e:
                return f"Error: {str(e)}"
        else:
            prompt = f"""
            The user asked: "{user_text}"
            Here is the archival database metadata for all 156 archives:
            {csv_data}
            Instructions: Answer the user's question using ONLY the database metadata provided above. List Archive IDs and Titles. DO NOT HALLUCINATE.
            """
            res = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
            return res.text

    # SCENARIO C: Fresh request for a Single Archive
    archive_id = requested_ids[0]
    row = df[df['ID'].astype(str).str.strip() == str(archive_id)].iloc[0]
    title = str(row.get('TITLE', 'Unknown'))
    artist = str(row.get('Artist (if known)', 'Unknown'))
    date = str(row.get('Date', 'Unknown'))
    style = str(row.get('Artistic style', 'Unknown'))
    source = str(row.get('Source', 'N/A'))
    csv_context = f"Title: {title}\nCreator: {artist}\nDate: {date}\nStyle: {style}\nSource: {source}"
    
    md_images = get_markdown_images(archive_id)
    orig_path = get_image_path(archive_id)
    img_b64 = compress_image_to_base64(orig_path) if orig_path else None

    prompt = f"""
    You are a strict, analytical archivist and historian. You are analyzing Archive ID {archive_id}.
    Here is the EXACT archival data for this archive:
    {csv_context}

    CRITICAL RULES (STRICTLY ENFORCED):
    1. YOU MUST NOT HALLUCINATE. Do not use any outside knowledge. If the archival data says 'Unknown' for the Creator, you MUST state "Creator Unknown". Do not invent names, dates, or historical facts not present in the archival data.
    2. YOUR RESPONSE MUST BE EXACTLY FOUR PARAGRAPHS. EACH PARAGRAPH MUST BE CONCISE (about 80 words each). Total ~300 words.
    3. Your response must be STRAIGHTFORWARD, SYSTEMATIC, AND BENEFICIAL. STRICTLY NO REPETITION of phrases or concepts between paragraphs. Do not provide generic introductions or conclusions. Go directly into deep, analytical prose.

    PARAGRAPH STRUCTURE AND REQUIREMENTS:
    - Paragraph 1 (Archival & Contextual Analysis): Analyze the archive using ONLY the archival data provided above.
    - Paragraph 2 (Visual Analysis): Analyze the visual composition of the attached image. Discuss mood, lighting, brushwork, and materiality.
    - Paragraph 3 (Urban & Environmental Context): Relate the archive to the urban and environmental history of Adelaide based strictly on visual evidence and archival date.
    - Paragraph 4 (Semantic Segmentation Analysis): Conduct a rigorous textual analysis of how a semantic segmentation algorithm would break down this image. Discuss distinct spatial regions, boundaries, and colour fields.
    """
    
    try:
        contents = [prompt]
        if img_b64:
            contents.append(types.Part.from_bytes(data=base64.b64decode(img_b64), mime_type="image/jpeg"))
        res = client.models.generate_content(model="gemini-3.6-flash", contents=contents)
        res_text = res.text

        return f"{md_images}\n{res_text}"
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Removed type="messages" and multimodal=True to fix the TypeError and hide the upload button
demo = gr.ChatInterface(fn=answer_question, title="Adelaide Archives AI (1-156)")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
