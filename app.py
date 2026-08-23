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

def get_image_path(image_id):
    for filename in os.listdir("."):
        if filename.lower().startswith(f"{image_id}.") and filename.split('.')[-1].lower() in ["jpg", "jpeg", "png", "webp"]:
            return filename
    return None

def compress_image_to_base64(img_path):
    """Compresses image to 512px and converts to base64"""
    try:
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=65)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except:
        return None

def image_to_html(filepath, caption):
    """Reads pre-processed image, converts to base64, and wraps in HTML to display BEFORE text"""
    if not filepath or not os.path.exists(filepath): return ""
    try:
        with open(filepath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        return f"""
        <div style="display:inline-block; width:32%; vertical-align:top; margin-right:1%;">
            <p style="font-weight:bold; font-size:14px; margin-bottom:5px;">{caption}</p>
            <img src="data:image/jpeg;base64,{b64}" style="width:100%; border-radius:8px; border:1px solid #444;">
        </div>
        """
    except:
        return ""

def answer_question(user_text, history):
    if df.empty: return "Error loading data."
    
    user_text = user_text.strip()
    nums = re.findall(r'\b(\d+)\b', user_text)
    requested_ids = [int(n) for n in nums if 1 <= int(n) <= 156]
    requested_ids = list(dict.fromkeys(requested_ids)) # Remove duplicates

    # SCENARIO A: Multiple Artworks Requested (e.g., "Compare 6 and 13")
    if len(requested_ids) > 1:
        parts = [f"The user asked: '{user_text}'. Here are the requested artworks for you to compare/analyze:"]
        html_images = "<div style='margin-bottom: 20px;'>"
        
        for art_id in requested_ids:
            match_df = df[df['ID'].astype(str).str.strip() == str(art_id)]
            if not match_df.empty:
                row = match_df.iloc[0]
                title = str(row.get('TITLE', 'Unknown'))
                orig_path = get_image_path(art_id)
                img_b64 = compress_image_to_base64(orig_path) if orig_path else None
                
                if img_b64:
                    parts.append(f"Artwork ID {art_id} ({title}):")
                    parts.append(types.Part.from_bytes(data=base64.b64decode(img_b64), mime_type="image/jpeg"))
                    html_images += image_to_html(orig_path, f"Artwork {art_id}")
        html_images += "</div><br>"

        prompt = f"""
        CRITICAL RULES:
        1. YOU MUST NOT HALLUCINATE. Use only the provided data and images.
        2. Provide EXACTLY FOUR concise paragraphs (about 80 words each). 
        3. Compare the artworks based on their visual composition, color, and historical context.
        """
        try:
            res = client.models.generate_content(model="gemini-3.6-flash", contents=parts + [prompt])
            res_text = res.text
            return f"{html_images}\n\n{res_text}"
        except Exception as e:
            return f"Error comparing artworks: {str(e)}"

    # SCENARIO B: No number provided (Handles BOTH Follow-ups AND General Questions)
    if not requested_ids:
        if not user_text: return "Please enter a number 1-156."
        csv_data = df.to_string(index=False)
        is_followup = False
        if history:
            matches = re.findall(r'Artwork ID (\d+)', str(history))
            if matches:
                requested_ids = [int(matches[-1])]
                is_followup = True

        if is_followup:
            art_id = requested_ids[0]
            row = df[df['ID'].astype(str).str.strip() == str(art_id)].iloc[0]
            title = str(row.get('TITLE', 'Unknown'))
            orig_path = get_image_path(art_id)
            img_b64 = compress_image_to_base64(orig_path) if orig_path else None

            prompt = f"""
            The user asked: "{user_text}"
            You are currently analyzing Artwork ID {art_id} ({title}). The image is attached.
            The full database metadata for all 156 artworks is provided below:
            {csv_data}
            INSTRUCTIONS:
            - If the user is asking a follow-up question or challenging your previous analysis about Artwork {art_id}, respond conversationally in under 150 words based on the attached image.
            - If the user is asking a GENERAL question (e.g., "which ones have boats or animals?"), IGNORE the attached image and answer their question by searching the database metadata provided above. List the Artwork IDs and Titles.
            - YOU MUST NOT HALLUCINATE. Use only the provided data.
            """
            try:
                contents = [prompt]
                if img_b64:
                    contents.append(types.Part.from_bytes(data=base64.b64decode(img_b64), mime_type="image/jpeg"))
                res = client.models.generate_content(model="gemini-3.6-flash", contents=contents)
                return res.text
            except Exception as e:
                return f"Error: {str(e)}"
        else:
            prompt = f"""
            The user asked: "{user_text}"
            Here is the archival database metadata for all 156 artworks:
            {csv_data}
            Instructions: Answer the user's question using ONLY the database metadata provided above. List Artwork IDs and Titles. DO NOT HALLUCINATE.
            """
            res = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
            return res.text

    # SCENARIO C: Fresh request for a Single Artwork
    art_id = requested_ids[0]
    row = df[df['ID'].astype(str).str.strip() == str(art_id)].iloc[0]
    title = str(row.get('TITLE', 'Unknown'))
    artist = str(row.get('Artist (if known)', 'Unknown'))
    date = str(row.get('Date', 'Unknown'))
    style = str(row.get('Artistic style', 'Unknown'))
    source = str(row.get('Source', 'N/A'))
    csv_context = f"Title: {title}\nArtist: {artist}\nDate: {date}\nStyle: {style}\nSource: {source}"
    
    orig_path = get_image_path(art_id)
    seg_path = f"preloaded_segments/seg_{art_id}.jpg"
    col_path = f"preloaded_colors/colors_{art_id}.jpg"
    img_b64 = compress_image_to_base64(orig_path) if orig_path else None

    prompt = f"""
    You are a strict, analytical art historian. You are analyzing Artwork ID {art_id}.
    Here is the EXACT archival data for this artwork:
    {csv_context}

    CRITICAL RULES (STRICTLY ENFORCED):
    1. YOU MUST NOT HALLUCINATE. Do not use any outside knowledge. If the archival data says 'Unknown' for the Artist, you MUST state "Artist Unknown". Do not invent names, dates, or historical facts not present in the archival data.
    2. YOUR RESPONSE MUST BE EXACTLY FOUR PARAGRAPHS. EACH PARAGRAPH MUST BE CONCISE (about 80 words each). Total ~300 words.
    3. Do not provide generic introductions or conclusions. Go directly into deep, analytical prose.

    PARAGRAPH STRUCTURE AND REQUIREMENTS:
    - Paragraph 1 (Archival & Contextual Analysis): Analyze the artwork using ONLY the archival data provided above.
    - Paragraph 2 (Visual Analysis): Analyze the visual composition of the attached image. Discuss mood, lighting, brushwork, and materiality.
    - Paragraph 3 (Urban & Environmental Context): Relate the artwork to the urban and environmental history of Adelaide based strictly on visual evidence and archival date.
    - Paragraph 4 (Semantic Segmentation Analysis): Conduct a rigorous textual analysis of how a semantic segmentation algorithm would break down this image. Discuss distinct spatial regions, boundaries, and color fields.
    """
    
    try:
        contents = [prompt]
        if img_b64:
            contents.append(types.Part.from_bytes(data=base64.b64decode(img_b64), mime_type="image/jpeg"))
        res = client.models.generate_content(model="gemini-3.6-flash", contents=contents)
        res_text = res.text

        # Build the HTML to display the 3 images BEFORE the text
        html_images = "<div style='margin-bottom: 20px; overflow: hidden;'>"
        html_images += image_to_html(orig_path, "Original Artwork")
        html_images += image_to_html(seg_path, "Semantic Segmentation")
        html_images += image_to_html(col_path, "Dominant Color Palette")
        html_images += "</div><br>"

        return f"{html_images}\n\n{res_text}"
    except Exception as e:
        return f"Error generating response: {str(e)}"

# multimodal=False (no upload button), type="messages" removed for max stability
demo = gr.ChatInterface(fn=answer_question, title="Adelaide Artworks AI (1-156)")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
