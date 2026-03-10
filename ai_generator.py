"""
AI Generator — uses OpenAI to create product listings following the rules document.
Also generates product images via DALL-E.
"""

import os
import json
import base64
import requests
from openai import OpenAI
from google import genai

client = None
gemini_client = None

def init_openai():
    global client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def init_gemini():
    global gemini_client
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


LISTING_SYSTEM_PROMPT = """You are a Shopify product listing expert for a UK industrial store.
You create professional product listings from scraped product data.

STRICT RULES:
1. TITLE: Main Keyword – Size (if needed) – Colour – Important Detail – Use Case
   If multiple sizes/colours exist as variants, don't put them in title.

2. DESCRIPTION in this exact order:
   - Short Description (3-4 lines, NO heading above it — just the text directly, no "Short Description" heading)
   - Specifications (with heading)
   - Features (with heading)
   - Benefits (with heading)
   - Use Cases (with heading)
   - FAQ (2-3 Q&As, last section)

3. SPECIFICATIONS: Extract ALL technical data from the source. Rewrite in your own words. Include ALL measurements. Use structured format.

4. PRODUCT TYPE: Must match the collection name exactly.

5. TAGS: Include collection name, original product title, website name (without .com/.co.uk)

6. VARIANTS: Include ALL variants exactly as shown — sizes, colours, dimensions. Do NOT miss any. Do NOT guess.

7. PRICING: Selling price = reference price (inc VAT) × 2

8. INVENTORY: Always disabled.

9. DO NOT INCLUDE: Brand names, website names, emails, phones, shipping info, links, image URLs.

OUTPUT FORMAT — Return valid JSON:
{
  "title": "Product Title Following The Rule",
  "body_html": "<p>Short description text here without any heading...</p><h3>Specifications</h3><p>...</p><h3>Features</h3><ul>...</ul><h3>Benefits</h3><p>...</p><h3>Use Cases</h3><p>...</p><h3>FAQ</h3><p>...</p>",
  "product_type": "Collection Name",
  "tags": ["tag1", "tag2", "tag3"],
  "variants": [
    {
      "option1": "Size/Option Value",
      "price": "29.99",
      "sku": "",
      "inventory_management": null,
      "inventory_policy": "continue"
    }
  ],
  "options": [
    {
      "name": "Size",
      "values": ["Value1", "Value2"]
    }
  ],
  "image_prompt_1": "Detailed prompt for primary product image...",
  "image_prompt_2": "Detailed prompt for use-case image..."
}

IMPORTANT:
- All prices must be 2x the reference price (inc VAT)
- body_html must be valid HTML
- Include ALL variants found in the data
- If only one variant/size, still create one variant entry
- Generate detailed image prompts following these STRICT image rules:

IMAGE PROMPT 1 (PRIMARY — Real-Life Application):
- Show the product in REAL USE in an authentic location (warehouse, parking lot, industrial site, workshop, etc.)
- Product must dominate 50-70% of the frame
- 1000x1000 px (1:1 aspect ratio)
- NO installation scenes, NO hands installing, NO tools, NO setup process
- Preserve exact product details: materials, colors, finishes, dimensions, proportions, structural elements, textures
- No text, labels, captions, or markings on image
- No studio backgrounds — must be authentic real-world environment
- Sharp focus on product, slightly softer background, realistic professional lighting
- No logos, no branding, no company names
- SMALL PRODUCT RULE (under 10cm): Use macro close-up, product fills 70-85% of frame, soft blurred background
- ACCESSORY RULE: If product is small add-on/accessory, describe a split image — main use-case shot + macro detail panel showing texture/edges/finish
- People only if needed for scale — small, non-distracting, natural, NOT installing

IMAGE PROMPT 2 (USE CASE — Real-World Application):
- Show product ACTIVELY performing its intended purpose
- 1000x1000 px (1:1 aspect ratio)
- Documentary-style photography, medium or wide framing
- Demonstrate what the product DOES (e.g., speed bump with car driving over, bollard restricting entry, rack with items stored)
- Viewer should instantly understand: what it does, why it's useful, how it improves workflow/safety
- Must be real industrial/workplace environment, active and natural
- People ENCOURAGED — natural interaction, correct attire, must not block product
- No installation or setup scenes
- No logos, no branding, no text overlays
- Realistic environmental lighting, no staged studio look
"""


def generate_listing(product_data, collection_name, website_name):
    """Generate a Shopify product listing from scraped product data."""
    if not client:
        init_openai()

    # Build context from scraped data
    context = f"""
COLLECTION NAME: {collection_name}
WEBSITE NAME: {website_name}
PRODUCT TITLE: {product_data.get('title', 'Unknown')}
PRODUCT URL: {product_data.get('url', '')}

PRICES FOUND: {json.dumps(product_data.get('prices', []))}

DESCRIPTION FROM PAGE:
{product_data.get('description', 'No description found')[:5000]}

SPECIFICATION TABLES:
{json.dumps(product_data.get('tables', []), indent=2)[:5000]}

VARIANTS/OPTIONS FOUND:
{json.dumps(product_data.get('variants', []), indent=2)[:3000]}

BREADCRUMBS: {' > '.join(product_data.get('breadcrumbs', []))}

FULL PAGE TEXT (for additional context):
{product_data.get('full_text', '')[:8000]}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": LISTING_SYSTEM_PROMPT},
            {"role": "user", "content": f"Create a Shopify product listing from this scraped product data:\n\n{context}"}
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
        max_tokens=4000,
    )

    result = json.loads(response.choices[0].message.content)
    return result


def download_reference_images(image_urls, max_images=3):
    """Download reference images from supplier website for Gemini to analyze."""
    ref_images = []
    for url in image_urls[:max_images]:
        try:
            resp = requests.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }, timeout=15)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "image/jpeg")
            if "png" in content_type:
                mime = "image/png"
            elif "webp" in content_type:
                mime = "image/webp"
            else:
                mime = "image/jpeg"
            ref_images.append({"data": resp.content, "mime": mime})
        except Exception:
            continue
    return ref_images


def generate_product_image(prompt, reference_images=None):
    """Generate a product image using Gemini, with reference images for accuracy."""
    if not gemini_client:
        init_gemini()

    # Build content parts: reference images + text prompt
    contents = []
    
    if reference_images:
        for ref in reference_images:
            contents.append(genai.types.Part(
                inline_data=genai.types.Blob(
                    mime_type=ref["mime"],
                    data=ref["data"],
                )
            ))
        contents.append(genai.types.Part(
            text=f"""Above are the REFERENCE IMAGES of the actual product from the supplier website.
Study them carefully — the product's exact shape, color, material, texture, proportions, and all details.

Now generate a NEW image based on these rules:

{prompt}

CRITICAL: The product in your generated image MUST look exactly like the reference images above — same shape, same color, same material, same proportions. Do NOT simplify or change any detail."""
        ))
    else:
        contents.append(genai.types.Part(text=prompt))

    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",
        contents=contents,
        config=genai.types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    # Extract image from response
    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            return part.inline_data.data
    
    raise Exception("No image generated by Imagen")


def generate_images_for_product(listing_data, product_data=None, log_callback=None):
    """Generate both product images using reference images from supplier."""
    images = []
    
    if not os.getenv("GEMINI_API_KEY"):
        if log_callback:
            log_callback("⚠️ GEMINI_API_KEY not set — skipping image generation")
        return images
    
    # Download reference images from supplier
    ref_images = []
    source_images = (product_data or {}).get("images", [])
    if source_images:
        if log_callback:
            log_callback(f"📷 Downloading {min(len(source_images), 3)} reference images from supplier...")
        ref_images = download_reference_images(source_images, max_images=3)
        if log_callback:
            log_callback(f"✅ Downloaded {len(ref_images)} reference images")
    else:
        if log_callback:
            log_callback("⚠️ No reference images found from supplier")
    
    prompt1 = listing_data.get("image_prompt_1", "")
    prompt2 = listing_data.get("image_prompt_2", "")
    
    if prompt1:
        try:
            if log_callback:
                log_callback("🎨 Generating image 1 (primary) with Gemini + reference images...")
            img1 = generate_product_image(prompt1, ref_images)
            images.append({"data": img1, "filename": "product-main.png", "type": "primary"})
            if log_callback:
                log_callback("✅ Image 1 generated")
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Image 1 failed: {e}")
    else:
        if log_callback:
            log_callback("⚠️ No image_prompt_1 in AI response — skipping image 1")
    
    if prompt2:
        try:
            if log_callback:
                log_callback("🎨 Generating image 2 (use-case) with Gemini + reference images...")
            img2 = generate_product_image(prompt2, ref_images)
            images.append({"data": img2, "filename": "product-usecase.png", "type": "use-case"})
            if log_callback:
                log_callback("✅ Image 2 generated")
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Image 2 failed: {e}")
    else:
        if log_callback:
            log_callback("⚠️ No image_prompt_2 in AI response — skipping image 2")
    
    return images
