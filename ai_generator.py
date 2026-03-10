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

5. TAGS: Include collection name, original product title (WITHOUT any brand names), and the reference website name (without .com/.co.uk). Website name in tags is OK — just no brand names in title/description.

6. VARIANTS: Include ALL variants exactly as shown — sizes, colours, dimensions. Do NOT miss any. Do NOT guess.

7. PRICING: Selling price = reference price (inc VAT) × 2

8. INVENTORY: Always disabled.

9. DO NOT INCLUDE (VERY IMPORTANT — STRICTLY FORBIDDEN):
   - ANY brand names from the reference website (manufacturer names, supplier names, store names)
   - The reference website name or domain
   - Emails, phone numbers, shipping details
   - Links, image URLs, download links
   - Do NOT mention the source/supplier brand ANYWHERE — not in title, description, tags, features, specs, or FAQ
   - If the original product title contains a brand name, REMOVE it and rewrite without it
   - Replace brand references with generic terms (e.g., "premium quality" instead of brand name)

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


def generate_product_image_openai(prompt, reference_images=None):
    """Generate a product image using OpenAI gpt-image-1.5 with reference images."""
    if not client:
        init_openai()

    full_prompt = f"""Look at the reference product images provided. Now generate a NEW image following these rules:

{prompt}

ABSOLUTE RULES — DO NOT BREAK ANY:

PRODUCT (MUST BE 100% IDENTICAL TO REFERENCE IMAGES — ZERO CHANGES):
- The product must look EXACTLY like the reference images — same color, same shape, same material, same finish.
- Every curve, edge, corner, hole, groove, ridge, fastener must match perfectly.
- Same proportions and dimensions — do NOT make it bigger, smaller, thicker, or thinner.
- Same surface texture — matte stays matte, glossy stays glossy, brushed stays brushed.
- This is the SAME physical product from the reference images, just in a different place.

WHAT MUST BE DIFFERENT (ONLY THESE):
- The ENVIRONMENT/LOCATION — completely different real-world setting than the reference.
- The USE CASE — different but realistic scenario showing the product in use.
- People, surroundings, background — all different from reference.

OTHER RULES:
- Professional photorealistic quality, natural lighting.
- No text, no logos, no branding, no labels on the image.
- IMAGE SIZE: exactly 1000 x 1000 pixels, 1:1 square aspect ratio."""

    # Build image inputs from reference images
    image_inputs = []
    if reference_images:
        for ref in reference_images:
            img_b64 = base64.b64encode(ref["data"]).decode("utf-8")
            image_inputs.append({
                "type": "input_image",
                "input_image": {
                    "image_data": img_b64,
                    "media_type": ref["mime"],
                },
            })

    # Add text prompt
    image_inputs.append({
        "type": "text",
        "text": full_prompt,
    })

    response = client.images.generate(
        model="gpt-image-1.5",
        prompt=image_inputs,
        size="1024x1024",
        quality="high",
        n=1,
    )

    # gpt-image returns base64 data
    image_b64 = response.data[0].b64_json
    if image_b64:
        return base64.b64decode(image_b64)

    # Fallback to URL
    image_url = response.data[0].url
    img_response = requests.get(image_url, timeout=60)
    img_response.raise_for_status()
    return img_response.content


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

CRITICAL: The product in your generated image MUST look exactly like the reference images above — same shape, same color, same material, same proportions. Do NOT simplify or change any detail.

IMAGE SIZE: The image MUST be exactly 1000 x 1000 pixels with 1:1 aspect ratio (square)."""
        ))
    else:
        contents.append(genai.types.Part(text=f"{prompt}\n\nIMAGE SIZE: The image MUST be exactly 1000 x 1000 pixels with 1:1 aspect ratio (square)."))

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-image",
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


def generate_images_for_product(listing_data, product_data=None, log_callback=None, image_provider="gemini"):
    """Generate both product images. image_provider: 'gemini' or 'openai'."""
    images = []
    
    if image_provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        if log_callback:
            log_callback("⚠️ GEMINI_API_KEY not set — skipping image generation")
        return images
    
    if image_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        if log_callback:
            log_callback("⚠️ OPENAI_API_KEY not set — skipping image generation")
        return images
    
    provider_name = "Gemini" if image_provider == "gemini" else "OpenAI"
    
    # Download reference images from supplier (for both Gemini and OpenAI)
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
    
    for idx, prompt in enumerate([prompt1, prompt2], 1):
        label = "primary" if idx == 1 else "use-case"
        if not prompt:
            if log_callback:
                log_callback(f"⚠️ No image_prompt_{idx} in AI response — skipping image {idx}")
            continue
        
        try:
            if log_callback:
                log_callback(f"🎨 Generating image {idx} ({label}) with {provider_name}...")
            
            if image_provider == "openai":
                img = generate_product_image_openai(prompt, ref_images)
            else:
                img = generate_product_image(prompt, ref_images)
            
            images.append({"data": img, "filename": f"product-{label}.png", "type": label})
            if log_callback:
                log_callback(f"✅ Image {idx} generated")
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Image {idx} failed: {e}")
    
    return images
