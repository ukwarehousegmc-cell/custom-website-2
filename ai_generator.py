"""
AI Generator — uses OpenAI to create product listings following the rules document.
Also generates product images via DALL-E.
"""

import os
import io
import json
import base64
import requests
from openai import OpenAI
from PIL import Image as PILImage
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
1. TITLE: Must be SEO optimised with the main keyword FIRST. Format: Main Keyword – Size (if needed) – Colour – Important Detail – Use Case
   The most important searchable keyword must always come first in the title.
   If multiple sizes/colours exist as variants, don't put them in title.

2. DESCRIPTION in this exact order:
   - Short Description (3-4 lines, NO heading above it — just the text directly, no "Short Description" heading)
   - Specifications (with heading)
   - Features (with heading)
   - Benefits (with heading)
   - Use Cases (with heading)
   - FAQ (2-3 Q&As, last section)

3. SPECIFICATIONS: Extract ALL technical data from the source, especially from the "product-specs" section. Include ALL measurements. Present specifications as an HTML table format (<table> with rows). Rewrite in your own words but keep all values accurate.

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

IMAGE PROMPT 1 (PRIMARY — Product in Environment):
- Recreate the product exactly as it appears — same color, shape, size, texture, design, number of parts
- Place it in a natural real-world usage environment relevant to the product
- Product must be the main subject, dominating the frame
- 1000x1000 px (1:1 aspect ratio), high resolution
- Photorealistic, professional ecommerce product photography
- High detail, realistic lighting and shadows
- No text, no logos, no watermarks
- Maximum ONE person if needed — person is secondary, product is the star

IMAGE PROMPT 2 (SECOND IMAGE — Same rules as Image 1):
- SAME RULES as Image Prompt 1 — recreate the product exactly, same color, shape, size, texture, design, number of parts
- Place it in a DIFFERENT realistic environment than Image 1
- Product must be the main subject, dominating the frame
- 1000x1000 px (1:1 aspect ratio), high resolution
- Photorealistic, professional ecommerce product photography
- High detail, realistic lighting and shadows
- No text, no logos, no watermarks
- Maximum ONE person if needed — person is secondary, product is the star
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

PRODUCT SPECIFICATIONS (from product-specs section — MUST include ALL of these in Specifications as HTML table):
{json.dumps(product_data.get('product_specs', []), indent=2) if product_data.get('product_specs') else product_data.get('product_specs_text', 'None found')}

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

    full_prompt = f"""Use the provided reference image to recreate the SAME product.

{prompt}

CRITICAL RULE:
The product must remain IDENTICAL to the reference image.

Do NOT modify the product in any way.

STRICT PRODUCT PRESERVATION RULES:
- Keep the exact same color
- Keep the exact same shape
- Keep the exact same size and proportions
- Keep the exact same structure and design
- Keep the exact same number of parts, panels, holes, windows, patterns, screws, or segments
- Keep the same texture and materials

DO NOT:
- add extra parts
- remove any part
- change the design
- change the number of components
- redesign the product

HUMAN RULE:
- If a person is shown, only ONE human is allowed in the image.
- Do NOT include multiple people.
- The human should interact naturally with the product.
- The product must remain the main focus, not the person.

ENVIRONMENT RULES:
Only change the background or environment to a realistic real-world usage scenario relevant to the product.

STYLE:
photorealistic
professional ecommerce product photography
high detail
realistic lighting and shadows

IMAGE REQUIREMENTS:
square image
1000 x 1000 px
no text
no logo
no watermark"""

    # If reference images available, use chat completions with image input for better accuracy
    if reference_images:
        messages = [
            {"role": "system", "content": "You are generating product images for ecommerce listings. Use the provided reference image as the primary source. The product must remain IDENTICAL — same color, shape, size, proportions, structure, texture, materials. Do NOT modify, redesign, add or remove any part. Place in a realistic environment. Max ONE person. No text, logos, or watermarks."},
            {"role": "user", "content": []}
        ]
        
        # Add reference images
        for ref in reference_images:
            img_b64 = base64.b64encode(ref["data"]).decode("utf-8")
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:{ref['mime']};base64,{img_b64}"}
            })
        
        # Add text prompt
        messages[1]["content"].append({
            "type": "text",
            "text": full_prompt
        })
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=100,
        )
        
        # Extract image description from GPT-4o, then generate with gpt-image-1.5
        enhanced_prompt = response.choices[0].message.content + "\n\n" + full_prompt
        
        response = client.images.generate(
            model="gpt-image-1.5",
            prompt=enhanced_prompt[:4000],
            size="1024x1024",
            quality="high",
            n=1,
        )
    else:
        response = client.images.generate(
            model="gpt-image-1.5",
            prompt=full_prompt,
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


def _detect_scenario_type(product_title):
    """Detect whether product needs LIFESTYLE or INDUSTRIAL scenario."""
    product_lower = product_title.lower()
    is_furniture = any(kw in product_lower for kw in [
        'bench', 'chair', 'seat', 'table', 'sofa', 'couch', 'stool', 'furniture',
        'lounger', 'hammock', 'swing', 'gazebo', 'pergola', 'planter', 'pot'
    ])
    is_outdoor = any(kw in product_lower for kw in [
        'garden', 'outdoor', 'patio', 'deck', 'bbq', 'grill', 'fire pit',
        'umbrella', 'parasol', 'fountain', 'statue', 'ornament'
    ])
    return "LIFESTYLE" if (is_furniture or is_outdoor) else "INDUSTRIAL"


def _build_edit_prompt(product_title, variation="product_in_use", num_ref_images=1):
    """Build the comprehensive image editing prompt matching the proven gemini_service.py approach."""
    scenario_type = _detect_scenario_type(product_title)

    size_context = ""
    if num_ref_images > 1:
        size_context = f"""
📏 SIZE & SCALE CONTEXT:
- You have been provided with {num_ref_images} reference images of this product
- Analyze ALL images to understand the product's ACTUAL REAL-WORLD SIZE and proportions
- The product title "{product_title}" indicates the true nature and scale of this product
- PAY CLOSE ATTENTION to size indicators in the images (people, objects, measurements)
- Ensure the generated image shows the product at its CORRECT REAL-WORLD SCALE
- If the product is large (barriers, bollards, parking equipment, industrial items), show it at FULL SIZE
- If the product is small (tools, accessories), show it at appropriate human-scale

🎯 COMPLETE PRODUCT SETUP:
- Study ALL {num_ref_images} reference images to see the COMPLETE product setup
- If images show containers/tanks ON a pallet (like IBC Spill Pallets) → show WITH containers
- If images show items IN/ON racks or shelves → show WITH items stored
- If images show equipment WITH accessories or attachments → show COMPLETE assembly
- DO NOT generate just an empty base/frame if references show it loaded or complete
- The reference images show how the product is MEANT TO LOOK - replicate that exactly
"""

    # Variation-specific instructions
    if variation == "product_in_use":
        variation_instructions = """
📸 IMAGE 1 — REAL-LIFE APPLICATION WITH PRODUCT AS MAIN FOCUS (PRIMARY IMAGE)

🎯 OBJECTIVE:
Create a REAL-LIFE APPLICATION image showing the product EXACTLY as it appears in ALL reference images, with the product dominating 50–70% of the frame.
✅ Image must be 1000 × 1000 px (1:1 aspect ratio).

⚠️ CRITICAL - NO INSTALLATION SCENES:
❌ DO NOT show installation, setup, or assembly processes
❌ DO NOT show hands installing or tools installing
❌ DO NOT show workers setting up the product
✅ ONLY show the product in real use - already installed, already functioning, actively being used

🎯 WHAT TO SHOW (MANDATORY):
1. Analyze the product title and ALL reference images to understand the product's purpose, material, design, proportions, and functional details.
2. Show the product actively performing its real-world purpose in an authentic location.
3. The product must appear 100% IDENTICAL to references — no modifications, no missing features, no simplification.
4. Do NOT add any text, labels, captions, or markings on the image.

🎯 PRODUCT DETAIL REQUIREMENTS (MUST MATCH REFERENCES):
Preserve EVERY detail exactly:
- Exact materials (metal, plastic, rubber, composite, etc.)
- Exact colors & finishes (matte, glossy, textured, brushed, powder-coated, etc.)
- Exact dimensions & proportions relative to environment
- All structural elements (holes, grooves, ridges, fasteners, handles, brackets, bolts, clips, hinges)
- Surface texture and fine detailing (grain, welds, seams, edges, corners)
- Any technical or functional components (mechanisms, moving parts, connectors)

🔍 SMALL PRODUCT RULE (FOR PRODUCTS UNDER 10CM):
IF the product is SMALL (clips, screws, brackets, connectors, fasteners, hooks, small accessories):
✅ Use macro close-up photography
✅ Product fills 70–85% of frame
✅ Only ONE product in the frame (no multiples)
✅ Soft blurred background for depth

📸 COMPOSITION RULES:
- Product is the main subject
- Sharp focus on the product
- Background slightly softer
- Realistic lighting appropriate to environment
- Professional product photography style
- Product occupies 50–70% of frame (70–85% for small products)
"""
    else:
        variation_instructions = """
📸 IMAGE 2 — REAL-WORLD APPLICATION (USE CASE DEMONSTRATION)

🎯 OBJECTIVE:
Show the product actively performing its intended purpose in a real-world environment.
✅ Image size must be 1000 × 1000 px (1:1)
✅ Use-case only — NO installation scenes, NO setup, NO assembly

⚠️ CRITICAL - NO INSTALLATION ALLOWED:
❌ DO NOT show installation, setup, or assembly
❌ DO NOT show workers installing the product
❌ DO NOT show hands or tools setting up the product
✅ ONLY show: Product already installed, already in place, actively being USED

🎯 WHAT TO SHOW:
1. Demonstrate HOW and WHY the product is used in real scenarios
2. Show the exact environment where it is needed
3. Show the product in REAL ACTION (already installed, functioning)

💡 VALUE DEMONSTRATION:
The viewer should instantly understand:
✅ What the product does in practice
✅ Why it is useful and valuable
✅ How it improves workflow, safety, or organization

👥 PEOPLE (ENCOURAGED):
✅ Should be interacting naturally with the product IN USE
✅ Show people BENEFITING from the product
✅ Wearing correct attire (work clothes, safety gear if appropriate)
✅ Must NOT overshadow the product
❌ NO installation actions, NO setup activities

🎬 STYLE:
✅ Documentary-style photography showing "how it's really used"
✅ Medium or wide framing showing product IN CONTEXT
✅ Realistic environmental lighting
"""

    edit_prompt = f"""You are a professional lifestyle product photographer. Transform this product image into a compelling, real-world application photograph showing the product in use.

PRODUCT: {product_title}
{size_context}
{variation_instructions}

🎯 PHOTOGRAPHY OBJECTIVE:
Create a REALISTIC, professional photograph showing this product being used in its INTENDED REAL-WORLD APPLICATION.
The product SIZE and SCALE must be ACCURATE based on the product title and reference images provided.

⚠️ CRITICAL: PRESERVE THE EXACT PRODUCT APPEARANCE
🔒 PRODUCT INTEGRITY - MUST NOT CHANGE:
1. Keep the product's EXACT PHYSICAL DESIGN - do not alter shape, form, or structure
2. Preserve EXACT COLORS - maintain all original colors of the product precisely
3. Keep EXACT MATERIALS and textures - metal stays metal, plastic stays plastic, etc.
4. Maintain EXACT DIMENSIONS and proportions as shown in reference images
5. Keep ALL PHYSICAL FEATURES - buttons, grooves, edges, patterns exactly as they are
6. Do NOT redesign, modify, or "improve" the product in any way
7. The product must be IDENTICAL to the original - only remove text/logos/brands

✅ WHAT YOU CAN CHANGE:
- The ENVIRONMENT and background (add realistic workplace/lifestyle setting)
- The LIGHTING and photography angle
- Add PEOPLE interacting with the product (hands, workers, users)
- Add CONTEXT objects (tools, vehicles, other environmental items)
- The SCENARIO showing how the product is used

❌ WHAT YOU CANNOT CHANGE:
- The product's physical appearance, design, or features
- The product's colors or materials
- The product's size or proportions
- The product's shape or structure

🎯 RESULT: The SAME product in a NEW realistic environment/scenario

SCENARIO TYPE: {scenario_type}

👤 HUMAN INTERACTION:
{"LIFESTYLE SCENARIO - Natural, Relaxed Usage:" if scenario_type == "LIFESTYLE" else "ACTIVE USE SCENARIO - Installation/Operation:"}
{"- Show person naturally using or enjoying the product (sitting, relaxing, etc.)" if scenario_type == "LIFESTYLE" else "- Show professional worker actively using or operating the product"}
{"- Person dressed casually and comfortably for the setting" if scenario_type == "LIFESTYLE" else "- Person dressed appropriately (safety gear, work clothes, etc.)"}
{"- Natural, relaxed posture - enjoying the product" if scenario_type == "LIFESTYLE" else "- Focus on HANDS and product interaction - holding, operating"}
{"- Authentic lifestyle moment captured naturally" if scenario_type == "LIFESTYLE" else "- Natural, authentic body language and realistic usage posture"}

🏗️ ENVIRONMENT & SETTING:
{"LIFESTYLE SETTING - Beautiful, Natural Environment:" if scenario_type == "LIFESTYLE" else "WORKPLACE SETTING - Authentic Work Environment:"}
{"- Outdoor garden, patio, deck, backyard, or beautiful home setting" if scenario_type == "LIFESTYLE" else "- Job site, workshop, garage, construction area, or workplace"}
{"- Lush greenery, flowers, natural landscaping in background (softly blurred)" if scenario_type == "LIFESTYLE" else "- Work surfaces, tools, equipment, materials in background (blurred)"}
{"- Natural sunlight, golden hour lighting, or soft outdoor illumination" if scenario_type == "LIFESTYLE" else "- Workshop lighting, natural daylight, or work environment lighting"}

📸 PROFESSIONAL PHOTOGRAPHY QUALITY:
1. Photorealistic, looks like actual product photography
2. Natural lighting appropriate to the environment
3. Shallow depth of field - product and person in focus, background beautifully blurred
4. Professional color grading with authentic, natural tones
5. Camera angle: Eye-level or slightly above, showing product in perfect context

ENVIRONMENT VARIATION RULE (MANDATORY):
The environment in the generated image must be COMPLETELY DIFFERENT from the reference images.
While the product must remain 100% identical, the background, surroundings, layout, floor type, walls, lighting style, and overall setting must be changed entirely.
❌ Do NOT replicate or closely match the reference image environment, camera angle, composition, or scene layout.

🚫 BRAND & LOGO REMOVAL - CRITICAL:
1. Remove ALL text from the PRODUCT itself:
   - Brand names, model numbers, manufacturer marks, logos
   - Product labels, serial numbers, company names
   - Replace with CLEAN surfaces matching the product's material
   - The product must be completely TEXT-FREE and BRAND-FREE

2. KEEP realistic environmental text for authenticity:
   ✅ KEEP: Safety signs ("DANGER", "CAUTION", "WARNING", "SAFETY FIRST")
   ✅ KEEP: Directional signs ("EXIT", "ENTRANCE")
   ✅ KEEP: Generic workplace signage

3. REMOVE from environment:
   ❌ REMOVE: Company names, business logos, brand names
   ❌ REMOVE: Phone numbers, websites, email addresses

🎯 FINAL RESULT:
A compelling, photorealistic lifestyle image showing the product being used in its intended real-world application, with appropriate human interaction and environment - professional, authentic, engaging, and completely text-free.
✅ Image must be 1000 × 1000 px (1:1 aspect ratio)."""

    return edit_prompt


def generate_product_image(prompt, reference_images=None, product_title=None, variation="product_in_use"):
    """Generate a product image using Gemini with the proven gemini_service.py prompt approach.
    
    When reference images are provided, uses the comprehensive edit prompt with smart
    category detection (LIFESTYLE vs INDUSTRIAL) for best results.
    Falls back to simple prompt mode when no references available.
    """
    if not gemini_client:
        init_gemini()

    contents = []

    if reference_images:
        # Use the comprehensive edit prompt (proven approach from gemini_service.py)
        title = product_title or "Product"
        edit_prompt = _build_edit_prompt(title, variation=variation, num_ref_images=len(reference_images))

        # Load reference images as PIL Images for Gemini
        pil_images = []
        for ref in reference_images:
            try:
                img = PILImage.open(io.BytesIO(ref["data"]))
                pil_images.append(img)
            except Exception:
                continue

        # Build contents: prompt first, then all reference images
        contents = [edit_prompt]
        contents.extend(pil_images)

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=contents,
        )
    else:
        # No reference images — use simple prompt mode
        contents = [genai.types.Part(text=f"""{prompt}

STYLE: photorealistic, professional ecommerce product photography, high detail, realistic lighting and shadows
IMAGE REQUIREMENTS: square image, 1000 x 1000 px, aspect ratio 1:1, high resolution, sharp and crisp quality, no text, no logo, no watermark
HUMAN RULE: Maximum ONE person if needed. Product must remain the main focus.""")]

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

    raise Exception("No image generated by Gemini")


def generate_images_for_product(listing_data, product_data=None, log_callback=None, image_provider="gemini"):
    """Generate both product images. Each image uses ONE reference image from the gallery.
    Image 1: uses first gallery image as reference.
    Image 2: uses second gallery image as reference.
    Only one reference image is sent to AI at a time — never multiple together.
    Reference product title is also passed to AI for context."""
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
    
    source_images = (product_data or {}).get("images", [])
    ref_title = (product_data or {}).get("title", "")
    
    prompt1 = listing_data.get("image_prompt_1", "")
    prompt2 = listing_data.get("image_prompt_2", "")
    
    for idx, prompt in enumerate([prompt1, prompt2], 1):
        label = "primary" if idx == 1 else "use-case"
        if not prompt:
            if log_callback:
                log_callback(f"⚠️ No image_prompt_{idx} in AI response — skipping image {idx}")
            continue
        
        # Download ONE reference image for this specific generation
        # Image 1 uses first gallery image, Image 2 uses second gallery image
        ref_single = []
        gallery_idx = idx - 1  # 0 for first image, 1 for second
        if gallery_idx < len(source_images):
            ref_url = source_images[gallery_idx]
            if log_callback:
                log_callback(f"📷 Downloading reference image {idx} from gallery (image {gallery_idx + 1}/{len(source_images)})...")
            downloaded = download_reference_images([ref_url], max_images=1)
            if downloaded:
                ref_single = downloaded
                if log_callback:
                    log_callback(f"✅ Reference image {idx} downloaded")
            else:
                if log_callback:
                    log_callback(f"⚠️ Failed to download reference image {idx}")
        else:
            if log_callback:
                log_callback(f"⚠️ No gallery image {gallery_idx + 1} available — generating without reference")
        
        try:
            # Add reference product title to prompt for context
            full_prompt = prompt
            if ref_title:
                full_prompt = f"Reference product: {ref_title}\nMake image with use of this product with related person.\n\n{prompt}"
            
            if log_callback:
                log_callback(f"🎨 Generating image {idx} ({label}) with {provider_name} using {'1 reference' if ref_single else 'no reference'}...")
            
            if image_provider == "openai":
                img = generate_product_image_openai(full_prompt, ref_single if ref_single else None)
            else:
                # Use variation type matching gemini_service.py approach
                var_type = "product_in_use" if idx == 1 else "installation"
                img = generate_product_image(full_prompt, ref_single if ref_single else None, product_title=ref_title, variation=var_type)
            
            images.append({"data": img, "filename": f"product-{label}.png", "type": label})
            if log_callback:
                log_callback(f"✅ Image {idx} generated")
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Image {idx} failed: {e}")
    
    return images
