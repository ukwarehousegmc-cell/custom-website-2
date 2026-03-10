"""
AI Generator — uses OpenAI to create product listings following the rules document.
Also generates product images via DALL-E.
"""

import os
import json
import base64
import requests
from openai import OpenAI

client = None

def init_openai():
    global client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


LISTING_SYSTEM_PROMPT = """You are a Shopify product listing expert for a UK industrial store.
You create professional product listings from scraped product data.

STRICT RULES:
1. TITLE: Main Keyword – Size (if needed) – Colour – Important Detail – Use Case
   If multiple sizes/colours exist as variants, don't put them in title.

2. DESCRIPTION in this exact order:
   - Short Description (3-4 lines)
   - Features (bullet points from reference data)
   - Benefits (customer advantages)
   - Use Cases
   - FAQ (2-3 Q&As)
   - Specifications / Technical Details

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
  "body_html": "<h3>Short Description</h3>...<h3>Features</h3><ul>...</ul><h3>Benefits</h3>...<h3>Use Cases</h3>...<h3>FAQ</h3>...<h3>Specifications</h3>...",
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
- Generate detailed image prompts following the image rules (real-life use, 1000x1000, no text, no logos, no installation scenes)
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


def generate_product_image(prompt, size="1024x1024"):
    """Generate a product image using GPT Image 1 (gpt-image-1)."""
    if not client:
        init_openai()

    response = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=size,
        quality="high",
        n=1,
    )

    # gpt-image-1 returns base64 data
    image_b64 = response.data[0].b64_json
    if image_b64:
        return base64.b64decode(image_b64)
    
    # Fallback to URL if provided
    image_url = response.data[0].url
    img_response = requests.get(image_url, timeout=60)
    img_response.raise_for_status()
    return img_response.content


def generate_images_for_product(listing_data):
    """Generate both product images and return as list of image bytes."""
    images = []
    
    prompt1 = listing_data.get("image_prompt_1", "")
    prompt2 = listing_data.get("image_prompt_2", "")
    
    if prompt1:
        try:
            img1 = generate_product_image(prompt1)
            images.append({"data": img1, "filename": "product-main.png", "type": "primary"})
        except Exception as e:
            print(f"Error generating image 1: {e}")
    
    if prompt2:
        try:
            img2 = generate_product_image(prompt2)
            images.append({"data": img2, "filename": "product-usecase.png", "type": "use-case"})
        except Exception as e:
            print(f"Error generating image 2: {e}")
    
    return images
