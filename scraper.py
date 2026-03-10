"""
Scraper module — extracts products from website collection pages.
Uses Shopify JSON API when available, falls back to HTML scraping.
"""

import re
import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}


def get_soup(url, session=None):
    """Fetch URL and return BeautifulSoup object."""
    s = session or requests.Session()
    resp = s.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml"), s


# ─── Shopify JSON API approach (preferred) ───

def try_shopify_json(collection_url, session=None):
    """
    Try to get products via Shopify's /products.json endpoint.
    Returns list of product dicts or None if not a Shopify store.
    """
    s = session or requests.Session()
    parsed = urlparse(collection_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path.rstrip("/")
    
    # Try /collections/xxx/products.json
    json_url = f"{base}{path}/products.json?limit=250"
    
    try:
        resp = s.get(json_url, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if "products" in data:
                return data["products"], s
    except Exception:
        pass
    
    return None, s


def scrape_shopify_product(product_json, base_url, collection_name, website_name):
    """Convert Shopify JSON product data into our standard format."""
    data = {
        "url": f"{base_url}/products/{product_json['handle']}",
        "title": product_json.get("title", ""),
        "description": "",
        "prices": [],
        "images": [],
        "tables": [],
        "variants": [],
        "breadcrumbs": [],
        "full_text": "",
    }
    
    # Description (HTML)
    body_html = product_json.get("body_html", "") or ""
    if body_html:
        soup = BeautifulSoup(body_html, "lxml")
        data["description"] = soup.get_text(separator="\n", strip=True)
        
        # Extract tables from description
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(cells)
            if rows:
                data["tables"].append(rows)
        
        data["full_text"] = data["description"]
    
    # Images
    for img in product_json.get("images", []):
        src = img.get("src", "")
        if src:
            data["images"].append(src)
    
    # Prices
    for variant in product_json.get("variants", []):
        price = variant.get("price")
        if price:
            data["prices"].append(f"£{price}")
    data["prices"] = list(set(data["prices"]))
    
    # Variants — extract option names and values
    options = product_json.get("options", [])
    for opt in options:
        opt_name = opt.get("name", "")
        opt_values = opt.get("values", [])
        if opt_values and not (len(opt_values) == 1 and opt_values[0].lower() in ["default title", "default"]):
            data["variants"].append({
                "label": opt_name,
                "options": opt_values,
            })
    
    # Also store raw variant data for AI to use
    raw_variants = []
    for v in product_json.get("variants", []):
        rv = {
            "title": v.get("title", ""),
            "price": v.get("price", ""),
            "sku": v.get("sku", ""),
            "option1": v.get("option1"),
            "option2": v.get("option2"),
            "option3": v.get("option3"),
        }
        raw_variants.append(rv)
    data["raw_variants"] = raw_variants
    
    # Product type & tags from JSON
    data["product_type"] = product_json.get("product_type", "")
    data["vendor"] = product_json.get("vendor", "")
    data["tags"] = product_json.get("tags", [])
    if isinstance(data["tags"], str):
        data["tags"] = [t.strip() for t in data["tags"].split(",")]
    
    return data


# ─── HTML scraping fallback ───

def extract_product_links(soup, base_url):
    """Extract product links ONLY from the main collection grid.
    Ignores recently viewed, recommended, related products sections."""
    links = set()
    
    # Sections to EXCLUDE
    exclude_patterns = re.compile(
        r"recent|recommend|related|also.like|you.may|featured|trending|best.sell|popular|cross.sell|upsell|viewed|suggested",
        re.I
    )
    
    # Remove excluded sections
    sections_to_remove = []
    for section in soup.find_all(["section", "div", "aside"]):
        try:
            classes = " ".join(section.get("class") or [])
            section_id = section.get("id") or ""
            heading = section.find(["h2", "h3", "h4"])
            heading_text = heading.get_text(strip=True) if heading else ""
            
            if exclude_patterns.search(classes) or exclude_patterns.search(section_id) or exclude_patterns.search(heading_text):
                sections_to_remove.append(section)
        except Exception:
            continue
    
    for section in sections_to_remove:
        try:
            section.decompose()
        except Exception:
            pass
    
    # Extract product links
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(base_url, href)
        if any(p in href.lower() for p in ["/product/", "/products/", "/p/", "/-p-", "/item/"]):
            links.add(full)
    
    return list(links)


def find_next_page(soup, base_url):
    """Find next page URL if pagination exists."""
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True).lower()
        classes = " ".join(a.get("class", [])).lower()
        if "next" in text or "next" in classes or "›" in text or "»" in text:
            return urljoin(base_url, a["href"])
    
    link = soup.find("a", rel="next")
    if link and link.get("href"):
        return urljoin(base_url, link["href"])
    
    return None


def scrape_product_page(url, session=None):
    """Scrape a single product page via HTML and extract all available data."""
    s = session or requests.Session()
    
    # First try Shopify product JSON
    parsed = urlparse(url)
    json_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}.json"
    
    try:
        resp = s.get(json_url, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            pdata = resp.json().get("product")
            if pdata:
                base_url = f"{parsed.scheme}://{parsed.netloc}"
                result = scrape_shopify_product(pdata, base_url, "", "")
                return result, s
    except Exception:
        pass
    
    # Fallback to HTML scraping
    soup, s = get_soup(url, s)
    data = {"url": url}
    
    # Title
    h1 = soup.find("h1")
    data["title"] = h1.get_text(strip=True) if h1 else ""
    
    # Price(s)
    prices = []
    for el in soup.find_all(class_=re.compile(r"price", re.I)):
        text = el.get_text(strip=True)
        found = re.findall(r"£[\d,]+\.?\d*", text)
        prices.extend(found)
    meta_price = soup.find("meta", {"property": "product:price:amount"})
    if meta_price:
        prices.append(f"£{meta_price['content']}")
    data["prices"] = list(set(prices))
    
    # Images
    images = []
    for img in soup.find_all("img", src=True):
        src = urljoin(url, img["src"])
        if any(p in src.lower() for p in ["product", "upload", "media", "image"]):
            images.append(src)
    data["images"] = list(set(images))
    
    # Description
    desc_candidates = soup.find_all(class_=re.compile(r"desc|detail|info|content|tab-pane|product-body", re.I))
    descriptions = []
    for d in desc_candidates:
        text = d.get_text(separator="\n", strip=True)
        if len(text) > 50:
            descriptions.append(text)
    data["description"] = "\n\n".join(descriptions) if descriptions else ""
    
    # Tables
    tables = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)
    data["tables"] = tables
    
    # Variants
    variants = []
    for select in soup.find_all("select"):
        label = ""
        sel_id = select.get("id", "")
        sel_name = select.get("name", "")
        if sel_id:
            lbl = soup.find("label", {"for": sel_id})
            if lbl:
                label = lbl.get_text(strip=True)
        if not label:
            label = sel_name
        options = []
        for opt in select.find_all("option"):
            val = opt.get_text(strip=True)
            if val and val.lower() not in ["select", "choose", "please select", "--"]:
                options.append(val)
        if options:
            variants.append({"label": label, "options": options})
    data["variants"] = variants
    
    # Breadcrumbs
    breadcrumbs = []
    for nav in soup.find_all(class_=re.compile(r"breadcrumb", re.I)):
        for a in nav.find_all("a"):
            breadcrumbs.append(a.get_text(strip=True))
    data["breadcrumbs"] = breadcrumbs
    
    # Full text
    body = soup.find("body")
    data["full_text"] = body.get_text(separator="\n", strip=True)[:15000] if body else ""
    
    return data, s


# ─── Main scrape function ───

def scrape_collection(collection_url, progress_callback=None):
    """Scrape all products from a collection URL. Returns list of product data dicts."""
    session = requests.Session()
    parsed = urlparse(collection_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    
    # Extract collection name from URL
    path = parsed.path.strip("/")
    collection_name = path.split("/")[-1].replace("-", " ").title()
    website_name = parsed.netloc.replace("www.", "")
    
    # ── Try Shopify JSON API first ──
    if progress_callback:
        progress_callback("🔍 Trying Shopify JSON API...")
    
    shopify_products, session = try_shopify_json(collection_url, session)
    
    if shopify_products is not None:
        if progress_callback:
            progress_callback(f"✅ Shopify API found! {len(shopify_products)} products in collection")
        
        products = []
        for i, sp in enumerate(shopify_products):
            if progress_callback:
                progress_callback(f"📦 Processing {i+1}/{len(shopify_products)}: {sp.get('title', 'Unknown')}")
            
            product_data = scrape_shopify_product(sp, base_url, collection_name, website_name)
            products.append(product_data)
        
        return {
            "collection_url": collection_url,
            "collection_name": collection_name,
            "website_name": website_name,
            "total_products": len(products),
            "products": products,
        }
    
    # ── Fallback to HTML scraping ──
    if progress_callback:
        progress_callback("⚠️ Not a Shopify store or JSON API unavailable. Using HTML scraping...")
    
    all_product_links = []
    page_url = collection_url
    page_num = 1
    
    while page_url:
        if progress_callback:
            progress_callback(f"Scanning page {page_num}: {page_url}")
        
        soup, session = get_soup(page_url, session)
        links = extract_product_links(soup, page_url)
        
        new_links = [l for l in links if l not in all_product_links]
        all_product_links.extend(new_links)
        
        if progress_callback:
            progress_callback(f"Found {len(new_links)} products on page {page_num} (total: {len(all_product_links)})")
        
        next_page = find_next_page(soup, page_url)
        if next_page and next_page != page_url:
            page_url = next_page
            page_num += 1
            time.sleep(1)
        else:
            break
    
    # Scrape each product page
    products = []
    for i, link in enumerate(all_product_links):
        if progress_callback:
            progress_callback(f"Scraping product {i+1}/{len(all_product_links)}: {link}")
        
        try:
            product_data, session = scrape_product_page(link, session)
            products.append(product_data)
            time.sleep(0.5)
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error scraping {link}: {e}")
            products.append({"url": link, "error": str(e)})
    
    return {
        "collection_url": collection_url,
        "collection_name": collection_name,
        "website_name": website_name,
        "total_products": len(products),
        "products": products,
    }
