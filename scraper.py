"""
Scraper module — extracts products from any website collection page.
Uses requests + BeautifulSoup. Handles pagination.
"""

import re
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


def extract_product_links(soup, base_url):
    """Extract product links ONLY from the main collection grid.
    Ignores recently viewed, recommended, related products sections."""
    links = set()
    
    # Sections to EXCLUDE — recently viewed, recommended, related, etc.
    exclude_patterns = re.compile(
        r"recent|recommend|related|also.like|you.may|featured|trending|best.sell|popular|cross.sell|upsell|viewed|suggested",
        re.I
    )
    
    # Remove excluded sections from soup before extracting
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
    
    # Now extract product links from the cleaned page
    # First try: find main collection/product grid container
    main_grid = soup.find(class_=re.compile(
        r"collection.?product|product.?grid|product.?list|collection.?grid|category.?product|main.?product|product.?container",
        re.I
    ))
    
    search_area = main_grid if main_grid else soup
    
    for a in search_area.find_all("a", href=True):
        href = a["href"]
        full = urljoin(base_url, href)
        if any(p in href.lower() for p in ["/product/", "/products/", "/p/", "/-p-", "/item/"]):
            links.add(full)
    
    # Fallback: if no product links found in grid, search broader but still exclude junk
    if not links:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full = urljoin(base_url, href)
            if any(p in href.lower() for p in ["/product/", "/products/", "/p/", "/-p-", "/item/"]):
                links.add(full)
    
    # Last resort: look in card/item containers
    if not links:
        containers = soup.find_all(class_=re.compile(
            r"product|item|card|grid-item|listing", re.I
        ))
        for container in containers:
            for a in container.find_all("a", href=True):
                full = urljoin(base_url, a["href"])
                if full != base_url and urlparse(full).netloc == urlparse(base_url).netloc:
                    links.add(full)
    
    return list(links)


def find_next_page(soup, base_url):
    """Find next page URL if pagination exists."""
    # Look for next page link
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True).lower()
        classes = " ".join(a.get("class", [])).lower()
        if "next" in text or "next" in classes or "›" in text or "»" in text:
            return urljoin(base_url, a["href"])
    
    # Check for rel="next"
    link = soup.find("a", rel="next")
    if link and link.get("href"):
        return urljoin(base_url, link["href"])
    
    return None


def scrape_product_page(url, session=None):
    """Scrape a single product page and extract all available data."""
    soup, session = get_soup(url, session)
    data = {"url": url, "raw_html": str(soup)}
    
    # Title
    h1 = soup.find("h1")
    data["title"] = h1.get_text(strip=True) if h1 else ""
    
    # Price(s)
    prices = []
    for el in soup.find_all(class_=re.compile(r"price", re.I)):
        text = el.get_text(strip=True)
        found = re.findall(r"£[\d,]+\.?\d*", text)
        prices.extend(found)
    # Also check meta
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
    # Also data-src, data-zoom
    for img in soup.find_all(attrs={"data-src": True}):
        images.append(urljoin(url, img["data-src"]))
    for img in soup.find_all(attrs={"data-zoom-image": True}):
        images.append(urljoin(url, img["data-zoom-image"]))
    data["images"] = list(set(images))
    
    # Description
    desc_candidates = soup.find_all(class_=re.compile(r"desc|detail|info|content|tab-pane|product-body", re.I))
    descriptions = []
    for d in desc_candidates:
        text = d.get_text(separator="\n", strip=True)
        if len(text) > 50:
            descriptions.append(text)
    data["description"] = "\n\n".join(descriptions) if descriptions else ""
    
    # Specifications / tables
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
    
    # Variants / dropdowns
    variants = []
    for select in soup.find_all("select"):
        label = ""
        sel_id = select.get("id", "")
        sel_name = select.get("name", "")
        # Find associated label
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
    
    # Radio buttons / swatches
    for fieldset in soup.find_all(["fieldset", "div"], class_=re.compile(r"variant|option|swatch", re.I)):
        label_el = fieldset.find(["legend", "label", "span"], class_=re.compile(r"label|title|name", re.I))
        label = label_el.get_text(strip=True) if label_el else ""
        options = []
        for inp in fieldset.find_all("input", {"type": ["radio", "checkbox"]}):
            val = inp.get("value", "") or inp.get("title", "")
            lbl = fieldset.find("label", {"for": inp.get("id", "")})
            if lbl:
                val = lbl.get_text(strip=True)
            if val:
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
    
    # Full page text for AI context
    body = soup.find("body")
    data["full_text"] = body.get_text(separator="\n", strip=True)[:15000] if body else ""
    
    return data, session


def scrape_collection(collection_url, progress_callback=None):
    """Scrape all products from a collection URL. Returns list of product data dicts."""
    session = requests.Session()
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
            time.sleep(1)  # Be polite
        else:
            break
    
    # Now scrape each product
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
    
    # Extract collection name from URL
    path = urlparse(collection_url).path.strip("/")
    collection_name = path.split("/")[-1].replace("-", " ").title()
    
    return {
        "collection_url": collection_url,
        "collection_name": collection_name,
        "website_name": urlparse(collection_url).netloc.replace("www.", ""),
        "total_products": len(products),
        "products": products,
    }
