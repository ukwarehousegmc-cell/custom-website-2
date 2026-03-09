"""
Shopify Uploader — creates products on Shopify store via REST Admin API.
Uses client_credentials flow for authentication.
"""

import os
import json
import base64
import time
import requests


class ShopifyUploader:
    def __init__(self, store=None, client_id=None, client_secret=None):
        self.store = store or os.getenv("SHOPIFY_STORE")
        self.client_id = client_id or os.getenv("SHOPIFY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SHOPIFY_CLIENT_SECRET")
        self.access_token = None
        self.base_url = f"https://{self.store}/admin/api/2024-10"

    def authenticate(self):
        """Get access token via client_credentials."""
        resp = requests.post(
            f"https://{self.store}/admin/oauth/access_token",
            json={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=30,
        )
        resp.raise_for_status()
        self.access_token = resp.json()["access_token"]
        return self.access_token

    def _headers(self):
        if not self.access_token:
            self.authenticate()
        return {
            "X-Shopify-Access-Token": self.access_token,
            "Content-Type": "application/json",
        }

    def create_product(self, listing_data, images=None):
        """
        Create a product on Shopify from the AI-generated listing data.
        listing_data: dict with title, body_html, product_type, tags, variants, options
        images: list of dicts with 'data' (bytes) and 'filename'
        """
        # Build product payload
        product = {
            "title": listing_data["title"],
            "body_html": listing_data["body_html"],
            "product_type": listing_data.get("product_type", ""),
            "tags": ", ".join(listing_data.get("tags", [])),
            "status": "draft",
        }

        # Options
        if listing_data.get("options"):
            product["options"] = listing_data["options"]

        # Variants
        if listing_data.get("variants"):
            product["variants"] = []
            for v in listing_data["variants"]:
                variant = {
                    "option1": v.get("option1", "Default"),
                    "price": str(v.get("price", "0.00")),
                    "sku": v.get("sku", ""),
                    "inventory_management": None,
                    "inventory_policy": "continue",
                    "requires_shipping": True,
                    "taxable": True,
                }
                if v.get("option2"):
                    variant["option2"] = v["option2"]
                if v.get("option3"):
                    variant["option3"] = v["option3"]
                product["variants"].append(variant)

        # Images (base64 encoded)
        if images:
            product["images"] = []
            for i, img in enumerate(images):
                img_data = {
                    "attachment": base64.b64encode(img["data"]).decode("utf-8"),
                    "filename": img.get("filename", f"product-{i+1}.png"),
                }
                if i == 0:
                    img_data["position"] = 1
                product["images"].append(img_data)

        # Create product
        resp = requests.post(
            f"{self.base_url}/products.json",
            headers=self._headers(),
            json={"product": product},
            timeout=60,
        )

        if resp.status_code == 401:
            # Token expired, re-auth and retry
            self.authenticate()
            resp = requests.post(
                f"{self.base_url}/products.json",
                headers=self._headers(),
                json={"product": product},
                timeout=60,
            )

        resp.raise_for_status()
        created = resp.json()["product"]
        
        return {
            "id": created["id"],
            "title": created["title"],
            "handle": created["handle"],
            "status": created["status"],
            "variants_count": len(created.get("variants", [])),
            "images_count": len(created.get("images", [])),
            "admin_url": f"https://{self.store}/admin/products/{created['id']}",
        }

    def get_product_count(self):
        """Get total product count."""
        resp = requests.get(
            f"{self.base_url}/products/count.json",
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["count"]
