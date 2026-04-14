"""
Lister AI v2 — Scanner Worker
Pipeline: Claude (identification) → eBay API (category) → Gemini (pricing)
"""

import os, io, re, json, time, base64, uuid
from datetime import datetime

import requests
from dotenv import load_dotenv
from supabase import create_client
import anthropic
from google import genai
from google.genai import types

try:
    from PIL import Image, ImageOps
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

load_dotenv()

# ── credentials ───────────────────────────────────────────────────────────────
SUPABASE_URL      = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY      = os.getenv("SUPABASE_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
EBAY_APP_ID       = os.getenv("EBAY_APP_ID", "")
EBAY_CERT_ID      = os.getenv("EBAY_CERT_ID", "")

missing = [k for k, v in {
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_KEY": SUPABASE_KEY,
    "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
    "GEMINI_API_KEY": GEMINI_API_KEY,
}.items() if not v]
if missing:
    print(f"❌ Missing env vars: {', '.join(missing)}")
    exit(1)

supabase       = create_client(SUPABASE_URL, SUPABASE_KEY)
claude_client  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
gemini_client  = genai.Client(api_key=GEMINI_API_KEY)

# ── seen files tracking ───────────────────────────────────────────────────────
_seen_cache: set = set()

def load_seen():
    global _seen_cache
    try:
        result = supabase.table("seen_files").select("filename").execute()
        _seen_cache = {r["filename"] for r in (result.data or [])}
        print(f"📋 Loaded {len(_seen_cache)} seen files from Supabase")
    except Exception as e:
        print(f"⚠️  Could not load seen files: {e}")

def mark_seen(filename: str):
    if filename in _seen_cache:
        return
    _seen_cache.add(filename)
    try:
        supabase.table("seen_files").upsert({"filename": filename}).execute()
    except Exception:
        pass

# ── image utilities ───────────────────────────────────────────────────────────
def to_jpeg_bytes(raw: bytes) -> bytes:
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    except Exception as e:
        print(f"   ⚠️  Image conversion failed: {e}")
        return raw

def to_base64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("utf-8")

def parse_num(val) -> float:
    try:
        return round(float(re.sub(r"[^0-9.]", "", str(val))), 2)
    except (ValueError, TypeError):
        return 0.0

def safe_json(text: str) -> dict:
    """Extract JSON from text that may have surrounding content."""
    if not text:
        return {}
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```$", "", text).strip()
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}

# ── eBay OAuth token ──────────────────────────────────────────────────────────
_ebay_token = None
_ebay_token_expiry = 0

def get_ebay_token() -> str:
    global _ebay_token, _ebay_token_expiry
    if _ebay_token and time.time() < _ebay_token_expiry - 60:
        return _ebay_token
    if not EBAY_APP_ID or not EBAY_CERT_ID:
        return ""
    try:
        creds = base64.b64encode(f"{EBAY_APP_ID}:{EBAY_CERT_ID}".encode()).decode()
        resp  = requests.post(
            "https://api.ebay.com/identity/v1/oauth2/token",
            headers={
                "Authorization": f"Basic {creds}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data="grant_type=client_credentials&scope=https://api.ebay.com/oauth/api_scope",
            timeout=10,
        )
        data = resp.json()
        _ebay_token        = data.get("access_token", "")
        expires_in         = int(data.get("expires_in", 7200))
        _ebay_token_expiry = time.time() + expires_in
        return _ebay_token
    except Exception as e:
        print(f"   ⚠️  eBay token error: {e}")
        return ""

# ── eBay category lookup ──────────────────────────────────────────────────────
def lookup_ebay_category(title: str, hint: str = "") -> tuple[str, str]:
    """Returns (category_id, category_name). Falls back to hint if API fails."""
    token = get_ebay_token()
    if not token:
        return "12576", hint or "Business & Industrial"
    query = title[:80]
    for attempt in range(2):
        try:
            resp = requests.get(
                "https://api.ebay.com/commerce/taxonomy/v1/category_tree/0/get_category_suggestions",
                params={"q": query},
                headers={"Authorization": f"Bearer {token}"},
                timeout=8,
            )
            if resp.status_code == 401:
                _ebay_token = None
                token = get_ebay_token()
                continue
            if resp.status_code != 200:
                break
            suggestions = resp.json().get("categorySuggestions", [])
            if suggestions:
                cat = suggestions[0]["category"]
                cat_id   = str(cat.get("categoryId", "12576"))
                cat_name = cat.get("categoryName", "")
                ancestry = suggestions[0].get("categoryTreeNodeAncestors", [])
                if ancestry:
                    path = " > ".join(a.get("categoryName", "") for a in reversed(ancestry))
                    cat_name = f"{path} > {cat_name}" if path else cat_name
                return cat_id, cat_name
            break
        except Exception as e:
            print(f"   ⚠️  eBay category lookup failed: {e}")
            break
    return "12576", hint or "Business & Industrial"

# ── Claude identification ─────────────────────────────────────────────────────
CLAUDE_PROMPT = """You are scanning a product for eBay listing. Analyze these {n} photo(s).

PRIORITY 1 — Read every single piece of text visible:
- Part numbers, model numbers, serial numbers (e.g. "A75-3275", "1756-OF4", "P182050")
- Brand names and manufacturer names
- Specifications: voltage, amperage, size, weight, pressure ratings
- Any alphanumeric codes on labels, stamps, or engravings
- Even partially visible text — read what you can

PRIORITY 2 — Identify the item precisely:
- What is this product called?
- What industry/application is it for?
- What makes this specific (size, rating, material)?

PRIORITY 3 — Write an eBay title:
- Start with the brand name
- Include the exact model/part number
- Include the item type
- Include key specs
- Under 80 characters, keyword-rich

Return ONLY a raw JSON object, no markdown, no backticks, nothing else:
{{
  "title": "Brand PartNumber ItemType KeySpec [under 80 chars]",
  "brand": "exact brand name from label",
  "model": "exact model or part number from label",
  "item_type": "what this item is",
  "ebay_category_hint": "most specific eBay category path",
  "weight_oz": 0,
  "weight_lb": 0
}}

IMPORTANT: If you see ANY alphanumeric code on the item, it goes in the title. Never describe generically when a specific model number is visible. "Meritor A75-3275S1059" is better than "Slack Adjuster". "Allen-Bradley 1756-OF4" is better than "PLC Module"."""

def identify_with_claude(image_parts: list[dict], photo_count: int) -> dict:
    """Send photos to Claude for identification. Returns parsed dict."""
    prompt = CLAUDE_PROMPT.format(n=photo_count)
    content = image_parts + [{"type": "text", "text": prompt}]
    for attempt in range(3):
        try:
            response = claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": content}],
            )
            text = response.content[0].text if response.content else ""
            data = safe_json(text)
            if data.get("title"):
                # If no model number found, do a focused retry scan
                if not data.get("model") or len(data.get("model","")) < 3:
                    print(f"   🔍 No model number detected, doing focused text scan...")
                    try:
                        retry_prompt = 'Look very carefully. Read every alphanumeric code, number, and text on the product, label, or tag. Return ONLY JSON: {"text_found": "all text you can read", "model_number": "most likely part or model number", "brand": "brand name"}'
                        retry_resp = claude_client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=256,
                            messages=[{"role": "user", "content": image_parts + [{"type": "text", "text": retry_prompt}]}],
                        )
                        rd = safe_json(retry_resp.content[0].text if retry_resp.content else "")
                        if rd.get("model_number") and len(rd.get("model_number","")) > 3:
                            brand = rd.get("brand") or data.get("brand","")
                            model = rd["model_number"]
                            itype = data.get("item_type","")
                            data["title"] = f"{brand} {model} {itype}".strip()[:80]
                            data["model"] = model
                            print(f"   🔍 Refined title: {data['title']}")
                    except Exception:
                        pass
                return data
            print(f"   ⚠️  Claude returned empty title, attempt {attempt+1}/3")
        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"   ⏳ Claude rate limit, waiting {wait}s...")
            time.sleep(wait)
        except anthropic.APIError as e:
            print(f"   ⚠️  Claude API error: {e}")
            if attempt < 2:
                time.sleep(5)
    return {}

# ── Gemini pricing ────────────────────────────────────────────────────────────
GEMINI_PRICING_PROMPT = """You are a pricing expert for eBay resellers.

Find current eBay market prices for this item:
Title: "{title}"
Condition being sold: {condition}

Search eBay for:
1. Recently SOLD listings (completed sales) — most important
2. Active Buy It Now listings

Return ONLY a raw JSON object, no markdown, no backticks:
{{
  "price_used": recommended listing price for used/pre-owned condition as number,
  "price_new": recommended listing price for new/sealed condition as number,
  "price_used_low": lowest used price found as number,
  "price_used_high": highest used price found as number,
  "price_new_low": lowest new price found as number,
  "price_new_high": highest new price found as number
}}

Use 0 if no prices found for a condition. Base recommendations on SOLD listings, not just active."""

def get_gemini_pricing(title: str, condition: str) -> dict:
    """Use Gemini with web search to find eBay pricing."""
    prompt = GEMINI_PRICING_PROMPT.format(title=title, condition=condition)
    for attempt in range(3):
        try:
            cfg = types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
            response = gemini_client.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=[prompt],
                config=cfg,
            )
            raw = ""
            if response.text:
                raw = response.text
            elif response.candidates:
                for cand in response.candidates:
                    for part in (getattr(cand.content, "parts", None) or []):
                        t = getattr(part, "text", None)
                        if t:
                            raw += t
            data = safe_json(raw)
            if any(data.get(k, 0) > 0 for k in ["price_used", "price_new"]):
                return data
            print(f"   ⚠️  Gemini returned no prices, attempt {attempt+1}/3")
            time.sleep(5)
        except Exception as e:
            err = str(e)
            if "503" in err or "UNAVAILABLE" in err:
                wait = 15 * (attempt + 1)
                print(f"   ⏳ Gemini overloaded, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"   ⚠️  Gemini error: {e}")
                break
    return {}

# ── process one group ─────────────────────────────────────────────────────────
def process_group(group: dict):
    group_id  = group["id"]
    condition = group.get("condition", "used")
    quantity  = int(group.get("quantity", 1) or 1)
    print(f"\n📦 Processing group {group_id} — condition: {condition}, qty: {quantity}")

    supabase.table("listing_groups").update({"status": "processing"}).eq("id", group_id).execute()

    photos_result = (
        supabase.table("group_photos")
        .select("*")
        .eq("group_id", group_id)
        .order("uploaded_at")
        .execute()
    )

    if not photos_result.data:
        print(f"   ⚠️  No photos found")
        supabase.table("listing_groups").update({"status": "error"}).eq("id", group_id).execute()
        return

    image_parts   = []
    primary_photo = None
    scanned_at    = datetime.now().isoformat()

    for i, record in enumerate(photos_result.data):
        old_name = record["photo_id"]
        try:
            raw   = supabase.storage.from_("part-photos").download(old_name)
            jpg   = to_jpeg_bytes(raw)
            ts    = datetime.now().strftime("%d%m%y_%H%M%S")
            new_name = f"{ts}_{i}.jpg"
            try:
                supabase.storage.from_("part-photos").upload(
                    path=new_name, file=jpg,
                    file_options={"content-type": "image/jpeg", "upsert": "true"},
                )
                supabase.storage.from_("part-photos").remove([old_name])
                supabase.table("group_photos").update({"photo_id": new_name}).eq("id", record["id"]).execute()
                mark_seen(old_name)
                mark_seen(new_name)
                final_name = new_name
            except Exception:
                final_name = old_name

            if i == 0:
                primary_photo = final_name

            image_parts.append({
                "type": "image",
                "source": {
                    "type":       "base64",
                    "media_type": "image/jpeg",
                    "data":       to_base64(jpg),
                },
            })
            time.sleep(0.5)
        except Exception as e:
            print(f"   ⚠️  Photo {old_name} failed: {e}")

    if not image_parts:
        print(f"   ⚠️  No images loaded")
        supabase.table("listing_groups").update({"status": "error"}).eq("id", group_id).execute()
        return

    # Step 1: Claude identification
    print(f"   🤖 Step 1: Claude identifying item from {len(image_parts)} photo(s)...")
    claude_data = identify_with_claude(image_parts, len(image_parts))
    title       = claude_data.get("title", "Unknown Item").strip()[:80]
    cat_hint    = claude_data.get("ebay_category_hint", "")
    weight_oz   = parse_num(claude_data.get("weight_oz", 0))
    weight_lb   = parse_num(claude_data.get("weight_lb", 0))
    print(f"   ✅ Claude: {title}")

    # Step 2: eBay category
    print(f"   🏷️  Step 2: eBay category lookup...")
    ebay_category_id, ebay_category = "12576", cat_hint or "Business & Industrial"
    if title != "Unknown Item":
        ebay_category_id, ebay_category = lookup_ebay_category(title, cat_hint)
    print(f"   ✅ Category: {ebay_category} (ID: {ebay_category_id})")

    # Step 3: Gemini pricing
    price_used = price_new = 0.0
    price_used_low = price_used_high = 0.0
    price_new_low  = price_new_high  = 0.0
    if title != "Unknown Item":
        print(f"   💰 Step 3: Gemini pricing research...")
        pricing = get_gemini_pricing(title, condition)
        price_used      = parse_num(pricing.get("price_used", 0))
        price_new       = parse_num(pricing.get("price_new", 0))
        price_used_low  = parse_num(pricing.get("price_used_low", 0))
        price_used_high = parse_num(pricing.get("price_used_high", 0))
        price_new_low   = parse_num(pricing.get("price_new_low", 0))
        price_new_high  = parse_num(pricing.get("price_new_high", 0))
        print(f"   ✅ Pricing — used: ${price_used:.2f} / new: ${price_new:.2f}")

    # Determine active price based on condition
    if condition == "used":
        active_price = price_used if price_used > 0 else price_new
        price_note   = "new" if price_used == 0 and price_new > 0 else ""
    else:
        active_price = price_new if price_new > 0 else price_used
        price_note   = "used" if price_new == 0 and price_used > 0 else ""

    # Step 4: Save to listings
    print(f"   💾 Step 4: Saving to Supabase...")
    try:
        insert_result = supabase.table("listings").insert({
            "title":            title,
            "ebay_category":    ebay_category,
            "ebay_category_id": ebay_category_id,
            "weight_oz":        weight_oz,
            "weight_lb":        weight_lb,
            "price":            active_price,
            "price_note":       price_note,
            "price_used":       price_used,
            "price_new":        price_new,
            "photo_id":         primary_photo,
            "condition":        condition,
            "quantity":         quantity,
            "status":           "scanned",
            "created_at":       scanned_at,
        }).execute()
        if not insert_result.data:
            print(f"   ⚠️  Insert returned no data — possible error")
        else:
            print(f"   ✅ Saved: {title}")
    except Exception as e:
        print(f"   ❌ Insert failed: {e}")
        supabase.table("listing_groups").update({"status": "error"}).eq("id", group_id).execute()
        return

    supabase.table("listing_groups").update({"status": "done"}).eq("id", group_id).execute()
    print(f"   ✅ Done: {title} — ${active_price:.2f} | {ebay_category}")

# ── watcher loop ──────────────────────────────────────────────────────────────
def main():
    print("🚀 Lister AI v2 Scanner Worker starting...")
    print(f"   Claude: claude-sonnet-4-20250514")
    print(f"   Gemini: gemini-2.5-flash (pricing)")
    print(f"   eBay:   {'OAuth token' if EBAY_APP_ID and EBAY_CERT_ID else 'APP_ID only (no cert)'}")
    load_seen()

    while True:
        try:
            pending = (
                supabase.table("listing_groups")
                .select("*")
                .eq("status", "pending")
                .order("created_at")
                .limit(5)
                .execute()
            )
            for group in (pending.data or []):
                try:
                    process_group(group)
                except Exception as e:
                    print(f"   ❌ Group {group['id']} failed: {e}")
                    supabase.table("listing_groups").update({"status": "error"}).eq("id", group["id"]).execute()
        except Exception as e:
            print(f"⚠️  Watcher error: {e}")
        time.sleep(10)

if __name__ == "__main__":
    main()
