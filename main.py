"""
Lister AI — FastAPI Web Dashboard
Replaces Streamlit for real-time performance.
"""
import os
import csv
import io
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from supabase import create_client
from pydantic import BaseModel
from typing import Optional

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase     = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Lister AI")
import os as _os
if _os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

EBAY_DESCRIPTION = """Shipped primarily with UPS and sometimes USPS. If you have special packing or shipping needs, please send a message.\n\nThis item is sold in "as-is" condition. The seller assumes no liability for the use, operation, or installation of this product. Due to the technical nature of this equipment, the buyer is responsible for having the item professionally inspected and installed by a certified technician prior to use."""

def photo_url(photo_id: str, thumb: bool = False) -> str:
    if not photo_id or photo_id in ("", "nan", "0"):
        return ""
    if thumb:
        return f"{SUPABASE_URL}/storage/v1/render/image/public/part-photos/{photo_id}?width=500&height=500&resize=cover&quality=80"
    return f"{SUPABASE_URL}/storage/v1/object/public/part-photos/{photo_id}"

# ── PAGES ─────────────────────────────────────────────────────── #

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ── API: LISTINGS ─────────────────────────────────────────────── #

@app.get("/api/listings")
async def get_listings():
    try:
        res = supabase.table("listings")\
            .select("*")\
            .neq("status", "archived")\
            .order("created_at", desc=True)\
            .execute()
        listings = res.data or []

        # Batch fetch all group photos for these listings
        primary_pids = [str(l.get("photo_id") or "") for l in listings if l.get("photo_id")]
        group_photo_map = {}  # photo_id -> [all photo_ids in same group]
        if primary_pids:
            try:
                gp_res = supabase.table("group_photos")                    .select("group_id, photo_id")                    .in_("photo_id", primary_pids[:100])                    .execute()
                # Map primary photo -> group_id
                pid_to_gid = {row["photo_id"]: row["group_id"] for row in (gp_res.data or [])}
                group_ids = list(set(pid_to_gid.values()))
                if group_ids:
                    all_gp = supabase.table("group_photos")                        .select("group_id, photo_id")                        .in_("group_id", group_ids)                        .execute()
                    # Build group_id -> [photo_ids]
                    gid_to_photos = {}
                    for row in (all_gp.data or []):
                        gid_to_photos.setdefault(row["group_id"], []).append(row["photo_id"])
                    # Map primary photo_id -> all photos in its group
                    for pid, gid in pid_to_gid.items():
                        group_photo_map[pid] = gid_to_photos.get(gid, [pid])
            except Exception:
                pass

        for l in listings:
            pid = str(l.get("photo_id") or "")
            all_photos = group_photo_map.get(pid, [pid] if pid else [])
            l["thumb_url"]  = photo_url(pid, thumb=True)
            l["full_url"]   = photo_url(pid)
            l["all_photos"] = [{"thumb": photo_url(p, thumb=True), "full": photo_url(p)} for p in all_photos if p]
            # Coerce types
            l["price"]      = float(l.get("price") or 0)
            l["price_used"] = float(l.get("price_used") or 0)
            l["price_new"]  = float(l.get("price_new") or 0)
            l["quantity"]   = int(l.get("quantity") or 1)
            # Normalize condition — default to "used" if blank/null
            cond = str(l.get("condition") or "").strip().lower()
            l["condition"] = cond if cond in ("new", "used") else "used"
            # Normalize listing_type — items from batch upload are not auctions
            lt = str(l.get("listing_type") or "").strip().lower()
            if lt not in ("auction", "fixed"):
                l["listing_type"] = "fixed"
        return JSONResponse(listings)
    except Exception as e:
        raise HTTPException(500, str(e))

class UpdateField(BaseModel):
    field: str
    value: object

@app.patch("/api/listings/{item_id}")
async def update_listing(item_id: str, body: UpdateField):
    try:
        supabase.table("listings").update({body.field: body.value}).eq("id", item_id).execute()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/listings/{item_id}/rescan")
async def rescan_listing(item_id: str):
    try:
        res = supabase.table("listings").select("photo_id").eq("id", item_id).limit(1).execute()
        pid = (res.data[0].get("photo_id", "") if res.data else "")
        if pid:
            grp = supabase.table("group_photos").select("group_id").eq("photo_id", pid).limit(1).execute()
            if grp.data:
                gid = grp.data[0]["group_id"]
                supabase.table("listing_groups").update({"status": "pending"}).eq("id", gid).execute()
        supabase.table("listings").update({"status": "pending", "title": "Scanning..."}).eq("id", item_id).execute()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/listings/archive-batch")
async def archive_batch():
    try:
        res = supabase.table("listings").select("*").neq("status", "archived").execute()
        items = res.data or []
        if items:
            ids = [str(i["id"]) for i in items]
            supabase.table("listings").update({"status": "archived"}).in_("id", ids).execute()
        return {"ok": True, "count": len(items)}
    except Exception as e:
        raise HTTPException(500, str(e))

# ── API: BATCH UPLOAD ─────────────────────────────────────────── #

class CreateGroup(BaseModel):
    session_id: str
    condition:  str

@app.post("/api/groups")
async def create_group(body: CreateGroup):
    try:
        res = supabase.table("listing_groups").insert({
            "session_id": body.session_id,
            "status":     "waiting",
            "quantity":   1,
            "condition":  body.condition,
        }).execute()
        import traceback
        print(f"Group insert result: {res}")
        data = res.data
        gid = data[0]["id"] if isinstance(data, list) and data else (data.get("id") if isinstance(data, dict) else None)
        if not gid:
            raise Exception(f"No ID returned. data={data}")
        return {"id": gid}
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))

class SubmitGroup(BaseModel):
    group_id:  str
    condition: str
    quantity:  int

@app.post("/api/groups/submit")
async def submit_group(body: SubmitGroup):
    try:
        supabase.table("listing_groups").update({
            "condition": body.condition,
            "quantity":  body.quantity,
            "status":    "pending",
        }).eq("id", body.group_id).execute()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/photos/upload")
async def upload_photo(request: Request):
    try:
        form     = await request.form()
        file     = form["file"]
        gid      = str(form["group_id"])
        idx      = int(form.get("index", 0))
        contents = await file.read()
        dt  = datetime.now()
        fn  = f"{dt.strftime('%d%m%y')}_{dt.strftime('%H%M%S')}_{idx}.jpg"
        print(f"Uploading photo: {fn}, size={len(contents)}, group={gid}")
        supabase.storage.from_("part-photos").upload(
            path=fn,
            file=contents,
            file_options={"content-type": "image/jpeg", "upsert": "true"}
        )
        supabase.table("group_photos").insert({"group_id": gid, "photo_id": fn}).execute()
        return {"ok": True, "photo_id": fn, "url": photo_url(fn, thumb=True)}
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))

# ── API: AUCTION ──────────────────────────────────────────────── #

class ScanAuction(BaseModel):
    url: str

@app.post("/api/auction/scan")
async def scan_auction(body: ScanAuction):
    try:
        import uuid, threading
        session_id = str(uuid.uuid4())[:8]
        # Store session first
        supabase.table("auction_sessions").insert({
            "session_id":    session_id,
            "source_url":    body.url,
            "label":         body.url.split("/")[-1][:40] or "Scan",
            "item_count":    0,
            "status":        "active",
        }).execute()

        def run_scan():
            try:
                from auction_scraper import scrape_and_store
                ids = scrape_and_store(body.url, session_id, [1])
                supabase.table("auction_sessions").update({
                    "item_count": len(ids),
                }).eq("session_id", session_id).execute()
                # Enrich in same thread (auction_worker will handle it if deployed)
                try:
                    from auction_scraper import enrich_values
                    enrich_values(ids)
                except Exception as e:
                    print(f"Enrich error: {e}")
            except Exception as e:
                print(f"Scan error: {e}")

        t = threading.Thread(target=run_scan, daemon=True)
        t.start()
        return {"ok": True, "session_id": session_id, "count": 0}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/auction/sessions")
async def get_sessions():
    try:
        res = supabase.table("auction_sessions").select("*").order("created_at", desc=True).execute()
        return JSONResponse(res.data or [])
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/auction/items/{session_id}")
async def get_items(session_id: str):
    try:
        res = supabase.table("auction_items").select("*").eq("session_id", session_id).order("scraped_at").execute()
        return JSONResponse(res.data or [])
    except Exception as e:
        raise HTTPException(500, str(e))

@app.patch("/api/auction/items/{item_id}/favorite")
async def toggle_favorite(item_id: str):
    try:
        cur = supabase.table("auction_items").select("favorited").eq("id", item_id).single().execute()
        new_val = not bool(cur.data.get("favorited", False))
        supabase.table("auction_items").update({"favorited": new_val}).eq("id", item_id).execute()
        return {"favorited": new_val}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.patch("/api/auction/sessions/{session_id}/archive")
async def archive_session(session_id: str):
    try:
        supabase.table("auction_sessions").update({"status": "archived"}).eq("session_id", session_id).execute()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, str(e))

# ── DOWNLOAD: eBay CSV ────────────────────────────────────────── #

@app.get("/api/export/ebay-csv")
async def export_ebay_csv():
    try:
        res = supabase.table("listings").select("*").neq("status", "archived").execute()
        listings = res.data or []

        output = io.StringIO()
        output.write('#INFO,Version=0.0.2,Template= eBay-draft-listings-template_US,,,,,,,,\n')
        output.write('#INFO Action and Category ID are required fields.,,,,,,,,,,\n')
        output.write('#INFO,,,,,,,,,,\n')
        output.write('Action(SiteID=US|Country=US|Currency=USD|Version=1193|CC=UTF-8),Custom label (SKU),Category ID,Title,UPC,Price,Quantity,Item photo URL,Condition ID,Description,Format\n')
        writer = csv.writer(output, quoting=csv.QUOTE_ALL)

        for l in listings:
            cond = str(l.get("condition") or "used").strip().lower()
            if cond not in ("new", "used"):
                cond = "used"
            cond_id = "1000" if cond == "new" else "3000"
            pid = str(l.get("photo_id","") or "")
            pic = photo_url(pid) if pid else ""
            writer.writerow([
                "Draft", "",
                str(l.get("ebay_category_id","") or "").replace(".0",""),
                str(l.get("title",""))[:80],
                "",
                f"{float(l.get('price',0) or 0):.2f}",
                str(int(l.get("quantity",1) or 1)),
                pic, cond_id,
                EBAY_DESCRIPTION,
                "BestOfferEnabled",
            ])

        csv_bytes = output.getvalue().encode("utf-8")
        fn = f"listerai_ebay_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        return StreamingResponse(io.BytesIO(csv_bytes),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={fn}"})
    except Exception as e:
        raise HTTPException(500, str(e))


# ── API: PARTS LOOKUP ────────────────────────────────────────── #

@app.get("/api/parts/photos")
async def get_unmatched_photos():
    """Get all photos from storage that haven't been matched yet."""
    try:
        # Get all files in part-photos bucket
        res = supabase.storage.from_("part-photos").list()
        files = [f["name"] for f in (res or []) if f.get("name") and not f["name"].startswith(".")]
        return {"photos": files, "count": len(files)}
    except Exception as e:
        raise HTTPException(500, str(e))

class ScanPartsBody(BaseModel):
    part_numbers: list
    photo_ids:    list
    gemini_key:   Optional[str] = None

@app.post("/api/parts/scan")
async def scan_parts(body: ScanPartsBody):
    """
    Scan a batch of photos through Gemini Vision.
    For each photo, extract any visible part numbers and check against the list.
    Returns matches with confidence.
    """
    import threading
    gemini_key = body.gemini_key or os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        raise HTTPException(400, "Gemini API key required")
    if not body.part_numbers:
        raise HTTPException(400, "No part numbers provided")

    results = []
    part_set = [str(p).strip().upper() for p in body.part_numbers if str(p).strip()]

    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-1.5-pro")
    except Exception:
        try:
            from google import genai as genai2
            from google.genai import types
            client = genai2.Client(api_key=gemini_key)
        except Exception as e:
            raise HTTPException(500, f"Gemini init failed: {e}")

    for photo_id in body.photo_ids[:50]:  # max 50 at a time
        try:
            # Download photo from Supabase
            img_bytes = supabase.storage.from_("part-photos").download(photo_id)
            if not img_bytes:
                continue

            # Build prompt
            parts_list = "\n".join(part_set[:200])
            prompt = f"""Examine this image carefully. 
Read ALL visible text including: part numbers, model numbers, serial numbers, labels, stamps, engravings, stickers, tags.

I am looking for matches to this list of part numbers:
{parts_list}

Return ONLY a JSON object:
{{
  "visible_text": ["list", "of", "all", "text", "you", "can", "read"],
  "part_numbers_found": ["any", "part", "numbers", "you", "see"],
  "matches": ["part numbers that exactly or closely match the search list"],
  "confidence": "high/medium/low",
  "notes": "brief note on what you see"
}}

If no text visible or no matches, still return the JSON with empty arrays."""

            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel("gemini-1.5-pro")
                import PIL.Image
                import io
                img = PIL.Image.open(io.BytesIO(img_bytes))
                response = model.generate_content([prompt, img])
                raw = response.text or ""
            except Exception:
                try:
                    from google import genai as gc
                    from google.genai import types as gt
                    cl = gc.Client(api_key=gemini_key)
                    models = [m.name for m in cl.models.list()]
                    best = next((m for m in models if "gemini-2.5" in m or "gemini-2.0" in m), models[0] if models else "models/gemini-1.5-pro")
                    resp = cl.models.generate_content(
                        model=best,
                        contents=[gt.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"), prompt]
                    )
                    raw = resp.text or ""
                except Exception as e2:
                    results.append({"photo_id": photo_id, "error": str(e2), "matches": []})
                    continue

            import re, json
            raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\n?```$", "", raw).strip()
            jm = re.search(r'\{.*\}', raw, re.DOTALL)
            if jm:
                data = json.loads(jm.group())
                results.append({
                    "photo_id":        photo_id,
                    "url":             photo_url(photo_id, thumb=True),
                    "full_url":        photo_url(photo_id),
                    "visible_text":    data.get("visible_text", []),
                    "part_numbers_found": data.get("part_numbers_found", []),
                    "matches":         data.get("matches", []),
                    "confidence":      data.get("confidence", ""),
                    "notes":           data.get("notes", ""),
                    "has_match":       len(data.get("matches", [])) > 0,
                })
            else:
                results.append({"photo_id": photo_id, "matches": [], "notes": "Could not parse response"})

        except Exception as e:
            results.append({"photo_id": photo_id, "error": str(e), "matches": []})

    matches    = [r for r in results if r.get("has_match")]
    no_matches = [r for r in results if not r.get("has_match")]
    return {
        "results":     results,
        "matches":     matches,
        "no_matches":  no_matches,
        "match_count": len(matches),
        "scanned":     len(results),
    }

# ── SETTINGS ──────────────────────────────────────────────────── #

@app.get("/api/settings")
async def get_settings():
    try:
        res = supabase.table("app_settings").select("*").execute()
        return {row["key"]: row["value"] for row in (res.data or [])}
    except Exception:
        return {}

class SaveSetting(BaseModel):
    key:   str
    value: str

@app.post("/api/settings")
async def save_setting(body: SaveSetting):
    try:
        supabase.table("app_settings").upsert({"key": body.key, "value": body.value}).execute()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, str(e))
