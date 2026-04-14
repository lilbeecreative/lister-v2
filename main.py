import os, io, uuid, csv, base64
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client

try:
    from PIL import Image, ImageOps
except ImportError:
    pass

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
supabase     = create_client(SUPABASE_URL, SUPABASE_KEY)

app       = FastAPI(title="Lister AI v2")
templates = Jinja2Templates(directory="templates")

EBAY_DESCRIPTION = (
    "Shipped primarily with UPS and sometimes USPS. "
    "If you have special packing or shipping needs, please send a message. "
    "This item is sold in as-is condition. The seller assumes no liability for the use, "
    "operation, or installation of this product. Due to the technical nature of this equipment, "
    "the buyer is responsible for having the item professionally inspected and installed by a "
    "certified technician prior to use."
)

# ── helpers ──────────────────────────────────────────────────────────────────

def fix_rotation(img_bytes: bytes) -> bytes:
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    except Exception:
        return img_bytes

def photo_url(photo_id: str) -> str:
    if not photo_id:
        return ""
    return f"{SUPABASE_URL}/storage/v1/object/public/part-photos/{photo_id}"

# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/groups")
async def create_group(body: dict):
    session_id = body.get("session_id", str(uuid.uuid4()))
    condition  = body.get("condition", "used")
    result = supabase.table("listing_groups").insert({
        "session_id": session_id,
        "status":     "waiting",
        "quantity":   1,
        "condition":  condition,
    }).execute()
    return {"group_id": result.data[0]["id"], "session_id": session_id}


@app.post("/api/groups/{group_id}/photos")
async def upload_photo(group_id: str, file: UploadFile = File(...)):
    contents = await file.read()
    corrected = fix_rotation(contents)
    ts  = datetime.now().strftime("%d%m%y_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    filename = f"{ts}_{uid}.jpg"
    try:
        supabase.storage.from_("part-photos").upload(
            path=filename,
            file=corrected,
            file_options={"content-type": "image/jpeg", "upsert": "true"},
        )
        supabase.table("group_photos").insert({
            "group_id": group_id,
            "photo_id": filename,
        }).execute()
        return {"photo_id": filename, "url": photo_url(filename)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/api/groups/{group_id}/submit")
async def submit_group(group_id: str, body: dict):
    condition = body.get("condition", "used")
    quantity  = int(body.get("quantity", 1))
    supabase.table("listing_groups").update({
        "status":    "pending",
        "condition": condition,
        "quantity":  quantity,
    }).eq("id", group_id).execute()
    return {"ok": True}


@app.get("/api/groups/{group_id}/status")
async def get_group_status(group_id: str):
    result = supabase.table("listing_groups").select("status").eq("id", group_id).execute()
    if not result.data:
        raise HTTPException(404, "Group not found")
    return {"status": result.data[0]["status"]}


@app.get("/api/listings")
async def get_listings():
    result = (
        supabase.table("listings")
        .select("*")
        .eq("status", "scanned")
        .order("created_at", desc=True)
        .execute()
    )
    items = []
    for row in (result.data or []):
        items.append({
            "id":              row.get("id"),
            "title":           row.get("title", ""),
            "ebay_category":   row.get("ebay_category", ""),
            "ebay_category_id":str(row.get("ebay_category_id", "") or ""),
            "price":           float(row.get("price", 0) or 0),
            "price_used":      float(row.get("price_used", 0) or 0),
            "price_new":       float(row.get("price_new", 0) or 0),
            "price_note":      row.get("price_note", ""),
            "condition":       row.get("condition", "used"),
            "quantity":        int(row.get("quantity", 1) or 1),
            "photo_id":        row.get("photo_id", ""),
            "photo_url":       photo_url(row.get("photo_id", "")),
            "created_at":      str(row.get("created_at", "")),
        })
    return {"items": items}


@app.patch("/api/listings/{item_id}")
async def update_listing(item_id: str, body: dict):
    allowed = {"title", "price", "condition", "ebay_category", "ebay_category_id",
               "price_used", "price_new", "price_note", "quantity"}
    update  = {k: v for k, v in body.items() if k in allowed}
    if not update:
        raise HTTPException(400, "No valid fields")
    if "condition" in update:
        cond = update["condition"]
        row = supabase.table("listings").select("price_used,price_new").eq("id", item_id).execute()
        if row.data:
            pu = float(row.data[0].get("price_used", 0) or 0)
            pn = float(row.data[0].get("price_new", 0) or 0)
            if cond == "used":
                update["price"]      = pu if pu > 0 else pn
                update["price_note"] = "new" if pu == 0 and pn > 0 else ""
            else:
                update["price"]      = pn if pn > 0 else pu
                update["price_note"] = "used" if pn == 0 and pu > 0 else ""
    supabase.table("listings").update(update).eq("id", item_id).execute()
    return {"ok": True}


@app.delete("/api/listings/{item_id}")
async def delete_listing(item_id: str):
    supabase.table("listings").delete().eq("id", item_id).execute()
    return {"ok": True}


@app.delete("/api/listings")
async def clear_all_listings():
    supabase.table("listings").delete().eq("status", "scanned").execute()
    return {"ok": True}


@app.get("/api/stats")
async def get_stats():
    result = (
        supabase.table("listings")
        .select("price, status, condition")
        .eq("status", "scanned")
        .execute()
    )
    items   = result.data or []
    total   = len(items)
    value   = sum(float(i.get("price", 0) or 0) for i in items)
    pending = (
        supabase.table("listing_groups")
        .select("id")
        .in_("status", ["pending", "processing"])
        .execute()
    )
    return {
        "total":      total,
        "processing": len(pending.data or []),
        "value":      round(value, 2),
    }


@app.get("/api/export/csv")
async def export_csv():
    result = (
        supabase.table("listings")
        .select("*")
        .eq("status", "scanned")
        .order("created_at", desc=True)
        .execute()
    )
    rows   = result.data or []
    output = io.StringIO()
    output.write("#INFO,Version=0.0.2,Template= eBay-draft-listings-template_US,,,,,,,,\n")
    output.write("#INFO Action and Category ID are required fields.,,,,,,,,,,\n")
    output.write("#INFO,,,,,,,,,,\n")
    output.write("Action(SiteID=US|Country=US|Currency=USD|Version=1193|CC=UTF-8),"
                 "Custom label (SKU),Category ID,Title,UPC,Price,Quantity,"
                 "Item photo URL,Condition ID,Description,Format\n")
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    for row in rows:
        cond   = str(row.get("condition", "used") or "used").lower()
        cond_id = "1000" if cond == "new" else "3000"
        cat_id  = str(row.get("ebay_category_id", "") or "").replace(".0", "") or "12576"
        price   = max(float(row.get("price", 0) or 0), 1.00)
        qty     = int(row.get("quantity", 1) or 1)
        pid     = str(row.get("photo_id", "") or "")
        pic_url = photo_url(pid) if pid else ""
        title   = str(row.get("title", ""))[:80]
        writer.writerow([
            "Draft", "", cat_id, title, "",
            f"{price:.2f}", qty, pic_url,
            cond_id, EBAY_DESCRIPTION, "FixedPrice",
        ])
    fn = f"lister_v2_ebay_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={fn}"},
    )
