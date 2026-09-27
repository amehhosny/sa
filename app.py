import os
import asyncio
import aiohttp
from aiohttp import web
import json
from datetime import datetime

# التخزين المؤقت للـ Logs والإحصائيات لرؤيتها في لوحة التحكم
logs = []
stats = {
    "total_connections": 0,
    "active_connections": 0,
    "bytes_sent": 0,
    "bytes_received": 0
}

def add_log(message, log_type="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] [{log_type}] {message}"
    logs.append(entry)
    if len(logs) > 100:  # الاحتفاظ بأخر 100 السجلات فقط لمنع استهلاك الذاكرة
        logs.pop(0)

# ----------------- Dashboard Routes -----------------
async def handle_dashboard(request):
    """صفحة لوحة التحكم لعرض البيانات"""
    html_content = """
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <title>Tunnel Dashboard - Roblox Proxy</title>
        <style>
            body { font-family: monospace; background: #121212; color: #00ff66; padding: 20px; }
            h1 { color: #fff; border-bottom: 2px solid #00ff66; padding-bottom: 10px; }
            .stats-container { display: flex; gap: 15px; margin-bottom: 20px; }
            .card { background: #1e1e1e; padding: 15px; border-radius: 8px; border: 1px solid #333; flex: 1; }
            .card h3 { margin: 0 0 10px 0; color: #aaa; font-size: 14px; }
            .card p { margin: 0; font-size: 22px; font-weight: bold; color: #fff; }
            #logs { background: #000; border: 1px solid #333; padding: 15px; height: 350px; overflow-y: scroll; border-radius: 8px; }
            .log-entry { margin-bottom: 5px; font-size: 13px; }
        </style>
    </head>
    <body>
        <h1>لوحة تحكم النفق (Tunnel Control Panel)</h1>
        
        <div class="stats-container">
            <div class="card"><h3>إجمالي الاتصالات</h3><p id="total_conn">0</p></div>
            <div class="card"><h3>الاتصالات النشطة</h3><p id="active_conn">0</p></div>
            <div class="card"><h3>البيانات المستلمة</h3><p id="bytes_rx">0 KB</p></div>
            <div class="card"><h3>البيانات المرسلة</h3><p id="bytes_tx">0 KB</p></div>
        </div>

        <h3>سجل الأحداث (Live Logs)</h3>
        <div id="logs"></div>

        <script>
            async function updateDashboard() {
                try {
                    const res = await fetch('/api/stats');
                    const data = await res.json();
                    
                    document.getElementById('total_conn').innerText = data.stats.total_connections;
                    document.getElementById('active_conn').innerText = data.stats.active_connections;
                    document.getElementById('bytes_rx').innerText = (data.stats.bytes_received / 1024).toFixed(2) + " KB";
                    document.getElementById('bytes_tx').innerText = (data.stats.bytes_sent / 1024).toFixed(2) + " KB";

                    const logsDiv = document.getElementById('logs');
                    logsDiv.innerHTML = data.logs.map(l => `<div class="log-entry">${l}</div>`).join('');
                    logsDiv.scrollTop = logsDiv.scrollHeight;
                } catch(e) {}
            }
            setInterval(updateDashboard, 1500); // تحديث كل ثانية ونصف
        </script>
    </body>
    </html>
    """
    return web.Response(text=html_content, content_type='text/html')

async def handle_stats_api(request):
    """API لإرسال البيانات للوحة التحكم"""
    return web.json_response({"stats": stats, "logs": logs})

# ----------------- Tunnel WebSockets Engine -----------------
async def handle_tunnel(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    stats["total_connections"] += 1
    stats["active_connections"] += 1
    add_log(f"اتصال جديد قادم من الجهاز المحلي: {request.remote}")

    session = aiohttp.ClientSession()

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                stats["bytes_received"] += len(msg.data)
                
                # توجيه البيانات المطلوبة لسيرفرات Roblox
                # هنا يتم التعامل مع الحزم المستلمة
                add_log(f"تم استقبال حزمة بيانات بحجم: {len(msg.data)} bytes")
                
                # إرجاع رد تجريبي للـ Client (أو استهداف السيرفر المطلوب)
                reply = b"ACK_" + msg.data
                await ws.send_bytes(reply)
                stats["bytes_sent"] += len(reply)

            elif msg.type == aiohttp.WSMsgType.ERROR:
                add_log(f"خطأ في اتصال WebSocket: {ws.exception()}", "ERROR")

    finally:
        stats["active_connections"] -= 1
        await session.close()
        add_log("تم إغلاق الاتصال النفق مع الجهاز المحلي.")

    return ws

# ----------------- Start App -----------------
app = web.Application()
app.router.add_get('/', handle_dashboard)
app.router.add_get('/api/stats', handle_stats_api)
app.router.add_get('/ws', handle_tunnel)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    add_log(f"تم تشغيل السيرفر بنجاح على المنفذ {port}")
    web.run_app(app, port=port)
